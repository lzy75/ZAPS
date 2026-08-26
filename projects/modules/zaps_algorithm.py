"""
核心公式（论文 Algorithm 1）：
  观测模型     : y = H(x_0) + n，n ~ N(0, σ_n²·I)
  Tweedie 估计 : x̂_0 = (x_t + (1-ᾱ_t)·s_θ(x_t,t)) / √ᾱ_t
  无条件步     : x'_{t-1} = DDPM_posterior(x_t, x̂_0, t, t_prev)
  引导修正     : δ = ζ_t · (1/√ᾱ_t)·(I + (1−ᾱ_t)·W D_t W^⊤)·H^T(y − H(x̂_0))
                  W D_t W^⊤ = 逆DWT( D_t ⊙ 正DWT(·) )  （论文 Eq.20/22，db4 小波域对角 Hessian）
  完整步       : x_{t-1} = x'_{t-1} + δ
  损失函数     : L = ‖y − H(x_0^est)‖²   （物理引导）
  可学习参数   : {ζ_t}（对数似然权重）, {D_t}（小波域逐系数对角 Hessian 近似）
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.wavelet import OrthogonalDWT2D

# ═══════════════════════════════════════════════════════
# 可调超参数（ZAPS 算法层面）
# ═══════════════════════════════════════════════════════
NUM_SAMPLING_STEPS  = 30            # ← 可调：总采样步数，论文值 30
TIMESTEP_SCHEDULE   = (15, 10, 5)   # ← 可调：低/中/高噪声区间步数分配（论文 "15,10,5"）
TIMESTEP_SPACING    = "linear"      # ← 可调：linear | quadratic
SCHEDULE_POWER      = 2.0           # ← 可调：quadratic/power 非线性取点指数
NUM_EPOCHS          = 10            # ← 可调：零样本优化轮数，论文值 10
LEARNING_RATE       = 5e-3          # ← 可调：Adam 学习率
ZETA_INIT           = 0.2           # ← 可调：ζ 初始值（论文：高斯/运动模糊 0.2，inpaint/SR 0.1）
D_INIT              = 0.2           # ← 可调：D_t 对角初值，论文统一 0.2
WAVELET             = "db4"         # ← 可调：正交小波，论文用 db4
WAVELET_LEVEL       = 3             # ← 可调：DWT 分解级数
TOTAL_DIFFUSION_STEPS = 1000        # 扩散模型总步数（与预训练模型匹配）
# ═══════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────
# 辅助：不规则时间步构建
# ───────────────────────────────────────────────────────

def _segment_timesteps(start: int, end: int, count: int,
                       spacing: str, power: float,
                       include_left: bool, include_right: bool) -> torch.Tensor:
    """按指定 spacing 在闭区间内取点，再按端点开闭裁剪。"""
    raw_count = count + int(not include_left) + int(not include_right)
    u = torch.linspace(0.0, 1.0, raw_count)
    if spacing == "linear":
        warped = u
    elif spacing in ("quadratic", "power"):
        warped = u.pow(power)
    else:
        raise ValueError(f"未知 timestep spacing: {spacing}")
    vals = (start + (end - start) * warped).long()
    if not include_left:
        vals = vals[1:]
    if not include_right:
        vals = vals[:-1]
    return vals


def build_irregular_timesteps(
    total_steps: int = TOTAL_DIFFUSION_STEPS,
    schedule: tuple = TIMESTEP_SCHEDULE,
    spacing: str = TIMESTEP_SPACING,
    schedule_power: float = SCHEDULE_POWER,
) -> torch.Tensor:
    """
    构建不规则时间步子集 τ ⊂ {0,...,T-1}
    低噪声区间（小 t）密集采样，高噪声区间稀疏采样

    参数:
        total_steps : 扩散模型总步数（T=1000）
        schedule    : (n_low, n_mid, n_high) 各区间步数，论文值 (15,10,5)
        spacing     : linear 保持原始线性分段；quadratic/power 用非线性取点
        schedule_power : 非线性指数，>1 时每段更密集靠近较低噪声边界
    返回:
        tau : [S] 升序整数张量，S = sum(schedule)
    """
    n_low, n_mid, n_high = schedule
    T = total_steps - 1  # 999

    # 三个区间：[0, T/3], [T/3, 2T/3], [2T/3, T]
    boundary_low  = T // 3        # 333
    boundary_mid  = 2 * T // 3   # 666

    t_low  = _segment_timesteps(0, boundary_low, n_low, spacing, schedule_power, True, True)
    t_mid  = _segment_timesteps(boundary_low, boundary_mid, n_mid, spacing, schedule_power, False, False)
    t_high = _segment_timesteps(boundary_mid, T, n_high, spacing, schedule_power, False, True)

    tau = torch.cat([t_low, t_mid, t_high])
    tau = tau.unique().sort().values   # 去重 + 排序
    return tau

# ───────────────────────────────────────────────────────
# 辅助：DDPM 后验采样步（支持跳步）
# ───────────────────────────────────────────────────────

def ddpm_posterior_step(
    x_t:            torch.Tensor,
    x0_pred:        torch.Tensor,
    t_curr:         int,
    t_prev:         int,
    alphas_cumprod: torch.Tensor,
    eta:            float = 1.0,
    learned_log_var: torch.Tensor = None,
) -> torch.Tensor:
    """
    DDPM 后验采样：q(x_{t_prev} | x_t, x̂_0)
    支持任意跳步（t_curr → t_prev，不要求 t_prev = t_curr - 1）

    公式（DDPM 论文 Eq.7 推广到跳步）：
        μ̃ = c1 · x̂_0 + c2 · x_t
        β̃ = (1 − ᾱ_{t_prev}) / (1 − ᾱ_t) · (1 − ᾱ_t / ᾱ_{t_prev})
        x_{t_prev} = μ̃ + √β̃ · z，z ~ N(0,I)

    参数:
        x_t            : [B, C, H, W] 当前噪声图
        x0_pred        : [B, C, H, W] Tweedie 估计的干净图
        t_curr         : 当前时间步整数索引
        t_prev         : 目标时间步整数索引（-1 表示 t=0，直接返回 x̂_0）
        alphas_cumprod : [T] 预计算 ᾱ 序列
        eta            : ← 可调，DDIM 噪声系数，1.0=DDPM，0.0=DDIM 确定性
        learned_log_var: 可选，模型学出的 log 方差 [B,C,H,W]（LEARNED_RANGE）。
                         提供时噪声尺度用 exp(0.5·log_var)（对齐 guided-diffusion
                         p_sample），而非固定 β̃；None 时退回固定 β̃（向后兼容）。
    返回:
        x_{t_prev} : [B, C, H, W]
    """
    device = x_t.device

    if t_prev < 0:
        # 最后一步直接返回 Tweedie 估计（无需加噪）
        return x0_pred

    ab_t    = alphas_cumprod[t_curr].to(device)   # ᾱ_t
    ab_prev = alphas_cumprod[t_prev].to(device)   # ᾱ_{t_prev}

    # 后验均值系数
    # c1 对应 x̂_0 的系数，c2 对应 x_t 的系数
    c1 = ab_prev.sqrt() * (1.0 - ab_t / ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
    c2 = (ab_t / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
    mean = c1 * x0_pred + c2 * x_t

    if eta <= 0:
        return mean                                # 确定性（DDIM eta=0）

    if learned_log_var is not None:
        # 用模型学出的方差注入噪声（对齐原文 p_sample: exp(0.5·log_var)·z）
        noise = torch.randn_like(x_t)
        return mean + eta * torch.exp(0.5 * learned_log_var) * noise

    # 后验方差 β̃（固定方差，向后兼容路径）
    beta_tilde = (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * (1.0 - ab_t / ab_prev)
    beta_tilde = beta_tilde.clamp(min=0.0)

    noise = torch.randn_like(x_t)
    return mean + eta * beta_tilde.sqrt() * noise


# ───────────────────────────────────────────────────────
# ZAPS 主类
# ───────────────────────────────────────────────────────

class ZAPS(nn.Module):
    """
    ZAPS 零样本近似后验采样器

    参数:
        diffusion_model  : BaseDiffusionModel 实例（FFHQ 或 ImageNet）
        forward_operator : 退化算子实例，需实现 H(x) 和 transpose(y)
        num_steps        : ← 可调，采样步数，论文值 30
        schedule         : ← 可调，时间步分配，论文值 (15,10,5)
        num_epochs       : ← 可调，优化轮数，论文值 10
        lr               : ← 可调，Adam 学习率
        zeta_init        : ← 可调，ζ 初始值
        d_init           : ← 可调，D_t 对角初值（论文 0.2）
        wave / level     : ← 可调，正交小波类型与 DWT 级数（论文 db4）
        eta              : ← 可调，采样随机性（1.0=DDPM，0.0=DDIM）
        use_learned_var  : ← 可调，注噪用模型学出的方差(LEARNED_RANGE,对齐原文);False=固定β̃
    """

    def __init__(
        self,
        diffusion_model,
        forward_operator:  nn.Module,
        num_steps:   int   = NUM_SAMPLING_STEPS,
        schedule:    tuple = TIMESTEP_SCHEDULE,
        timestep_spacing: str = TIMESTEP_SPACING,
        schedule_power: float = SCHEDULE_POWER,
        num_epochs:  int   = NUM_EPOCHS,
        lr:          float = LEARNING_RATE,
        zeta_init:   float = ZETA_INIT,
        d_init:      float = D_INIT,
        eta:         float = 1.0,               # ← 可调
        channels:    int   = 3,                 # ← 可调：图像通道
        img_size:    int   = 256,               # ← 可调：图像边长
        wave:        str   = WAVELET,           # ← 可调：小波类型
        level:       int   = WAVELET_LEVEL,     # ← 可调：DWT 级数
        use_learned_var: bool = True,           # ← 可调：注噪用模型学出的方差(对齐原文)
    ):
        super().__init__()
        self.device = diffusion_model.device
        self.dm     = diffusion_model       # 扩散模型（权重冻结）
        self.A      = forward_operator.to(self.device)
        self.eta    = eta
        self.channels = channels
        self.img_size = img_size
        self.use_learned_var = use_learned_var

        # ── 固定正交 db4 DWT（W），用于 Hessian 对角化近似（论文 Eq.22）──
        self.dwt = OrthogonalDWT2D(wave=wave, level=level).to(self.device)

        # ── 时间步 ──
        self.tau = build_irregular_timesteps(
            total_steps=self.dm.num_steps,
            schedule=schedule,
            spacing=timestep_spacing,
            schedule_power=schedule_power,
        ).to(self.device)                   # [S] 升序
        S = len(self.tau)

        # ── 可学习参数 ──
        # ζ_t：对数似然权重 [S]，初始化为 zeta_init
        self.zeta = nn.Parameter(torch.full((S,), zeta_init, device=self.device))
        # D_t：小波域逐系数对角 [S,C,H,W]（论文 Eq.22 的 {D_t}），统一初始化为 d_init
        self.D = nn.Parameter(
            torch.full((S, channels, img_size, img_size), d_init, device=self.device)
        )

        self.num_epochs = num_epochs
        self.lr         = lr

    # ── 内部：把模型学出的 var_values 转成 log 方差（LEARNED_RANGE 插值）──
    def _learned_log_var(self, var_values: torch.Tensor,
                         t_curr: int, t_prev: int) -> torch.Tensor:
        """
        对齐 guided-diffusion p_mean_variance（LEARNED_RANGE）：
            frac = (v + 1) / 2
            log_var = frac · log(β_skip) + (1 − frac) · log(β̃_skip)
        跳步下 β_skip = 1 − ᾱ_t/ᾱ_prev，β̃_skip = (1−ᾱ_prev)/(1−ᾱ_t)·β_skip
        （与 ddpm_posterior_step 里的 beta_tilde 一致）。仅用于噪声尺度，detach。
        """
        ab = self.dm.alphas_cumprod
        ab_t = ab[t_curr]; ab_prev = ab[t_prev]
        beta_skip = (1.0 - ab_t / ab_prev).clamp(min=1e-8, max=0.999)
        beta_tilde = ((1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * beta_skip).clamp(min=1e-20)
        min_log = torch.log(beta_tilde)              # 下界 β̃
        max_log = torch.log(beta_skip)               # 上界 β
        frac = (var_values.detach() + 1.0) / 2.0     # v∈[-1,1] → [0,1]
        return frac * max_log + (1.0 - frac) * min_log

    # ── 内部：单次反向扩散（可微分，梯度流经 ζ 和 d）──────────

    def _reverse_diffusion(self, y: torch.Tensor,
                           nfe_counter: list = None,
                           eta_override: float = None,
                           init_noise: torch.Tensor = None,
                           record_indicators: bool = False) -> torch.Tensor:
        """
        执行一次完整反向扩散，返回 x_0 估计
        ε_θ 在 no_grad 下调用（冻结），x 保持计算图以供反传

        参数:
            y           : [B, C, H, W] 观测图像；超分任务中 H/W 可小于图像空间
            nfe_counter : 可选，传入 [0] 列表，函数会累加本次调用的 NFE 次数
            init_noise  : 可选固定初始噪声，用于优化阶段保持 unroll 轨迹可复现
            record_indicators : 是否记录双重状态感知指标（创新点①诊断，仅采样阶段用）
                          被动记录，不改变采样行为；结果写入 self._indicator_log
        返回:
            x : [B, C, H, W] 最终重建图像（仍在计算图中）
        """
        B = y.shape[0]
        sample_shape = (B, self.channels, self.img_size, self.img_size)
        if init_noise is None:
            x = torch.randn(sample_shape, device=self.device)
        else:
            x = init_noise.to(self.device)
            if tuple(x.shape) != sample_shape:
                raise ValueError(f"init_noise 形状应为 {sample_shape}，实际为 {tuple(x.shape)}")

        tau = self.tau
        S   = len(tau)
        ab  = self.dm.alphas_cumprod
        eta = self.eta if eta_override is None else eta_override

        if record_indicators:
            self._indicator_log = []
            prev_delta = None          # 含噪 x 的上一步增量(原始形式)
            prev_x0 = None             # 上一步 x̂_0(用于算 Δx̂_0)
            prev_delta_x0 = None       # 上一步 Δx̂_0(用于算相邻余弦)

        for i in range(S - 1, -1, -1):
            t_curr = tau[i].item()
            t_prev = tau[i - 1].item() if i > 0 else -1

            t_batch = torch.full((B,), t_curr, device=self.device, dtype=torch.long)

            # ── 步骤 1：score 模型推断（每次调用计 1 NFE）──
            with torch.no_grad():
                if self.use_learned_var:
                    eps, var_values = self.dm._predict_eps_var(x, t_batch)
                else:
                    eps, var_values = self.dm._predict_eps(x, t_batch), None
            if nfe_counter is not None:
                nfe_counter[0] += 1

            # ── 步骤 2：Tweedie 估计 x̂_0（论文 Eq.9）──
            ab_t      = ab[t_curr]
            sqrt_ab_t = ab_t.sqrt()
            sqrt_1mab = (1.0 - ab_t).sqrt()
            x0_pred   = (x - sqrt_1mab * eps.detach()) / sqrt_ab_t.clamp(min=1e-8)
            x0_pred   = x0_pred.clamp(-1.0, 1.0)

            # ── 步骤 3：无条件 DDPM 后验步（论文 Eq.10）──
            # 保留 x0_pred 梯度路径（c1 系数），使 ζ/d 的梯度能流经完整轨迹
            llv = self._learned_log_var(var_values, t_curr, t_prev) \
                if (var_values is not None and t_prev >= 0) else None
            x_uncond = ddpm_posterior_step(
                x, x0_pred, t_curr, t_prev, ab, eta=eta, learned_log_var=llv,
            )

            # ── 步骤 4：ZAPS 引导修正（论文 Algorithm 1 line 9, Eq.20/22）──
            # δ = ζ_i · (1/√ᾱ_i)·(I + (1−ᾱ_i)·W D_i W^⊤)·A^⊤(y − A x̂_0)
            residual = y - self.A.H(x0_pred)
            v        = self.A.transpose(residual)                 # A^⊤(残差)，图像空间 [B,C,H,W]
            Hv       = self.dwt.synthesis(self.D[i] * self.dwt.analysis(v))  # W D_i W^⊤ v
            guided   = (v + (1.0 - ab_t) * Hv) / sqrt_ab_t.clamp(min=1e-8)
            correction = self.zeta[i] * guided

            x_before = x
            x = x_uncond + correction

            # ── 创新点①：双重状态感知指标（被动记录，不影响上面的采样）──
            if record_indicators:
                # 外在物理一致性：残差范数 ‖y − H(x̂_0)‖
                resid_norm = residual.detach().flatten().norm().item()

                # 内在稳定性(原始形式)：含噪迭代 x 的相邻更新方向余弦
                delta = (x - x_before).detach()
                if prev_delta is None:
                    cos_sim = float("nan")
                else:
                    cos_sim = F.cosine_similarity(
                        delta.flatten(), prev_delta.flatten(), dim=0
                    ).item()
                prev_delta = delta

                # 内在稳定性(x̂_0 稳定版)：干净估计轨迹 Δx̂_0 的相邻余弦
                # 无注入噪声,反映重建"逐步成形"的平滑度(实验003 机制修正)
                x0_now = x0_pred.detach()
                cos_sim_x0 = float("nan")
                if prev_x0 is not None:
                    delta_x0 = x0_now - prev_x0
                    if prev_delta_x0 is not None:
                        cos_sim_x0 = F.cosine_similarity(
                            delta_x0.flatten(), prev_delta_x0.flatten(), dim=0
                        ).item()
                    prev_delta_x0 = delta_x0
                prev_x0 = x0_now

                self._indicator_log.append({
                    "step": int(S - 1 - i),          # 采样进度序号（0 起）
                    "t": int(t_curr),                # 对应扩散时间步
                    "residual_norm": resid_norm,
                    "cosine_sim": cos_sim,           # 原始:含噪 x 轨迹
                    "cosine_sim_x0": cos_sim_x0,     # 新:x̂_0 稳定版轨迹
                })

        return x

    # ── 最近邻:自适应时把访问的 t 映射到固定网格,取对应 ζ/D ──
    def _nearest_zd_index(self, t_val: int) -> int:
        return int((self.tau - t_val).abs().argmin().item())

    # ── 自适应调度采样(创新点②③;优化与采样共用,ζ/D 按步位置索引)──
    def _reverse_diffusion_adaptive(self, y: torch.Tensor, scheduler,
                                    nfe_counter: list = None,
                                    eta_override: float = None,
                                    init_noise: torch.Tensor = None,
                                    record_indicators: bool = False) -> torch.Tensor:
        """
        用 StateAwareScheduler 在线选时间步。优化与最终采样共用此路径,
        ζ/D 按【步位置 k】索引(非时间步最近邻)→ 优化学 ζ[k]、采样用 ζ[k],
        无论第 k 步落在哪个 t 都完全同步(修复 ζ 错配)。
        eps 在 no_grad 下取(score 冻结),ζ/D 保留计算图供优化反传。
        """
        B = y.shape[0]
        sample_shape = (B, self.channels, self.img_size, self.img_size)
        x = torch.randn(sample_shape, device=self.device) if init_noise is None \
            else init_noise.to(self.device)
        ab  = self.dm.alphas_cumprod
        eta = self.eta if eta_override is None else eta_override
        S = self.zeta.shape[0]

        scheduler.reset()
        if record_indicators:
            self._indicator_log = []
        prev_x0 = None
        prev_delta_x0 = None
        t = int(self.tau.max().item())          # 从最高噪声步起
        k = 0                                    # 步位置:ζ/D 索引

        while not scheduler.done() and t > 0:
            kd = min(k, S - 1)                   # 步位置索引 ζ/D(越界钳到末位)
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                if self.use_learned_var:
                    eps, var_values = self.dm._predict_eps_var(x, t_batch)
                else:
                    eps, var_values = self.dm._predict_eps(x, t_batch), None
            if nfe_counter is not None:
                nfe_counter[0] += 1

            ab_t = ab[t]; sqrt_ab_t = ab_t.sqrt(); sqrt_1mab = (1.0 - ab_t).sqrt()
            x0_pred = ((x - sqrt_1mab * eps.detach()) / sqrt_ab_t.clamp(min=1e-8)).clamp(-1.0, 1.0)
            residual = y - self.A.H(x0_pred)     # 保留计算图(供 ζ/D 反传)
            resid_norm = residual.detach().flatten().norm().item()

            # 辅助信号:x̂₀ 轨迹的相邻余弦(仅用于选步长,detach)
            cos_x0 = float("nan")
            x0d = x0_pred.detach()
            if prev_x0 is not None:
                delta_x0 = x0d - prev_x0
                if prev_delta_x0 is not None:
                    cos_x0 = F.cosine_similarity(
                        delta_x0.flatten(), prev_delta_x0.flatten(), dim=0).item()
                prev_delta_x0 = delta_x0
            prev_x0 = x0d

            # 喂指标 → 选步长
            scheduler.update_state(resid_norm=resid_norm, cos_x0=cos_x0)
            h = scheduler.select_step(t)
            t_prev = t - h
            if scheduler.done() or t_prev <= 0:
                t_prev = -1                      # 末步:落到 x̂₀

            # 采样步 + 状态自适应权重(创新点③);ζ/D 按步位置 kd,传张量保留梯度
            llv = self._learned_log_var(var_values, t, t_prev) \
                if (var_values is not None and t_prev >= 0) else None
            x_uncond = ddpm_posterior_step(x, x0_pred, t, t_prev, ab, eta=eta,
                                           learned_log_var=llv)
            zeta_k = scheduler.adapt_weight(self.zeta[kd])       # 张量,梯度可回传
            v      = self.A.transpose(residual)
            Hv     = self.dwt.synthesis(self.D[kd] * self.dwt.analysis(v))
            guided = (v + (1.0 - ab_t) * Hv) / sqrt_ab_t.clamp(min=1e-8)
            x = x_uncond + zeta_k * guided

            if record_indicators:
                self._indicator_log.append({
                    "step": k, "t": int(t), "h": int(h),
                    "residual_norm": resid_norm,
                    "cosine_sim": float("nan"),
                    "cosine_sim_x0": cos_x0,
                })

            if t_prev < 0:
                break
            t = t_prev
            k += 1
        return x

    # ── 零样本优化（论文 Section 3.1）──────────────────────

    def optimize(self, y: torch.Tensor, verbose: bool = True,
                 x0_gt: torch.Tensor = None, scheduler=None) -> list:
        """
        零样本训练：以单张观测 y 为监督，最小化物理引导损失
        L(y, x̂_0) = ‖y − H(x̂_0)‖²

        只更新 {ζ_t, d_t}，扩散模型权重全程冻结

        参数:
            y       : [B, C, H, W] 观测图像，值域 [-1, 1]
            verbose : 是否打印每轮 loss
            scheduler: 传入则优化也走自适应调度(ζ 按步位置学,与采样同步);
                       None 则用固定调度(原 ZAPS 行为)
        返回:
            loss_history : 每轮 loss 列表
        """
        y = y.to(self.device)
        optimizer = torch.optim.Adam(
            [self.zeta, self.D], lr=self.lr
        )

        nfe_counter  = [0]
        loss_history = []
        t_start      = time.time()
        fixed_noise  = torch.randn(
            y.shape[0], self.channels, self.img_size, self.img_size,
            device=self.device,
        )
        self._last_opt_x0 = None
        self._last_opt_noise = fixed_noise.detach()

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            is_last = (epoch == self.num_epochs - 1)

            # 方案1:优化用 eta=self.eta(=1,对齐 ZAPS Algorithm 1,优化即采样,消除eta割裂)
            # 最后一轮记录调度分布(indicators),其 x0 作为最终输出(last_opt)
            if scheduler is not None:
                x0_est = self._reverse_diffusion_adaptive(
                    y, scheduler, nfe_counter=nfe_counter,
                    eta_override=self.eta, init_noise=fixed_noise,
                    record_indicators=is_last,
                )
            else:
                x0_est = self._reverse_diffusion(
                    y, nfe_counter=nfe_counter, eta_override=self.eta, init_noise=fixed_noise
                )

            # 物理引导损失
            loss = F.mse_loss(self.A.H(x0_est), y)
            loss.backward()
            # 梯度裁剪，防止完整梯度路径下的爆炸
            torch.nn.utils.clip_grad_norm_([self.zeta, self.D], max_norm=1.0)
            optimizer.step()

            # 约束：ζ > 0（对数似然权重为正）；D 为 Hessian 对角近似，不限符号
            with torch.no_grad():
                self.zeta.data.clamp_(min=1e-6)

            self._last_opt_x0 = x0_est.detach()
            loss_val = loss.item()
            loss_history.append(loss_val)
            elapsed  = time.time() - t_start

            if verbose:
                grad_zeta = self.zeta.grad.norm().item() if self.zeta.grad is not None else 0.0
                grad_d    = self.D.grad.norm().item() if self.D.grad is not None else 0.0
                # 若传入 GT，计算当前 epoch 重建的 PSNR（仅供参考，轻量）
                psnr_str = ""
                if x0_gt is not None:
                    with torch.no_grad():
                        mse = ((x0_est.clamp(-1,1) - x0_gt.to(self.device)) ** 2).mean()
                        psnr = -10 * torch.log10(mse + 1e-8) + 20 * torch.log10(torch.tensor(2.0))
                    psnr_str = f"  PSNR={psnr.item():.2f}"
                print(f"  [ZAPS] Epoch {epoch+1:2d}/{self.num_epochs}"
                      f"  loss={loss_val:.6f}"
                      f"  NFE={nfe_counter[0]}"
                      f"  time={elapsed:.1f}s"
                      f"  zeta={self.zeta.mean().item():.4f}  gz={grad_zeta:.4f}"
                      f"  D={self.D.mean().item():.4f}  gd={grad_d:.4f}"
                      f"{psnr_str}")

        self._last_nfe     = nfe_counter[0]
        self._last_opt_sec = time.time() - t_start
        return loss_history

    # ── 推断采样（优化后调用）──────────────────────────────

    @torch.no_grad()
    def sample(self, y: torch.Tensor, eta_override: float = None,
               init_noise: torch.Tensor = None, scheduler=None) -> tuple:
        """
        用已优化的 ζ、d 执行一次无梯度最终采样，统计 NFE 和耗时

        参数:
            y : [B, C, H, W] 观测图像
            eta_override : 可选最终采样 eta；None 时使用 self.eta
            init_noise   : 可选初始噪声；用于复用优化轨迹噪声做对照
            scheduler    : 传入 StateAwareScheduler 则走自适应调度(创新点②③);
                           None 则用固定调度(基线)
        返回:
            (x0_est, nfe, elapsed_sec)
        """
        nfe_counter = [0]
        t0 = time.time()
        if scheduler is not None:
            x0 = self._reverse_diffusion_adaptive(
                y.to(self.device), scheduler, nfe_counter=nfe_counter,
                eta_override=eta_override, init_noise=init_noise,
                record_indicators=True,
            )
        else:
            x0 = self._reverse_diffusion(
                y.to(self.device), nfe_counter=nfe_counter,
                eta_override=eta_override, init_noise=init_noise,
                record_indicators=True,
            )
        return x0, nfe_counter[0], time.time() - t0

    # ── 主接口：optimize → sample ──────────────────────────

    def run(self, y: torch.Tensor, verbose: bool = True, **kwargs) -> dict:
        """
        完整 ZAPS 流程：零样本优化 → 最终采样

        参数:
            y       : [B, C, H, W] 观测图像
            verbose : 是否打印优化过程
        返回:
            dict 包含:
              x0_final    - 重建图像 [B, C, H, W]
              loss_history- 每轮 loss 列表
              nfe_opt     - 优化阶段总 NFE（= num_epochs × num_steps）
              nfe_sample  - 最终采样 NFE（= num_steps）
              nfe_total   - 全程 NFE
              time_opt_s  - 优化耗时（秒）
              time_sample_s - 最终采样耗时（秒）
              time_total_s  - 全程耗时（秒）
        """
        scheduler = kwargs.get("scheduler", None)   # 传入则优化+采样共用自适应调度
        loss_hist = self.optimize(y, verbose=verbose, x0_gt=kwargs.get("x0_gt"),
                                  scheduler=scheduler)
        final_mode = kwargs.get("final_mode", "sample")
        sample_eta = kwargs.get("sample_eta", None)
        sample_init = kwargs.get("sample_init", "random")

        if final_mode == "last_opt":
            if self._last_opt_x0 is None:
                raise RuntimeError("last_opt 输出不可用，请先执行 optimize")
            x0_final, nfe_s, t_s = self._last_opt_x0, 0, 0.0
        elif final_mode == "sample":
            init_noise = self._last_opt_noise if sample_init == "opt_noise" else None
            x0_final, nfe_s, t_s = self.sample(
                y, eta_override=sample_eta, init_noise=init_noise,
                scheduler=scheduler,
            )
        else:
            raise ValueError(f"未知 final_mode: {final_mode}，可选 'sample' 或 'last_opt'")

        return dict(
            x0_final      = x0_final,
            loss_history  = loss_hist,
            nfe_opt       = self._last_nfe,
            nfe_sample    = nfe_s,
            nfe_total     = self._last_nfe + nfe_s,
            time_opt_s    = self._last_opt_sec,
            time_sample_s = t_s,
            time_total_s  = self._last_opt_sec + t_s,
            indicators    = getattr(self, "_indicator_log", []),
        )
