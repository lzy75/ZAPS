"""
状态感知自适应采样步长控制器(创新点②③ 骨架 · v1)

设计见 毕业论文/02_代码实验/创新点机制设计.md

职责:给定每步的双重状态指标(标量),在固定 NFE 预算下贪心选择下一步长,
      并按状态协同调节似然引导权重。

刻意做成 torch-free(只吃标量),便于无 GPU 单元测试;
张量→指标的计算留在采样侧(zaps_algorithm.py),复用已有的 x̂₀ 与 residual。

旧版 ``StateAwareScheduler`` 保留给历史实验；新的
``BudgetedStateAwareScheduler`` 专门用于严格配对的状态感知实验：它以一条
已经验证过的固定时间步网格为名义轨迹，状态信号关闭时必须逐点退化回该网格。
"""
from dataclasses import dataclass
import math


@dataclass
class SchedulerConfig:
    h_min: float = 1.0            # 最小步长(时间步单位)
    h_max: float = 120.0         # 最大步长(仅作安全上限)
    schedule_mode: str = "v2"    # "v2"=幂律基础+有界调制(现版,回退用);"v3"=统一代价函数
    # ── base 名义调度:EDM(Karras)ρ 幂律,低噪密采,按 N 自动生成(可跨 NFE)──
    p_schedule: float = 2.0      # EDM ρ:越大越偏低噪密采;逆问题用温和值(2≈ZAPS 5/10/15),=1 近均匀
    # ── 自适应调制(围绕基础调度做有界扰动,信号无效时退化为纯基础调度)──
    beta: float = 0.5            # 残差"下降速率"对步长的调制强度(相对量,非绝对残差)
    w_cos: float = 0.3           # 余弦(稳定性)辅助信号权重∈[0,1];0=纯基础+残差
    mod_min: float = 0.5         # 调制因子下限(防某步被压过小)
    mod_max: float = 1.5         # 调制因子上限(防某步冲过大)
    s_min: float = 0.05          # 余弦稳定因子下限
    # ── v3:统一代价评价函数(开题"统一误差空间")──
    # E = omega·E_c + (1-omega)·E_r; h = base · theta/(theta+E); base=15/10/5 名义步长
    omega: float = 0.5           # 曲率误差 vs 停滞误差 的权衡∈[0,1];0=纯残差,1=纯曲率
    theta: float = 0.5           # 容忍度:越大越激进(E 对步长压缩越弱)
    # ── 创新点③ 权重协同 ──
    gamma_r: float = 0.5         # 权重-残差耦合(残差大→升 ζ)
    gamma_s: float = 0.3         # 权重-稳定性耦合(轨迹弯→降 ζ)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class StateAwareScheduler:
    """
    固定预算 N 步的在线贪心步长控制器。

    用法(采样侧,伪码):
        sch = StateAwareScheduler(total_budget=N, t_start=T-1, cfg=...)
        t = t_start
        while not sch.done():
            x0_hat, resid_norm = 一步计算(t)         # 张量→标量
            sch.update_state(x0_hat_flat, resid_norm) # 传入展平后的 x̂₀ 或其变化
            h = sch.select_step(t)                     # 贪心选步长
            zeta_k = sch.adapt_weight(zeta_base_k)     # 创新点③
            t = t - h
    """

    def __init__(self, total_budget: int, t_start: int, cfg: SchedulerConfig = None):
        self.N = int(total_budget)
        self.t_start = int(t_start)
        self.cfg = cfg or SchedulerConfig()
        self.reset()

    def reset(self):
        self.used = 0
        self.r0 = None            # 首步残差,用于归一化
        self.s = float("nan")     # 当前稳定性 cos(x̂₀ 轨迹,采样侧算好传入)
        self.r_tilde = 1.0        # 相对残差 r_k / r0
        self.dr_rel = 0.0         # 残差相对下降速率 (r_{k-1}-r_k)/r_{k-1}
        self._prev_r = None
        self._dr_ema = None       # dr_rel 运行均值(自归一化基准,让调制对"偏离典型降速"响应)
        # v3 去趋势:曲率/残差降速的运行均值,评价函数只对"偏离常态的异常"响应
        self._ec_ema = None       # 曲率误差 E_c 的运行均值(该噪声水平的常态)
        self._er_ema = None       # 停滞误差 E_r 的运行均值
        # v3:预计算固定名义调度(低噪声密采先验,不随 t 漂),按步位置索引
        self._nominal = self._precompute_nominal()

    def _precompute_nominal(self):
        """
        预计算 N 步名义步长序列(降序,和为 t_start),用 EDM(Karras 2022)ρ 调度形状。
        EDM σ 域:σ_i = (σmax^(1/ρ) + i/(N-1)·(σmin^(1/ρ) − σmax^(1/ρ)))^ρ,i=0..N-1
        再线性映射到 t 域 [0, t_start](σ 大=高噪=t 大)。
        依据:EDM ρ 幂律是扩散采样主流调度,ρ 控低噪密采程度,且【按 N 自动生成】→ 可跨 NFE 迁移
             (相比 ZAPS 15/10/5 只对固定 NFE 手调)。ρ=p_schedule,逆问题用温和值(默认 2)。
        """
        c = self.cfg
        rho = max(1.0, c.p_schedule)   # 复用 p_schedule 承载 ρ
        smin, smax = 0.002, 80.0
        a, bmax = smin ** (1.0 / rho), smax ** (1.0 / rho)
        sig = [(bmax + i / (self.N - 1) * (a - bmax)) ** rho for i in range(self.N)] if self.N > 1 else [smax]
        lo_s, hi_s = min(sig), max(sig)
        rng = (hi_s - lo_s) or 1.0
        pts = [self.t_start * (s - lo_s) / rng for s in sig]   # 降序:σmax→t_start, σmin→0
        pts.append(0.0)
        return [max(0.0, pts[k] - pts[k + 1]) for k in range(self.N)]

    # ── 指标更新:采样侧算好标量后传入(torch 侧算余弦,避免展平成 list)──
    def update_state(self, resid_norm: float, cos_x0: float = float("nan")):
        """
        resid_norm : ‖y − H(x̂₀)‖ 标量。
        cos_x0     : 相邻 Δx̂₀ 的余弦(辅助信号),首步/无前向量时传 nan。
        """
        if self.r0 is None:
            self.r0 = resid_norm if resid_norm > 1e-12 else 1.0
        self.r_tilde = resid_norm / self.r0
        # 残差"相对下降速率"(v2 关键):归一化,消除噪声水平带来的绝对量差异
        if self._prev_r is not None and self._prev_r > 1e-12:
            self.dr_rel = (self._prev_r - resid_norm) / self._prev_r
        else:
            self.dr_rel = 0.0
        self._prev_r = resid_norm
        self.s = cos_x0

    # ── 贪心选步长(v2:幂律基础调度 + 有界自适应调制)──
    def select_step(self, t: int) -> int:
        """
        在当前 t、剩余预算下选步长 h。设计:
          1) 基础步长 = 把剩余时间 t 按幂律分给剩余 b 步 → 低噪声区(小 t)步更小,密采;
          2) 调制因子 = 残差降速 + 余弦稳定性,围绕基础步长做 [mod_min,mod_max] 有界扰动;
          3) 硬预算可行性:确保剩余步能恰好走到 0。
        信号无效(nan/为0)时调制→1,退化为纯幂律基础调度(不会崩)。
        """
        c = self.cfg
        b = max(1, self.N - self.used)     # 含当前步的剩余步数
        if b <= 1:
            h = t                          # 最后一步:直达 0
            self.used += 1
            return int(max(c.h_min, h))

        # ══ v3:统一代价评价函数(开题"统一误差空间")══
        if c.schedule_mode == "v3":
            h = self._select_step_v3(t, b)
            self.used += 1
            return h

        # ── 1) 幂律基础步长:把 [0,t] 分成 b 段,取第一段长度 ──
        # 线性分段 t/b;幂律让"靠近 0 的段更短" → 用当前占比的幂律权重
        base = t / b                        # 线性基准
        # 幂律修正:高噪声(t 大)放大步、低噪声(t 小)缩小步
        frac = t / max(1.0, self.t_start)   # 当前 t 在全程的占比∈(0,1]
        base = base * (c.p_schedule * frac + (1.0 - frac) * 1.0 / c.p_schedule)

        # ── 2) 有界自适应调制 ──
        # 残差降得比"近期典型速度"快 → 轨迹顺,迈大步;慢/反弹 → 迈小步。
        # 用相对降速与其运行均值的偏差,再乘增益放大,使 beta 有实际杠杆(自检项3)。
        if self._dr_ema is None:
            self._dr_ema = self.dr_rel
        dev = self.dr_rel - self._dr_ema           # 偏离近期典型降速
        self._dr_ema = 0.7 * self._dr_ema + 0.3 * self.dr_rel   # 更新运行均值
        mod_r = 1.0 + c.beta * _clip(dev / 0.05, -3.0, 3.0)     # /0.05 归一化到 O(1),×beta 放大
        # 余弦:s 高(平滑)→大步;经 w_cos 融入
        s_use = self.s if self.s == self.s else None
        if s_use is None:
            mod_s = 1.0
        else:
            stab = _clip((s_use + 1.0) / 2.0, c.s_min, 1.0)   # ∈[s_min,1]
            mod_s = (1.0 - c.w_cos) * 1.0 + c.w_cos * (2.0 * stab)  # 中性1,stab=1→放大
        mod = _clip(mod_r * mod_s, c.mod_min, c.mod_max)
        h_raw = base * mod

        # ── 3) 硬预算可行性:剩余 b 步每步∈[h_min,h_max],须能恰好到 0 ──
        lo = t - (b - 1) * c.h_max         # 后续全用最大步也要够 → 本步至少这么大
        hi = t - (b - 1) * c.h_min         # 后续全用最小步 → 本步至多这么大
        h = _clip(h_raw, max(c.h_min, lo), min(hi, t))
        h = int(round(_clip(h, c.h_min, max(c.h_min, t))))
        self.used += 1
        return h

    # ── v3 核心:统一代价评价函数选步长(去趋势版)──
    def _select_step_v3(self, t: int, b: int) -> int:
        """
        统一误差空间(开题设计)+ 去趋势:
          原始 e_c=(1−cos)/2 曲率误差, e_r=1−dr_rel 停滞误差 ∈[0,1]
          去趋势:各减去自身运行均值(该噪声水平的常态)→ 只对"偏离常态的异常"响应
          E = 0.5 + omega·dev_c + (1−omega)·dev_r    (dev 为去趋势偏差,E 中心 0.5)
          h = base · [1 + theta·(0.5 − E)·2]         E>0.5(异常高)→小步密采;E<0.5→大步
        base = 15/10/5 名义步长(低噪密采先验);去趋势使高噪声区正常大波动不再误判密采。
        """
        c = self.cfg
        base = self._nominal_base(t, b)
        # 原始误差
        e_c = (1.0 - self.s) / 2.0 if self.s == self.s else None    # 曲率误差(nan→跳过)
        e_r = 1.0 - _clip(self.dr_rel, 0.0, 1.0)                    # 停滞误差
        # 去趋势:减运行均值,得"偏离常态"的异常量 dev∈[-0.5,0.5]
        a = 0.3   # EMA 系数
        if e_c is not None:
            if self._ec_ema is None: self._ec_ema = e_c
            dev_c = _clip(e_c - self._ec_ema, -0.5, 0.5)
            self._ec_ema = (1 - a) * self._ec_ema + a * e_c
        else:
            dev_c = 0.0
        if self._er_ema is None: self._er_ema = e_r
        dev_r = _clip(e_r - self._er_ema, -0.5, 0.5)
        self._er_ema = (1 - a) * self._er_ema + a * e_r
        # 统一代价:以 0.5 为中心,异常偏高→E>0.5→密采
        E = 0.5 + c.omega * dev_c + (1.0 - c.omega) * dev_r
        E = _clip(E, 0.0, 1.0)
        mod = 1.0 + c.theta * (0.5 - E) * 2.0        # mod∈[1−θ,1+θ],均值≈1 保住 base 分布
        h_raw = base * mod
        # 硬预算可行性
        lo = t - (b - 1) * c.h_max
        hi = t - (b - 1) * c.h_min
        h = _clip(h_raw, max(c.h_min, lo), min(hi, t))
        return int(round(_clip(h, c.h_min, max(c.h_min, t))))

    def _nominal_base(self, t: int, b: int) -> float:
        """
        名义步长(自校正):把剩余时间 t 按"剩余名义步长的形状比例"分配给本步。
        base = t · nominal[used] / sum(nominal[used:])
        → 无论前面自适应如何偏离,剩余 t 总按名义形状(低噪密采)重新分摊,末段不爆步。
        """
        if self._nominal is not None and self.used < len(self._nominal):
            rem = sum(self._nominal[self.used:])
            if rem > 1e-9:
                return max(self.cfg.h_min, t * self._nominal[self.used] / rem)
        return t / b

    # ── 创新点③:权重协同 ──
    def adapt_weight(self, zeta_base):
        """
        按状态调制似然权重。zeta_base 可为 float 或 torch 张量:
        传张量时保留计算图(factor_r/factor_s 是常量标量,不断梯度),
        使优化阶段 ζ 的梯度能正常回传。
        """
        c = self.cfg
        s_use = self.s if self.s == self.s else 1.0
        factor_r = 1.0 + c.gamma_r * self.r_tilde
        factor_s = max(0.0, 1.0 - c.gamma_s * (1.0 - s_use) / 2.0)
        return zeta_base * factor_r * factor_s

    def done(self) -> bool:
        return self.used >= self.N


@dataclass
class BudgetedSchedulerConfig:
    """PPT 状态指标对应的固定预算调度参数。"""

    residual_weight: float = 0.8
    cosine_weight: float = 0.2
    response_strength: float = 0.5
    residual_target_drop: float = 0.05
    residual_ema_decay: float = 0.0
    residual_mode: str = "target"
    soft_baseline_decay: float = 0.7
    soft_scale_floor: float = 0.01
    soft_error_amplitude: float = 0.25
    profile_warmup_epochs: int = 1
    profile_center_decay: float = 0.8
    profile_residual_scale: float = 0.15
    profile_cosine_scale: float = 0.2
    profile_cosine_gate: float = 0.25
    profile_gate_mode: str = "symmetric"
    weight_mode: str = "identity"
    weight_residual_gain: float = 0.2
    weight_cosine_gain: float = 0.05
    weight_min: float = 0.9
    weight_max: float = 1.1
    mod_min: float = 0.75
    mod_max: float = 1.25


class BudgetedStateAwareScheduler:
    """在固定 NFE 下围绕给定基线网格做保守、可退化的在线调整。

    状态代价遵循汇报中的思路：残差为主、轨迹余弦为辅。为了避免此前
    “绝对残差大就一味缩步”造成高噪声区预算耗尽，残差项改成相邻步骤的
    相对下降是否停滞：

        d_k = (r_{k-1}-r_k)/r_{k-1}              # 可选 EMA 平滑
        E_r = clip(1 - d_k / q, 0, 1)             # target 模式
        E_r = 0.5-a*tanh((d_k-m_k)/s_k)           # adaptive_soft 模式
        E_c = clip((1-cos(Delta x0_k, Delta x0_{k-1}))/2, 0, 1)
        E   = w_r E_r + w_c E_c

    ``E>0.5`` 表示当前状态较难，缩小下一跨度；``E<0.5`` 则放大跨度。
    首步或余弦不可用时对应项取中性值 0.5。所有调整都以给定的名义网格
    为基准，并硬性保证恰好 N 次模型调用、最后一次调用位于 t=0。
    """

    include_zero = True

    def __init__(self, nominal_timesteps, cfg: BudgetedSchedulerConfig = None):
        grid = [int(value) for value in nominal_timesteps]
        if len(grid) < 2:
            raise ValueError("nominal_timesteps 至少需要两个点")
        if grid[-1] != 0:
            raise ValueError("nominal_timesteps 必须以 t=0 结束")
        if any(left <= right for left, right in zip(grid, grid[1:])):
            raise ValueError("nominal_timesteps 必须严格递减且不重复")

        self.nominal_timesteps = grid
        self.N = len(grid)
        self.t_start = grid[0]
        self.cfg = cfg or BudgetedSchedulerConfig()
        self._validate_config()
        self.reset()

    def _validate_config(self):
        c = self.cfg
        if c.residual_weight < 0 or c.cosine_weight < 0:
            raise ValueError("状态指标权重不能为负")
        if abs(c.residual_weight + c.cosine_weight - 1.0) > 1e-8:
            raise ValueError("residual_weight + cosine_weight 必须等于 1")
        if c.response_strength < 0:
            raise ValueError("response_strength 不能为负")
        if c.residual_target_drop <= 0:
            raise ValueError("residual_target_drop 必须为正")
        if not 0.0 <= c.residual_ema_decay < 1.0:
            raise ValueError("residual_ema_decay 必须位于 [0,1)")
        if c.residual_mode not in ("target", "adaptive_soft", "reference_profile"):
            raise ValueError(
                "residual_mode 必须是 target、adaptive_soft 或 reference_profile"
            )
        if not 0.0 <= c.soft_baseline_decay < 1.0:
            raise ValueError("soft_baseline_decay 必须位于 [0,1)")
        if c.soft_scale_floor <= 0:
            raise ValueError("soft_scale_floor 必须为正")
        if not 0 < c.soft_error_amplitude <= 0.5:
            raise ValueError("soft_error_amplitude 必须位于 (0,0.5]")
        if c.profile_warmup_epochs < 1:
            raise ValueError("profile_warmup_epochs 必须至少为 1")
        if not 0.0 <= c.profile_center_decay < 1.0:
            raise ValueError("profile_center_decay 必须位于 [0,1)")
        if c.profile_residual_scale <= 0 or c.profile_cosine_scale <= 0:
            raise ValueError("reference-profile 的归一化尺度必须为正")
        if not 0.0 <= c.profile_cosine_gate <= 1.0:
            raise ValueError("profile_cosine_gate 必须位于 [0,1]")
        if c.profile_gate_mode not in ("symmetric", "veto_only"):
            raise ValueError(
                "profile_gate_mode 必须是 symmetric 或 veto_only"
            )
        if c.weight_mode not in ("identity", "state_balanced"):
            raise ValueError("weight_mode 必须是 identity 或 state_balanced")
        if c.weight_residual_gain < 0 or c.weight_cosine_gain < 0:
            raise ValueError("状态权重增益不能为负")
        if not 0 < c.weight_min <= 1.0 <= c.weight_max:
            raise ValueError("权重边界必须满足 0 < min <= 1 <= max")
        if not (0 < c.mod_min <= 1.0 <= c.mod_max):
            raise ValueError("调制边界必须满足 0 < mod_min <= 1 <= mod_max")

    def reset(self):
        # ``reset`` 在每次 unroll 开始时调用。参考曲线跨 epoch 保留，
        # 但当前 epoch 的在线中心重新建立；这样首轮固定基线本身就是 pilot，
        # 不增加任何 NFE。
        if not hasattr(self, "_completed_unrolls"):
            self._completed_unrolls = 0
            self._profile_reference_residual = [float("nan")] * self.N
            self._profile_reference_cosine = [float("nan")] * self.N
        elif getattr(self, "used", 0) >= self.N:
            self._completed_unrolls += 1
        self._profile_log_ratio_ema = None
        self.used = 0
        self._prev_r = None
        self._residual_drop_ema = None
        self._soft_drop_baseline = None
        self._soft_abs_deviation = None
        self.residual_norm = float("nan")
        self.relative_residual_drop = float("nan")
        self.smoothed_relative_residual_drop = float("nan")
        self.residual_error = 0.5
        self.residual_baseline = float("nan")
        self.residual_scale = float("nan")
        self.residual_zscore = float("nan")
        self.cosine_x0 = float("nan")
        self.cosine_error = 0.5
        self.state_score = 0.5
        self.base_step = float("nan")
        self.step_modifier = 1.0
        self.selected_step = None
        self.guidance_modifier = 1.0
        self.profile_warmup = False
        self.profile_log_ratio = float("nan")
        self.profile_residual_signal = 0.0
        self.profile_cosine_signal = 0.0
        self.profile_confidence = 1.0

    def _update_reference_profile(
        self, resid_norm: float, cos_x0: float = float("nan")
    ) -> None:
        """Compare the current state with a schedule-specific pilot profile.

        The physical residual determines the sign of the schedule response.
        Cosine similarity cannot reverse that decision; it only increases or
        decreases confidence when the two signals agree or disagree.
        """
        c = self.cfg
        k = min(self.used, self.N - 1)
        if self._prev_r is None or self._prev_r <= 1e-12:
            relative_drop = float("nan")
        else:
            relative_drop = (self._prev_r - resid_norm) / self._prev_r
        self._prev_r = resid_norm

        warmup = self._completed_unrolls < c.profile_warmup_epochs
        if warmup:
            self._profile_reference_residual[k] = resid_norm
            if cos_x0 == cos_x0:
                self._profile_reference_cosine[k] = float(cos_x0)
            log_ratio = 0.0
            residual_signal = 0.0
            cosine_signal = 0.0
            confidence = 1.0
            combined_signal = 0.0
            reference_residual = resid_norm
        else:
            reference_residual = self._profile_reference_residual[k]
            if reference_residual != reference_residual or reference_residual <= 1e-12:
                reference_residual = resid_norm
            log_ratio = math.log(max(resid_norm, 1e-12) / reference_residual)

            if self._profile_log_ratio_ema is None:
                centered_log_ratio = 0.0
                self._profile_log_ratio_ema = log_ratio
            else:
                centered_log_ratio = log_ratio - self._profile_log_ratio_ema
                decay = c.profile_center_decay
                self._profile_log_ratio_ema = (
                    decay * self._profile_log_ratio_ema
                    + (1.0 - decay) * log_ratio
                )
            residual_signal = math.tanh(
                centered_log_ratio / c.profile_residual_scale
            )

            reference_cosine = self._profile_reference_cosine[k]
            if cos_x0 == cos_x0 and reference_cosine == reference_cosine:
                # 正值表示当前轨迹比 pilot 更不稳定。
                cosine_signal = math.tanh(
                    (reference_cosine - float(cos_x0)) / c.profile_cosine_scale
                )
            else:
                cosine_signal = 0.0

            if abs(residual_signal) <= 1e-12:
                confidence = 1.0
            else:
                agreement = (
                    (1.0 if residual_signal > 0 else -1.0) * cosine_signal
                )
                if c.profile_gate_mode == "veto_only":
                    # 余弦只在与物理指标冲突时削弱响应；两者一致时不再
                    # 额外放大，避免轨迹稳定性指标变成第二个加速器。
                    confidence = _clip(
                        1.0
                        - c.profile_cosine_gate * max(0.0, -agreement),
                        1.0 - c.profile_cosine_gate,
                        1.0,
                    )
                else:
                    confidence = _clip(
                        1.0 + c.profile_cosine_gate * agreement,
                        1.0 - c.profile_cosine_gate,
                        1.0 + c.profile_cosine_gate,
                    )
            combined_signal = _clip(
                residual_signal * confidence, -1.0, 1.0
            )

        self.residual_norm = resid_norm
        self.relative_residual_drop = relative_drop
        self.smoothed_relative_residual_drop = relative_drop
        self.residual_error = 0.5 + 0.5 * residual_signal
        self.residual_baseline = reference_residual
        self.residual_scale = c.profile_residual_scale
        self.residual_zscore = residual_signal
        self.cosine_x0 = float(cos_x0)
        self.cosine_error = 0.5 + 0.5 * cosine_signal
        self.state_score = 0.5 + 0.5 * combined_signal
        self.profile_warmup = warmup
        self.profile_log_ratio = log_ratio
        self.profile_residual_signal = residual_signal
        self.profile_cosine_signal = cosine_signal
        self.profile_confidence = confidence

    def update_state(self, resid_norm: float, cos_x0: float = float("nan")):
        resid_norm = float(resid_norm)
        if self.cfg.residual_mode == "reference_profile":
            self._update_reference_profile(resid_norm, cos_x0)
            return
        if self._prev_r is None or self._prev_r <= 1e-12:
            dr_rel = float("nan")
            smoothed_dr_rel = float("nan")
            residual_error = 0.5
            residual_baseline = float("nan")
            residual_scale = float("nan")
            residual_zscore = float("nan")
        else:
            dr_rel = (self._prev_r - resid_norm) / self._prev_r
            if self.cfg.residual_mode == "adaptive_soft":
                smoothed_dr_rel = dr_rel
                if self._soft_drop_baseline is None:
                    # 第一项只建立局部基线，不立即改变步长。
                    residual_baseline = dr_rel
                    residual_scale = self.cfg.soft_scale_floor
                    residual_zscore = 0.0
                    residual_error = 0.5
                    self._soft_drop_baseline = dr_rel
                    self._soft_abs_deviation = self.cfg.soft_scale_floor
                else:
                    baseline = self._soft_drop_baseline
                    scale = max(
                        self._soft_abs_deviation,
                        self.cfg.soft_scale_floor,
                    )
                    deviation = dr_rel - baseline
                    residual_zscore = deviation / scale
                    residual_error = (
                        0.5
                        - self.cfg.soft_error_amplitude
                        * math.tanh(residual_zscore)
                    )
                    decay = self.cfg.soft_baseline_decay
                    self._soft_drop_baseline = (
                        decay * baseline + (1.0 - decay) * dr_rel
                    )
                    self._soft_abs_deviation = (
                        decay * self._soft_abs_deviation
                        + (1.0 - decay) * abs(deviation)
                    )
                    residual_baseline = baseline
                    residual_scale = scale
            else:
                decay = self.cfg.residual_ema_decay
                if decay <= 0.0 or self._residual_drop_ema is None:
                    smoothed_dr_rel = dr_rel
                else:
                    smoothed_dr_rel = (
                        decay * self._residual_drop_ema + (1.0 - decay) * dr_rel
                    )
                self._residual_drop_ema = smoothed_dr_rel
                residual_error = (
                    1.0 - smoothed_dr_rel / self.cfg.residual_target_drop
                )
                residual_error = _clip(residual_error, 0.0, 1.0)
                residual_baseline = self.cfg.residual_target_drop
                residual_scale = self.cfg.residual_target_drop
                residual_zscore = (
                    (smoothed_dr_rel - self.cfg.residual_target_drop)
                    / self.cfg.residual_target_drop
                )
        self._prev_r = resid_norm

        if cos_x0 == cos_x0:
            cosine_error = _clip((1.0 - float(cos_x0)) / 2.0, 0.0, 1.0)
        else:
            cosine_error = 0.5

        c = self.cfg
        self.residual_norm = resid_norm
        self.relative_residual_drop = dr_rel
        self.smoothed_relative_residual_drop = smoothed_dr_rel
        self.residual_error = residual_error
        self.residual_baseline = residual_baseline
        self.residual_scale = residual_scale
        self.residual_zscore = residual_zscore
        self.cosine_x0 = float(cos_x0)
        self.cosine_error = cosine_error
        self.state_score = (
            c.residual_weight * residual_error
            + c.cosine_weight * cosine_error
        )

    def select_step(self, t: int) -> int:
        """选择跨度并占用一次 NFE 预算。"""
        if self.done():
            raise RuntimeError("调度预算已经用完")
        t = int(t)
        k = self.used

        # 最后一次模型调用必须发生在 t=0；返回 1 仅作为落到 x0 的哨兵跨度。
        if k == self.N - 1:
            if t != 0:
                raise RuntimeError(f"最后一次调用应位于 t=0，实际 t={t}")
            self.base_step = 1.0
            self.step_modifier = 1.0
            self.selected_step = 1
            self.used += 1
            return 1

        nominal_t = self.nominal_timesteps[k]
        nominal_next = self.nominal_timesteps[k + 1]
        nominal_h = nominal_t - nominal_next
        # 前面若已被状态调制，按剩余 t 等比例缩放名义跨度，避免末段补偿性大跳。
        base = t * nominal_h / max(1, nominal_t)
        c = self.cfg
        modifier = 1.0 + c.response_strength * (0.5 - self.state_score) * 2.0
        modifier = _clip(modifier, c.mod_min, c.mod_max)
        h_raw = base * modifier

        future_calls = self.N - k - 1
        if future_calls == 1:
            # 倒数第二次调用后必须精确到 0，确保最后一次评估与固定基线一致。
            h = t
        else:
            # 为后续每次至少下降 1 留足互异的整数时间步。
            max_h = t - (future_calls - 1)
            if max_h < 1:
                raise RuntimeError(
                    f"剩余时间步不足以满足固定预算: t={t}, future_calls={future_calls}"
                )
            h = int(round(_clip(h_raw, 1.0, float(max_h))))

        self.base_step = float(base)
        self.step_modifier = float(modifier)
        self.selected_step = int(h)
        self.used += 1
        return int(h)

    def adapt_weight(self, zeta_base):
        """可选地用同一状态小幅调制当前 ζ，不改变 ζ/D 的可学习性。

        残差停滞 ``E_r>0.5`` 时增强引导；轨迹不稳定 ``E_c>0.5`` 时
        抑制引导。两项分开进入，避免把余弦误当成时间步主信号。
        """
        c = self.cfg
        if c.weight_mode == "state_balanced":
            modifier = (
                1.0
                + c.weight_residual_gain * (self.residual_error - 0.5)
                - c.weight_cosine_gain * (self.cosine_error - 0.5)
            )
            modifier = _clip(modifier, c.weight_min, c.weight_max)
        else:
            modifier = 1.0
        self.guidance_modifier = float(modifier)
        return zeta_base * modifier

    def snapshot(self) -> dict:
        return {
            "relative_residual_drop": self.relative_residual_drop,
            "smoothed_relative_residual_drop": (
                self.smoothed_relative_residual_drop
            ),
            "residual_error": self.residual_error,
            "residual_baseline": self.residual_baseline,
            "residual_scale": self.residual_scale,
            "residual_zscore": self.residual_zscore,
            "cosine_error": self.cosine_error,
            "state_score": self.state_score,
            "base_step": self.base_step,
            "step_modifier": self.step_modifier,
            "guidance_modifier": self.guidance_modifier,
            "profile_epoch": self._completed_unrolls,
            "profile_warmup": self.profile_warmup,
            "profile_log_ratio": self.profile_log_ratio,
            "profile_residual_signal": self.profile_residual_signal,
            "profile_cosine_signal": self.profile_cosine_signal,
            "profile_confidence": self.profile_confidence,
        }

    def done(self) -> bool:
        return self.used >= self.N
