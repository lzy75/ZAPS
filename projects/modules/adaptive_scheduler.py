"""
状态感知自适应采样步长控制器(创新点②③ 骨架 · v1)

设计见 毕业论文/02_代码实验/创新点机制设计.md

职责:给定每步的双重状态指标(标量),在固定 NFE 预算下贪心选择下一步长,
      并按状态协同调节似然引导权重。

刻意做成 torch-free(只吃标量),便于无 GPU 单元测试;
张量→指标的计算留在采样侧(zaps_algorithm.py),复用已有的 x̂₀ 与 residual。

⚠️ 本文件为骨架,尚未接入 _reverse_diffusion 采样循环。
"""
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    h_min: float = 1.0            # 最小步长(时间步单位)
    h_max: float = 120.0         # 最大步长
    s_min: float = 0.05          # 稳定因子下限(避免完全停步)
    f_min: float = 0.1           # 保真因子下限
    beta: float = 1.0            # 残差对步长的抑制强度(主信号)
    w_cos: float = 0.3           # 余弦(稳定性)信号权重∈[0,1];0=纯残差,1=全余弦。
                                 # 实验005:余弦仅辅助(cosx0_last r≈−0.5),故默认小
    delta_max: float = 60.0      # 相邻步步长最大变化率(故障恢复,HSO 教训)
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
        self.h_prev = None
        self.r0 = None            # 首步残差,用于归一化
        self.s = float("nan")     # 当前稳定性 cos(x̂₀ 轨迹,采样侧算好传入)
        self.r_tilde = 1.0        # 相对残差 r_k / r0
        self.dr = 0.0             # 残差降速
        self._prev_r = None

    # ── 指标更新:采样侧算好标量后传入(torch 侧算余弦,避免展平成 list)──
    def update_state(self, resid_norm: float, cos_x0: float = float("nan")):
        """
        resid_norm : ‖y − H(x̂₀)‖ 标量(主信号)。
        cos_x0     : 相邻 Δx̂₀ 的余弦(辅助信号),首步/无前向量时传 nan。
        """
        # 外在物理一致性(主)
        if self.r0 is None:
            self.r0 = resid_norm if resid_norm > 1e-12 else 1.0
        self.r_tilde = resid_norm / self.r0
        self.dr = 0.0 if self._prev_r is None else (self._prev_r - resid_norm)
        self._prev_r = resid_norm
        # 内在轨迹稳定性(辅,采样侧已算好)
        self.s = cos_x0

    # ── 贪心选步长 ──
    def select_step(self, t: int) -> int:
        """在当前时间步 t、剩余预算下,贪心选下一步长 h(整数时间步)。"""
        c = self.cfg
        remaining = self.N - self.used          # 含当前步在内的剩余步数
        # 保真因子(主信号):残差大→小步
        fid = _clip(1.0 - c.beta * self.r_tilde, c.f_min, 1.0)
        # 稳定因子(辅信号):s 高→大步;经 w_cos 加权融入(w_cos=0 则完全不影响)
        s_use = self.s if self.s == self.s else 1.0   # nan→中性
        stab_pure = _clip((s_use + 1.0) / 2.0, c.s_min, 1.0)
        stab = (1.0 - c.w_cos) * 1.0 + c.w_cos * stab_pure   # 残差主、余弦辅的凸组合
        h_raw = c.h_max * stab * fid
        # 变化率限制(故障恢复)
        if self.h_prev is not None:
            h_raw = _clip(h_raw, self.h_prev - c.delta_max, self.h_prev + c.delta_max)
        # 预算可行性:剩余 b 步须恰好走完 t → 0
        b = max(1, remaining)
        if b <= 1:
            h = t                                # 最后一步直达 0
        else:
            lo = t - (b - 1) * c.h_max           # 后面就算全用最大步也要够
            hi = t - (b - 1) * c.h_min           # 后面全用最小步的上限
            h = _clip(h_raw, max(c.h_min, lo), min(hi, t))
        h = int(round(_clip(h, c.h_min, max(c.h_min, t))))
        self.h_prev = h
        self.used += 1
        return h

    # ── 创新点③:权重协同 ──
    def adapt_weight(self, zeta_base: float) -> float:
        c = self.cfg
        s_use = self.s if self.s == self.s else 1.0
        factor_r = 1.0 + c.gamma_r * self.r_tilde
        factor_s = 1.0 - c.gamma_s * (1.0 - s_use) / 2.0
        return zeta_base * factor_r * max(0.0, factor_s)

    def done(self) -> bool:
        return self.used >= self.N
