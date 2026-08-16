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
    h_max: float = 120.0         # 最大步长(仅作安全上限)
    # ── v2:幂律基础调度(dense in low-noise,复刻 ZAPS/OSS 先验)──
    p_schedule: float = 2.0      # 幂指数>1:低噪声区步长更小(密采),=1 退化为线性
    # ── 自适应调制(围绕基础调度做有界扰动,信号无效时退化为纯基础调度)──
    beta: float = 0.5            # 残差"下降速率"对步长的调制强度(相对量,非绝对残差)
    w_cos: float = 0.3           # 余弦(稳定性)辅助信号权重∈[0,1];0=纯基础+残差
    mod_min: float = 0.5         # 调制因子下限(防某步被压过小)
    mod_max: float = 1.5         # 调制因子上限(防某步冲过大)
    s_min: float = 0.05          # 余弦稳定因子下限
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

        # ── 1) 幂律基础步长:把 [0,t] 分成 b 段,取第一段长度 ──
        # 线性分段 t/b;幂律让"靠近 0 的段更短" → 用当前占比的幂律权重
        base = t / b                        # 线性基准
        # 幂律修正:高噪声(t 大)放大步、低噪声(t 小)缩小步
        frac = t / max(1.0, self.t_start)   # 当前 t 在全程的占比∈(0,1]
        base = base * (c.p_schedule * frac + (1.0 - frac) * 1.0 / c.p_schedule)

        # ── 2) 有界自适应调制 ──
        # 残差降得快 → 轨迹顺,可迈大步;降得慢/反弹 → 迈小步。用相对降速,无关噪声绝对量
        mod_r = 1.0 + c.beta * (self.dr_rel - 0.0)   # dr_rel>0 加速,<0 减速
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

    # ── 创新点③:权重协同 ──
    def adapt_weight(self, zeta_base: float) -> float:
        c = self.cfg
        s_use = self.s if self.s == self.s else 1.0
        factor_r = 1.0 + c.gamma_r * self.r_tilde
        factor_s = 1.0 - c.gamma_s * (1.0 - s_use) / 2.0
        return zeta_base * factor_r * max(0.0, factor_s)

    def done(self) -> bool:
        return self.used >= self.N
