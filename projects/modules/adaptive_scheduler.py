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
