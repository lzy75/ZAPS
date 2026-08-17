"""
自适应调度器 v2 逻辑自检(torch-free,秒级,无需 GPU/数据)

在大规模参数扫描前,先验证 scheduler 的核心逻辑是否正确:
  1. 预算可行性:任意参数/信号下,N 步必须恰好走到 t=0
  2. 基础调度形状:低噪声区(小 t)步长更密(与 ZAPS 15/10/5 先验一致)
  3. 参数单调性:w_cos / beta / p_schedule 改变时,行为按预期方向变化
  4. 退化安全:信号无效(nan)时退化为纯基础调度,不崩
  5. 边界:极小预算 N、极端残差轨迹

用法: python projects/modules/scheduler_selfcheck.py
全部 PASS 才建议进入参数扫描。
"""
import sys, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.adaptive_scheduler import StateAwareScheduler, SchedulerConfig


def simulate(N, T, cfg, resid_fn, cos_fn):
    """跑一条采样轨迹,返回 (终点t, [(t,h)...], [残差...])。"""
    sch = StateAwareScheduler(N, T - 1, cfg)
    t = T - 1; path = []; rs = []
    for k in range(N):
        r = resid_fn(t, k); rs.append(r)
        sch.update_state(resid_norm=r, cos_x0=cos_fn(k))
        h = sch.select_step(t); path.append((t, h)); t -= h
        if t <= 0:
            break
    return t, path, rs


# 残差轨迹模型:高噪声区残差大、随采样下降(真实趋势)
def resid_real(t, k): return max(1.0, 100.0 * (t / 999.0) + (k % 3))
def cos_mid(k): return float("nan") if k < 2 else -0.3
def cos_smooth(k): return float("nan") if k < 2 else 0.8


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def main():
    allok = True
    T = 1000

    # ── 1. 预算可行性:多参数×多 seed×多 N,终点必须=0 ──
    print("1. 预算可行性(终点必须 t=0)")
    bad = 0; total = 0
    import itertools
    for N in (10, 20, 30, 50):
        for wc in (0.0, 0.3, 0.6, 1.0):
            for beta in (0.0, 0.5, 1.0):
                for mm in (1.2, 1.5, 2.0):
                    cfg = SchedulerConfig(w_cos=wc, beta=beta, mod_max=mm)
                    tend, _, _ = simulate(N, T, cfg, resid_real, cos_mid)
                    total += 1
                    if tend != 0: bad += 1
    allok &= check(f"{total} 组参数组合终点全=0", bad == 0, f"失败 {bad} 组")

    # ── 2. 基础调度形状:低噪声区步长更密 ──
    print("2. 基础调度形状(低噪声区应更密)")
    cfg = SchedulerConfig(w_cos=0.0, beta=0.0)   # 关掉自适应,纯看基础调度
    _, path, _ = simulate(30, T, cfg, lambda t, k: 50.0, lambda k: float("nan"))
    hi = [h for t, h in path if t > 666]; lo = [h for t, h in path if t <= 333]
    mhi = st.mean(hi) if hi else 0; mlo = st.mean(lo) if lo else 0
    allok &= check("低噪声区平均步长 < 高噪声区", mlo < mhi,
                   f"高噪={mhi:.1f}(n={len(hi)}) 低噪={mlo:.1f}(n={len(lo)})")

    # ── 3. 参数单调性 ──
    print("3. 参数单调性(改参应产生预期方向变化)")
    # 3a. mod_max 越大,步长方差越大(自适应空间越大)
    def spread(mm):
        c = SchedulerConfig(w_cos=0.0, beta=1.0, mod_max=mm)
        _, p, _ = simulate(30, T, c, resid_real, cos_mid)
        return st.pstdev([h for _, h in p])
    s_narrow, s_wide = spread(1.2), spread(2.5)
    allok &= check("mod_max 增大 → 步长方差增大", s_wide > s_narrow,
                   f"mod_max=1.2 std={s_narrow:.1f}  mod_max=2.5 std={s_wide:.1f}")
    # 3b. w_cos 改变确实改变轨迹(余弦信号真的接入了)
    def traj(wc, cosfn):
        c = SchedulerConfig(w_cos=wc, beta=0.0)
        _, p, _ = simulate(30, T, c, lambda t, k: 50.0, cosfn)
        return [h for _, h in p]
    t_w0 = traj(0.0, cos_smooth); t_w1 = traj(1.0, cos_smooth)
    allok &= check("w_cos=0 与 w_cos=1 轨迹不同(余弦已接入)", t_w0 != t_w1,
                   f"前5步 w0={t_w0[:5]} w1={t_w1[:5]}")
    # 3c. 平滑(cos高)比弯曲(cos低)迈更大步
    sm = traj(1.0, cos_smooth); rg = traj(1.0, lambda k: float("nan") if k < 2 else -0.9)
    allok &= check("余弦高(平滑)平均步长 > 余弦低(弯曲)",
                   st.mean(sm) >= st.mean(rg),
                   f"平滑={st.mean(sm):.1f} 弯曲={st.mean(rg):.1f}")

    # ── 4. 退化安全:全程 nan 信号应退化为基础调度、不崩 ──
    print("4. 退化安全(信号全无效时不崩、仍到 0)")
    tend, path, _ = simulate(30, T, SchedulerConfig(),
                             lambda t, k: 50.0, lambda k: float("nan"))
    allok &= check("全 nan 余弦仍正常到 t=0", tend == 0 and len(path) == 30,
                   f"终点={tend} 步数={len(path)}")

    # ── 5. 边界:极小预算 ──
    print("5. 边界(极小预算 N)")
    okN = all(simulate(N, T, SchedulerConfig(), resid_real, cos_mid)[0] == 0
              for N in (1, 2, 3, 5))
    allok &= check("N∈{1,2,3,5} 均到 t=0", okN)

    # ── 6. v3 统一代价函数专项 ──
    print("6. v3 统一代价评价函数")
    def cfg_v3(**kw): return SchedulerConfig(schedule_mode="v3", **kw)
    # 6a. 预算可行性(多 omega/theta/N)
    bad3 = 0; tot3 = 0
    for N in (10, 20, 30, 50):
        for om in (0.0, 0.5, 1.0):
            for th in (0.3, 0.5, 1.0):
                tot3 += 1
                if simulate(N, T, cfg_v3(omega=om, theta=th), resid_real, cos_mid)[0] != 0:
                    bad3 += 1
    allok &= check(f"v3 预算可行性 {tot3} 组终点全=0", bad3 == 0, f"失败 {bad3}")
    # 6b. 低噪声区更密
    _, p3, _ = simulate(30, T, cfg_v3(), resid_real, cos_mid)
    hi3 = [h for t, h in p3 if t > 666]; lo3 = [h for t, h in p3 if t <= 333]
    allok &= check("v3 低噪声区步长 < 高噪声区",
                   (st.mean(lo3) if lo3 else 9) < (st.mean(hi3) if hi3 else 0),
                   f"高噪={st.mean(hi3):.1f} 低噪={st.mean(lo3):.1f}")
    # 6c. 去趋势:预热常态 EMA 后,异常偏离(骤弯+骤停)应比延续常态迈更小步(密采)
    def h_probe(anomaly):
        s = StateAwareScheduler(30, T - 1, cfg_v3())
        t = 800
        for _ in range(5):                       # 预热 5 步常态,建立 EMA 趋势基准
            s.update_state(resid_norm=50.0, cos_x0=-0.3); s.select_step(t); t -= 30
        if anomaly:                              # 异常:轨迹骤弯 + 残差骤停 → 应密采(小步)
            s.update_state(resid_norm=49.9, cos_x0=-0.9)
        else:                                    # 延续常态 → 跟随 base
            s.update_state(resid_norm=45.0, cos_x0=-0.3)
        return s._select_step_v3(t, 24)
    allok &= check("v3 去趋势:异常(骤弯骤停)步长 < 延续常态(异常处密采)",
                   h_probe(True) < h_probe(False),
                   f"异常={h_probe(True)} 常态={h_probe(False)}")

    print("\n" + ("=" * 40))
    print("总体:", "✅ 全部通过,可进入参数扫描" if allok else "❌ 有 FAIL,先修逻辑再扫参")
    print("=" * 40)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
