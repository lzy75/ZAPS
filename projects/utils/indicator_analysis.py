"""
创新点① 双重状态感知指标 · 有效性分析

读取 experiments/ 下各实验的 run.json，提取采样阶段记录的双重指标
（cosine_sim 轨迹稳定性 / residual_norm 物理一致性），
统计其与最终重建质量（PSNR / LPIPS）的相关性，验证指标是否有效。

用法:
    python utils/indicator_analysis.py [experiments根目录]
    默认根目录为 configs.config.EXPERIMENTS_DIR

输出:
    - 终端打印每张图的指标聚合量 + 与 PSNR/LPIPS 的 Pearson 相关系数
    - 若无 matplotlib 则跳过画图，仅打印
"""
import os
import sys
import json
import glob
import math


def _load_runs(root: str) -> list:
    """递归读取 root 下所有 run.json（含分组子目录）。"""
    runs = []
    for fp in glob.glob(os.path.join(root, "**", "run.json"), recursive=True):
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("indicators"):
            runs.append(d)
    return runs


def _aggregate(indicators: list) -> dict:
    """把一条采样轨迹的逐步指标聚合成标量特征。"""
    cos = [s["cosine_sim"] for s in indicators
           if s.get("cosine_sim") is not None and not math.isnan(s["cosine_sim"])]
    res = [s["residual_norm"] for s in indicators
           if s.get("residual_norm") is not None]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return {
        "cos_mean": mean(cos),
        "cos_last": cos[-1] if cos else float("nan"),
        "res_mean": mean(res),
        "res_last": res[-1] if res else float("nan"),
    }


def _pearson(xs: list, ys: list) -> float:
    """Pearson 相关系数（忽略含 nan 的配对）。"""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not (math.isnan(x) or math.isnan(y))]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def main(root: str):
    runs = _load_runs(root)
    if not runs:
        print(f"未在 {root} 下找到含 indicators 的 run.json。")
        print("提示: 需用更新后的代码重新跑实验，才会记录双重指标。")
        return

    rows = []
    for d in runs:
        agg = _aggregate(d["indicators"])
        m = d.get("metrics", {})
        cfg = d.get("config", {})
        rows.append({
            "exp_id": d.get("exp_id", "?"),
            "eta": cfg.get("sample_eta", cfg.get("eta", None)),
            "psnr": m.get("psnr", float("nan")),
            "lpips": m.get("lpips", float("nan")),
            **agg,
        })

    # 逐样本表
    print(f"共 {len(rows)} 张图（含指标记录）\n")
    hdr = f"{'exp_id':<44}{'PSNR':>7}{'LPIPS':>8}{'cos_mean':>10}{'res_mean':>11}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['exp_id']:<44}{r['psnr']:>7.2f}{r['lpips']:>8.4f}"
              f"{r['cos_mean']:>10.4f}{r['res_mean']:>11.2f}")

    # 相关性分析：指标特征 vs 重建质量（按 sample_eta 分组，避免辛普森悖论混批）
    feats = ["cos_mean", "cos_last", "res_mean", "res_last"]

    def _corr_table(sub, label):
        print(f"\n=== 指标有效性:Pearson 相关系数  [{label}] n={len(sub)} ===")
        if len(sub) < 3:
            print("  样本 <3,跳过"); return
        print(f"{'指标特征':<12}{'vs PSNR':>10}{'vs LPIPS':>11}")
        print("-" * 33)
        for fkey in feats:
            xs = [r[fkey] for r in sub]
            rp = _pearson(xs, [r["psnr"] for r in sub])
            rl = _pearson(xs, [r["lpips"] for r in sub])
            print(f"{fkey:<12}{rp:>10.3f}{rl:>11.3f}")

    # 按 eta 分组
    etas = sorted({r["eta"] for r in rows}, key=lambda e: (e is None, e))
    if len(etas) > 1:
        print("\n⚠️ 检测到多个 sample_eta,分组单独分析(混批相关系数无效):")
        for e in etas:
            _corr_table([r for r in rows if r["eta"] == e], f"eta={e}")
    else:
        _corr_table(rows, f"eta={etas[0] if etas else '?'}")

    print("\n判读:")
    print("  · 同一 eta 组内:res_* 与 PSNR 负相关、与 LPIPS 正相关 → 残差指标有效")
    print("  · cos_* 与质量显著相关(且 cos_mean/cos_last 同号自洽)→ 稳定性指标有效")
    print("  · 跨 eta 不可直接比较相关系数(采样模式不同,质量信号会在 PSNR/LPIPS 间迁移)")

    # 可选画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter([r["res_mean"] for r in rows], [r["psnr"] for r in rows])
        axes[0].set_xlabel("residual_norm (mean)"); axes[0].set_ylabel("PSNR")
        axes[0].set_title("残差 vs PSNR")
        axes[1].scatter([r["cos_mean"] for r in rows], [r["lpips"] for r in rows])
        axes[1].set_xlabel("cosine_sim (mean)"); axes[1].set_ylabel("LPIPS")
        axes[1].set_title("轨迹稳定性 vs LPIPS")
        out = os.path.join(root, "indicator_correlation.png")
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"\n散点图已保存: {out}")
    except ImportError:
        print("\n(未装 matplotlib,跳过画图)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, _ROOT)
        from configs.config import EXPERIMENTS_DIR
        root = EXPERIMENTS_DIR
    main(root)
