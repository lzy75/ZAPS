"""
诊断5: ImageNet 超分——步数扫描(检验"30步不收敛"假设)。

背景链条:
  诊断1: ζ/D 与 FFHQ 量级相同 → 排除引导过冲。
  诊断2: fp16=fp32 → 排除精度。
  诊断3: 无条件30步 FFHQ出图(TV6.6) / ImageNet纯噪声(TV51.7)。
  诊断4: 单步去噪 ImageNet 正常(小t 38dB, 随t优雅退化, 形态同FFHQ) → 先验前向OK。
  超分算子 FFHQ/ImageNet 共用同一份(degradations.py SuperResolutionOperator),
  FFHQ 超分能到 29.77 → 算子本身没问题。
  当前最强假设: 无条件 ImageNet 先验(1000类宽分布)在 30 步内不收敛,
               低频约束把它拉回一部分(所以没到纯噪声), 高频零空间仍是垃圾。

本实验: 同一张 ImageNet 图, 固定种子, 只变采样步数 num_steps=30/60/100
  (schedule 按比例放大, epoch/eta/last_opt 全不变), 比最终 recon PSNR。
  · recon 随步数明显爬升(30→100 涨几个dB) ⇒ 坐实"步数不足", 修复=ImageNet加步数
  · recon 平在 16~17 纹丝不动 ⇒ 排除步数, 不是收敛慢而是轨迹走错, 下一步dump逐步x̂₀

对照: 顺带跑一次 FFHQ 30步, 确认脚本口径与既有实验一致(应≈29.77)。
不改任何磁盘代码, 走真实 run_zaps_single 主路径。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_steps.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from configs.config import IMG_SIZE
from modules.main_single import run_zaps_single, load_image_as_tensor
from modules.degradations import get_operator
from configs.config import TASK_CONFIGS

TASK = "super_resolution"
SEED = 1000
IMG = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}
# (num_steps, schedule) —— schedule 三段和 = num_steps, 按 15/10/5 比例放大
SWEEP = [
    (30,  (15, 10, 5)),
    (60,  (30, 20, 10)),
    (100, (50, 34, 16)),
]


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def obs_psnr(dataset, device):
    """该图 bicubic 上采基线, 作为'什么都不做'的下界参照。"""
    gt = load_image_as_tensor(IMG[dataset]).to(device)
    op = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y = op(gt)
    y_up = F.interpolate(y, size=gt.shape[-2:], mode="bicubic", align_corners=False)
    return to_psnr(((y_up.clamp(-1, 1) - gt) ** 2).mean().item())


def run(dataset, num_steps, schedule, device):
    r = run_zaps_single(
        image_path=IMG[dataset], task=TASK, dataset=dataset, device=device,
        verbose=False, final_mode="last_opt", num_steps=num_steps,
        schedule=schedule, seed=SEED, purpose=f"diag_steps {dataset} S={num_steps}",
    )
    # run_zaps_single 返回 dict; PSNR 字段名兜底探测
    for k in ("psnr", "PSNR", "psnr_recon", "recon_psnr"):
        if k in r:
            return r[k], r
    # 没有现成 PSNR 就自己算
    gt = load_image_as_tensor(IMG[dataset]).to(device)
    x0 = r.get("x0_final")
    if x0 is not None:
        return to_psnr(((x0.clamp(-1, 1) - gt) ** 2).mean().item()), r
    return float("nan"), r


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}\n  ImageNet 超分 步数扫描 (固定种子 {SEED}, last_opt)\n{'='*60}", flush=True)
    ob = obs_psnr("imagenet", device)
    print(f"  bicubic基线(obs) ≈ {ob:.2f} dB  —— 低于此=比什么都不做还差\n", flush=True)
    print(f"  {'num_steps':>10s}{'schedule':>16s}{'recon PSNR':>13s}{'NFE':>8s}", flush=True)
    for S, sch in SWEEP:
        p, r = run("imagenet", S, sch, device)
        nfe = r.get("nfe_total", r.get("nfe_opt", "?"))
        print(f"  {S:10d}{str(sch):>16s}{p:13.2f}{str(nfe):>8s}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  {'-'*40}", flush=True)
    pf, _ = run("ffhq", 30, (15, 10, 5), device)
    print(f"  对照 FFHQ 30步 recon = {pf:.2f} dB (应≈29.77, 验脚本口径)", flush=True)

    print("\n判读:", flush=True)
    print("  · ImageNet recon 随步数明显爬升(30→100 涨几dB) ⇒ 坐实步数不足,"
          " 修复: ImageNet 用更大 num_steps/schedule 重跑四任务", flush=True)
    print("  · recon 平在16~17纹丝不动 ⇒ 排除步数, 是轨迹走错非收敛慢,"
          " 下一步 dump 逐步 x̂₀ 看引导把解推向何处", flush=True)


if __name__ == "__main__":
    main()
