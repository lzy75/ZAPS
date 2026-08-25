"""
诊断:对比 ImageNet vs FFHQ 在超分任务下,ZAPS 优化后学到的 ζ_t / D_t 量级,
以及最终采样每步的引导修正强度。用于判定 ImageNet 崩溃是"引导过冲"还是"先验太弱"。

用法(服务器):
    /home/lzy/anaconda3/bin/python3 projects/utils/diag_zeta_d.py

不改变任何实验代码,只读取优化后的参数。baseline 固定调度、last_opt、σ=0.05、seed=1000。
"""
import os, sys
_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

TASK = "super_resolution"
SEED = 1000
IMAGES = {
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
}


def run_one(dataset, image_path, device):
    print(f"\n{'='*60}\n  {dataset}  ({os.path.basename(image_path)})\n{'='*60}", flush=True)
    x0_gt = load_image_as_tensor(image_path).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y_obs = operator(x0_gt)

    dm = load_diffusion_model(dataset, device)
    cfg = {**ZAPS_CONFIG, "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"])}
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)

    result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")

    # 学到的 ζ_t / D_t
    zeta = zaps.zeta.detach().float().cpu()          # [S]
    D = zaps.D.detach().float().cpu()                # [S,C,H,W]
    x0_final = result["x0_final"]

    # 观测基线 PSNR(降质输入 vs GT,统一到 256 比较)
    import torch.nn.functional as F
    y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False) \
        if y_obs.shape[-1] != x0_gt.shape[-1] else y_obs
    obs_mse = ((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item()
    rec_mse = ((x0_final.clamp(-1, 1) - x0_gt) ** 2).mean().item()
    to_psnr = lambda mse: 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()  # 值域[-1,1]幅度2→峰值4

    print(f"  recon PSNR≈{to_psnr(rec_mse):.2f}  obs PSNR≈{to_psnr(obs_mse):.2f}  "
          f"loss末={result['loss_history'][-1]:.4f}", flush=True)
    print(f"  ζ_t: mean={zeta.mean():.4f} max={zeta.max():.4f} min={zeta.min():.4f}", flush=True)
    print(f"       前6步={[round(v,3) for v in zeta[:6].tolist()]}", flush=True)
    print(f"       后6步={[round(v,3) for v in zeta[-6:].tolist()]}", flush=True)
    print(f"  D_t: mean|D|={D.abs().mean():.4f} max|D|={D.abs().max():.4f} "
          f"std={D.std():.4f}", flush=True)
    # D 按步位置的能量(看是不是某些步 D 爆大)
    d_per_step = D.abs().flatten(1).mean(1)          # [S]
    print(f"       每步 mean|D| 前6={[round(v,3) for v in d_per_step[:6].tolist()]}", flush=True)
    print(f"       每步 mean|D| 后6={[round(v,3) for v in d_per_step[-6:].tolist()]}", flush=True)

    del dm, zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(dataset=dataset, zeta_mean=zeta.mean().item(), zeta_max=zeta.max().item(),
                d_mean=D.abs().mean().item(), d_max=D.abs().max().item(),
                recon=to_psnr(rec_mse), obs=to_psnr(obs_mse))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
            print(f"[跳过] {ds}: 图不存在 {img}", flush=True)
            continue
        rows.append(run_one(ds, img, device))

    print(f"\n{'='*60}\n  对比汇总\n{'='*60}", flush=True)
    print(f"{'数据集':10s}{'recon':>8s}{'obs':>8s}{'ζmean':>9s}{'ζmax':>9s}{'|D|mean':>10s}{'|D|max':>10s}", flush=True)
    for r in rows:
        print(f"{r['dataset']:10s}{r['recon']:8.2f}{r['obs']:8.2f}{r['zeta_mean']:9.4f}"
              f"{r['zeta_max']:9.4f}{r['d_mean']:10.4f}{r['d_max']:10.4f}", flush=True)
    print("\n判读:", flush=True)
    print("  · ImageNet 的 ζmax / |D|max 明显大于 FFHQ → 引导过冲(往零空间灌高频)→ 调 ζ/D 上限有救", flush=True)
    print("  · 两者 ζ/D 量级相近、仅 recon 差 → 纯先验弱,非参数问题,需换更强先验或降任务难度", flush=True)


if __name__ == "__main__":
    main()
