"""
诊断7: ImageNet 超分 eta 扫描——定位高频噪声是不是 eta=1 随机注入导致。

背景(截至此步的确定结论):
  · 原文 ImageNet = 30步/(15,10,5)/10ep/300NFE, 与我们完全一致, 原文SR=23.82dB → 步数不是原因。
  · diag2 fp16=fp32 → 排除精度。 diag4 单步去噪38dB正常 → 模型前向OK。
  · diag1 ζ/D 与基线同量级 → 排除引导过冲。
  · 现状 recon 16.33 < bicubic 22.77, TV_recon 41(≈2×GT), loss 0.004 → 低频对、高频是噪声。

假设: 每步 x_uncond 按 eta=1 注入 √β̃·z。窄分布(FFHQ)先验能把噪声拉回流形;
      宽分布(ImageNet 1000类)无条件先验高噪区 score 方向模糊, 注入噪声→流形外随机游走
      →30步 settle 不下来→高频噪声。引导管住低频(loss小)、管不住这些高频。

本实验: 同图同种子, 只变 self.eta ∈ {1.0, 0.5, 0.0}, 走真实 run(last_opt) 主路径,
        比 recon PSNR / TV_recon, 并存 recon 图。
  · eta=0 recon大涨、TV降到接近GT ⇒ 噪声=eta注入, ImageNet需降eta或用模型自带方差(learn_sigma)
  · eta=0 仍噪声(TV仍高、PSNR仍低) ⇒ 非注入问题, 下一步dump逐步x̂₀看引导轨迹

不改磁盘代码, 只在构造 ZAPS 时传不同 eta。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_eta_sweep.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE, RESULTS_DIR
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

TASK = "super_resolution"
SEED = 1000
DATASET = "imagenet"
IMG = "/home/lzy/imagenet/256x256/00000.png"
ETAS = [1.0, 0.5, 0.0]
OUT_DIR = os.path.join(RESULTS_DIR, "diag_eta_sweep")


def tv255(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def save_png(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)[0]
    arr = ((x + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        np.save(path.replace(".png", ".npy"), arr)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    x0_gt = load_image_as_tensor(IMG).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y_obs = operator(x0_gt)
    y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False)
    obs = to_psnr(((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item())

    # 模型只加载一次, 各 eta 复用
    dm = load_diffusion_model(DATASET, device)
    cfg = {**ZAPS_CONFIG, "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"])}

    print(f"\n{'='*60}\n  ImageNet 超分 eta 扫描 (SEED={SEED}, last_opt)\n{'='*60}", flush=True)
    print(f"  参照: bicubic(obs)={obs:.2f}dB  GT的TV={tv255(x0_gt):.1f}\n", flush=True)
    print(f"  {'eta':>5s}{'recon PSNR':>13s}{'TV_recon':>11s}{'loss末':>10s}{'vs bicubic':>12s}", flush=True)

    rows = []
    for eta in ETAS:
        cfg_e = {**cfg, "eta": eta}
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg_e)
        result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
        x0f = result["x0_final"]
        rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
        tvr = tv255(x0f)
        loss = result["loss_history"][-1]
        flag = "✓超过" if rec > obs else "✗劣于"
        print(f"  {eta:5.1f}{rec:13.2f}{tvr:11.1f}{loss:10.4f}{flag:>12s}", flush=True)
        save_png(x0f, os.path.join(OUT_DIR, f"imagenet_sr_eta{eta:.1f}.png"))
        rows.append((eta, rec, tvr))
        del zaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r[1])
    print(f"\n  最佳: eta={best[0]:.1f}  recon={best[1]:.2f}dB  TV={best[2]:.1f}", flush=True)
    print(f"  存图: {OUT_DIR}/imagenet_sr_eta*.png\n", flush=True)
    print("判读:", flush=True)
    print("  · eta=0 recon明显>eta=1 且 TV_recon 降到接近GT ⇒ 高频噪声=eta随机注入,"
          " ImageNet宽分布先验扛不住; 修复方向: ImageNet降eta / 用模型learn_sigma自带方差", flush=True)
    print("  · 各eta recon都低、TV都高 ⇒ 非注入问题, 是引导/轨迹本身走偏, 下一步dump逐步x̂₀", flush=True)


if __name__ == "__main__":
    main()
