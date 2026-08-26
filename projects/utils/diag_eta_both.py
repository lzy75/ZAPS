"""
诊断9: eta 扫描——ImageNet + FFHQ 同时测(判定 eta=0 是否通用干净解)。

背景(关键转折):
  · learned variance 修复失败: ImageNet 16.4→16.0(更差), 排除"方差大小"是根因。
  · 后验均值系数 c1/c2 已逐项核对 = 原文 q_posterior_mean_variance, 数学无误。
  · 三数据点: ImageNet eta=0→22.5(≈原文23.82), eta=1→16.4(噪)。
    → 问题在"要不要在30稀疏步注随机噪声", 不在方差。强烈怀疑原文采样是确定性(DDIM/eta=0),
      我之前"方案1全程eta=1"的假设可能根本错了。

本实验: imagenet+ffhq 各扫 eta∈{1.0,0.5,0.25,0.0}, 比 recon/TV, 存图。
  关键看 FFHQ:
   · FFHQ eta=0 也≈29.77不掉分 ⇒ eta=0 是两数据集通用干净解, 改默认eta=0即复现
   · FFHQ eta=0 明显掉分/需要eta=1 ⇒ 两数据集采样随机性需求不同, 要区别对待
  (等原文 Algorithm 1 采样公式核对结论一起定论)

use_learned_var 关掉(已证无用), 只扫 eta。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_eta_both.py
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
ETAS = [1.0, 0.5, 0.25, 0.0]
OUT_DIR = os.path.join(RESULTS_DIR, "diag_eta_both")
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}


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
    print(f"\n{'='*62}\n  eta 扫描 ImageNet+FFHQ (SEED={SEED}, SR, last_opt, 固定β̃)\n{'='*62}", flush=True)

    summary = {}
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
            print(f"[跳过] {ds}: {img} 不存在", flush=True)
            continue
        x0_gt = load_image_as_tensor(img).to(device)
        operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            y_obs = operator(x0_gt)
        y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False)
        obs = to_psnr(((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item())
        tv_gt = tv255(x0_gt)

        dm = load_diffusion_model(ds, device)
        cfg0 = {**ZAPS_CONFIG,
                "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
                "use_learned_var": False}
        print(f"\n── {ds}  (bicubic={obs:.2f}dB, GT_TV={tv_gt:.1f}) ──", flush=True)
        print(f"  {'eta':>6s}{'recon':>10s}{'TV_recon':>11s}{'loss末':>10s}", flush=True)
        rows = []
        for eta in ETAS:
            cfg = {**cfg0, "eta": eta}
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
            result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
            x0f = result["x0_final"]
            rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
            tvr = tv255(x0f)
            print(f"  {eta:6.2f}{rec:10.2f}{tvr:11.1f}{result['loss_history'][-1]:10.4f}", flush=True)
            save_png(x0f, os.path.join(OUT_DIR, f"{ds}_eta{eta:.2f}.png"))
            rows.append((eta, rec, tvr))
            del zaps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary[ds] = (obs, tv_gt, rows)
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*62}\n  汇总(各数据集最佳 eta)\n{'='*62}", flush=True)
    for ds, (obs, tv_gt, rows) in summary.items():
        best = max(rows, key=lambda r: r[1])
        print(f"  {ds:10s} bicubic={obs:.2f}  最佳 eta={best[0]:.2f} recon={best[1]:.2f} TV={best[2]:.1f}"
              f"  (eta=1.0 recon={rows[0][1]:.2f})", flush=True)
    print("\n判读:", flush=True)
    print("  · 两数据集都是 eta=0 最好且 FFHQ 不掉分 ⇒ 改默认 eta=0(确定性采样)即复现", flush=True)
    print("  · FFHQ 偏好 eta>0、ImageNet 偏好 eta=0 ⇒ 两数据集需不同 eta, 按数据集配", flush=True)
    print(f"\n  存图: {OUT_DIR}/  ({{数据集}}_eta{{值}}.png)", flush=True)


if __name__ == "__main__":
    main()
