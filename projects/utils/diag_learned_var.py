"""
诊断8/验证: learned variance 修复效果——ImageNet + FFHQ, use_learned_var False vs True。

背景(根因已坐实):
  guided-diffusion 的 FFHQ/ImageNet 模型都是 learn_sigma=True(LEARNED_RANGE),
  每步输出后半是学出的方差。原文 p_sample 用它注噪: sample=mean+exp(0.5·log_var)·z。
  旧实现两个bug: (1) _predict_eps 丢弃后半通道; (2) ddpm_posterior_step 用固定β̃注噪。
  ImageNet(1000类宽分布)高噪区本应学出小方差, 却被统一用偏大β̃ → 注噪过量 →
  宽分布先验扛不住 → 高频噪点(recon16.3<bicubic22.8, TV41≈2×GT)。
  FFHQ窄分布对方差不敏感, 侥幸没崩(29.77)。
  eta扫描佐证: ImageNet eta=0→22.5dB, eta=1→16.5dB, 差的6dB全是注入噪声。

本验证: 同图同种子, 对 imagenet+ffhq 各跑 use_learned_var ∈ {False, True}, 比 recon/TV, 存图。
  判据:
   · ImageNet True 时 recon 从16.3显著上升(目标接近原文23.82), TV从41降到接近GT(21) → 修复成功
   · FFHQ   True 时 recon 保持≈29.77(不掉分) → 无回归; 若掉分则该数据集需回退旧路径

不改磁盘代码, 只在构造 ZAPS 时传 use_learned_var。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_learned_var.py
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
OUT_DIR = os.path.join(RESULTS_DIR, "diag_learned_var")
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


def run(dataset, use_lv, dm, operator, x0_gt, y_obs, device):
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": use_lv}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
    result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
    x0f = result["x0_final"]
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    tvr = tv255(x0f)
    os.makedirs(OUT_DIR, exist_ok=True)
    save_png(x0f, os.path.join(OUT_DIR, f"{dataset}_lv{int(use_lv)}.png"))
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tvr, result["loss_history"][-1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*66}\n  learned variance 修复验证 (SEED={SEED}, SR, last_opt)\n{'='*66}", flush=True)
    summary = []
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
            print(f"[跳过] {ds}: 图不存在 {img}", flush=True)
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
        print(f"\n── {ds}  (bicubic={obs:.2f}dB, GT的TV={tv_gt:.1f}) ──", flush=True)
        print(f"  {'use_learned_var':>16s}{'recon PSNR':>13s}{'TV_recon':>11s}{'loss末':>10s}", flush=True)
        r_false = run(ds, False, dm, operator, x0_gt, y_obs, device)
        print(f"  {'False(旧/固定β̃)':>16s}{r_false[0]:13.2f}{r_false[1]:11.1f}{r_false[2]:10.4f}", flush=True)
        r_true = run(ds, True, dm, operator, x0_gt, y_obs, device)
        print(f"  {'True(新/学出方差)':>16s}{r_true[0]:13.2f}{r_true[1]:11.1f}{r_true[2]:10.4f}", flush=True)
        summary.append((ds, obs, tv_gt, r_false, r_true))
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*66}\n  汇总\n{'='*66}", flush=True)
    print(f"  {'数据集':10s}{'bicubic':>9s}{'旧recon':>9s}{'新recon':>9s}{'Δ':>8s}{'新TV':>7s}{'GT_TV':>7s}", flush=True)
    for ds, obs, tv_gt, rf, rt in summary:
        print(f"  {ds:10s}{obs:9.2f}{rf[0]:9.2f}{rt[0]:9.2f}{rt[0]-rf[0]:+8.2f}{rt[1]:7.1f}{tv_gt:7.1f}", flush=True)
    print("\n判读:", flush=True)
    print("  · ImageNet 新recon 大幅高于旧(目标≈原文23.82)、新TV 接近GT ⇒ learned variance 修复成功", flush=True)
    print("  · FFHQ 新recon ≈ 旧(≈29.77不掉分) ⇒ 无回归, 可全局默认开启", flush=True)
    print("  · 若 FFHQ 新recon 明显掉分 ⇒ 该数据集保留旧路径(use_learned_var=False)", flush=True)
    print(f"\n  存图: {OUT_DIR}/  ({{数据集}}_lv0.png=旧, _lv1.png=新)", flush=True)


if __name__ == "__main__":
    main()
