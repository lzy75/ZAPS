"""
诊断13/验证: 采样更新式 ddpm(后验均值) vs ddim(原文Eq.25) —— ImageNet + FFHQ。

背景(根因定位, 已排除一切其他):
  · 逐步dump显示: FFHQ中噪区(第9-11步)先验接管x̂0从15→25; ImageNet到16卡死不接管。
  · 唯一实质差异(fork从原文Eq.25逐字抠): 采样更新式。
    - 原文Eq.25 (DDIM/ε): x_{t-1}=√ᾱ_prev·x̂0 + √(1-ᾱ_prev-σ²)·ε + σ·z, ε由clamp后x̂0反推
    - 我们旧实现(DDPM后验均值): x_{t-1}=c1·x̂0 + c2·x_t + √β̃·z
    η=1理论等价, 但x̂0被clamp+引导介入后数值发散: 宽分布ImageNet的x̂0高频误差经c2·x_t
    反复带入→先验无法接管; DDIM用干净去噪方向ε替代→先验接管。窄分布FFHQ两形式差异小。

本验证: 同图同种子, imagenet+ffhq 各跑 sampler_mode ∈ {ddpm(旧), ddim(新/原文)}, 比recon/TV存图。
  判据:
   · ImageNet ddim recon从16.7大幅上升(目标≈原文23.85)、TV从41降到接近GT(21) ⇒ 根因坐实, 修复成功
   · FFHQ   ddim recon保持≈29.27(不掉分) ⇒ 无回归, 可全局默认ddim
   · ImageNet ddim仍崩 ⇒ 不是采样形式, 需回查(如ε应否用clamp前/后x̂0)

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_sampler_mode.py
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
OUT_DIR = os.path.join(RESULTS_DIR, "diag_sampler_mode")
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


def run(ds, mode, dm, operator, x0_gt, y_obs, device):
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False, "sampler_mode": mode}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
    result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
    x0f = result["x0_final"]
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    tvr = tv255(x0f)
    save_png(x0f, os.path.join(OUT_DIR, f"{ds}_{mode}.png"))
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tvr, result["loss_history"][-1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*66}\n  采样更新式 ddpm vs ddim(原文Eq.25) (SEED={SEED}, SR, last_opt)\n{'='*66}", flush=True)
    summary = []
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
        print(f"\n── {ds}  (bicubic={obs:.2f}dB, GT_TV={tv_gt:.1f}) ──", flush=True)
        print(f"  {'sampler_mode':>14s}{'recon':>9s}{'TV_recon':>10s}{'loss末':>10s}", flush=True)
        rec_p, tv_p, l_p = run(ds, "ddpm", dm, operator, x0_gt, y_obs, device)
        print(f"  {'ddpm(旧)':>14s}{rec_p:9.2f}{tv_p:10.1f}{l_p:10.4f}", flush=True)
        rec_d, tv_d, l_d = run(ds, "ddim", dm, operator, x0_gt, y_obs, device)
        print(f"  {'ddim(原文Eq25)':>14s}{rec_d:9.2f}{tv_d:10.1f}{l_d:10.4f}", flush=True)
        summary.append((ds, obs, tv_gt, rec_p, tv_p, rec_d, tv_d))
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*66}\n  汇总\n{'='*66}", flush=True)
    print(f"  {'数据集':10s}{'bicubic':>8s}{'ddpm旧':>8s}{'ddim新':>8s}{'Δ':>8s}{'新TV':>7s}{'GT_TV':>7s}", flush=True)
    for ds, obs, tv_gt, rp, tp, rd, td in summary:
        print(f"  {ds:10s}{obs:8.2f}{rp:8.2f}{rd:8.2f}{rd-rp:+8.2f}{td:7.1f}{tv_gt:7.1f}", flush=True)
    print("\n判读:", flush=True)
    print("  · ImageNet ddim recon大幅升(目标≈23.85)、TV降到接近GT ⇒ 采样形式是根因, 修复成功", flush=True)
    print("  · FFHQ ddim ≈29.27不掉分 ⇒ 无回归, 全局默认ddim", flush=True)
    print("  · ImageNet ddim仍崩 ⇒ 另查(ε用clamp前x̂0?/引导项在ddim下的位置?)", flush=True)
    print(f"\n  存图: {OUT_DIR}/  ({{ds}}_ddpm.png 旧, {{ds}}_ddim.png 新)", flush=True)


if __name__ == "__main__":
    main()
