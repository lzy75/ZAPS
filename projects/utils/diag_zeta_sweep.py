"""
诊断17: ζ 引导强度扫描——手动固定 ζ,不优化,看 ImageNet recon 随引导强度怎么变。

背景(模型/采样/伴随全排除后的最后方向):
  · 模型架构ckpt反推=config=官方FLAGS(彻底排除)。采样全家桶/伴随/fp16/值域全排除。
  · ζ=0引导给ImageNet+9dB(7.3→16.4)、给FFHQ+19dB(10.8→29.8) → 引导对ImageNet效力仅半。
  · optim_curve: ζ只0.10→0.11推不动(梯度gz≈0.02小)。因loss=‖y−A·x̂0‖²已很低(0.004),
    优化器"觉得够好"没动力加大ζ → 但recon差 = loss(数据保真)与recon(重建质量)脱节。
  · 假设: ζ=0.1对ImageNet宽先验太弱,引导推不动重建;优化又没能加大ζ。

本脚本(手动固定ζ、冻结、跳过优化、纯采样): imagenet+ffhq 各扫 ζ∈{0.1,0.3,0.5,1.0,2.0}
  (D固定0.2冻结), 直接 zaps.sample() 看 recon/TV。判据:
   · ImageNet recon 随ζ增大而上升(向23逼近) ⇒ 引导太弱是根因, 修法=调大ζ初值/改损失让优化推大ζ
   · ImageNet recon 平或某ζ后下降 ⇒ 引导强度非根因(存在最优ζ但上限就是16), 是更深的先验/损失问题
   · 对照FFHQ最优ζ位置: 若FFHQ最优在0.1附近而ImageNet最优在更大处 ⇒ 两数据集需不同ζ

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_zeta_sweep.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE, RESULTS_DIR
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

TASK = "super_resolution"
SEED = 1000
ZETAS = [0.1, 0.3, 0.5, 1.0, 2.0]
OUT_DIR = os.path.join(RESULTS_DIR, "diag_zeta_sweep")
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def tv255(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def save_png(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)[0]
    arr = ((x + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        import numpy as np
        np.save(path.replace(".png", ".npy"), arr)


def run_fixed_zeta(dm, operator, x0_gt, y_obs, zeta_val, device):
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False, "sampler_mode": "ddim"}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
    # 手动固定 ζ、D, 冻结, 跳过优化 → 直接采样, 隔离"引导强度"变量
    with torch.no_grad():
        zaps.zeta.fill_(zeta_val)
        zaps.D.fill_(0.2)
    zaps.zeta.requires_grad_(False)
    zaps.D.requires_grad_(False)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    x0f, _, _ = zaps.sample(y_obs, eta_override=None, init_noise=None, scheduler=None)
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    tvr = tv255(x0f)
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tvr, x0f


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*62}\n  ζ 引导强度扫描 (手动固定ζ,冻结,纯采样, SEED={SEED})\n{'='*62}", flush=True)
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
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

        dm = load_diffusion_model(ds, device)
        print(f"\n── {ds}  (bicubic={obs:.2f}dB) ──", flush=True)
        print(f"  {'ζ':>6s}{'recon':>9s}{'TV_recon':>10s}", flush=True)
        best = (-1, -1)
        for z in ZETAS:
            rec, tvr, x0f = run_fixed_zeta(dm, operator, x0_gt, y_obs, z, device)
            mark = ""
            if rec > best[1]:
                best = (z, rec); mark = " ←当前最佳"
            print(f"  {z:6.2f}{rec:9.2f}{tvr:10.1f}{mark}", flush=True)
            save_png(x0f, os.path.join(OUT_DIR, f"{ds}_zeta{z:.1f}.png"))
        print(f"  最佳: ζ={best[0]:.2f}  recon={best[1]:.2f}", flush=True)
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n判读:", flush=True)
    print("  · ImageNet recon 随ζ增大持续上升(向23逼近) ⇒ 引导太弱是根因, 调大ζ即可", flush=True)
    print("  · ImageNet 有最优ζ但峰值仍~16 ⇒ 引导强度非根因, 更深的损失/先验问题", flush=True)
    print("  · FFHQ最优ζ≈0.1而ImageNet最优在更大处 ⇒ 两数据集ζ初值需区分", flush=True)
    print(f"\n  存图: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
