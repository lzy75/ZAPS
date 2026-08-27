"""
诊断15: ImageNet 多图 recon 方差——单张 vs 整集平均是否可比。

背景(采样侧+引导算子侧均排除):
  · eta/方差/D/ddim 全没救; 伴随性A虽坏(178%误差)但C修transpose没用(16.38→16.47);
    B ζ=0关引导后ImageNet 7.31/FFHQ 10.79 两者都暴跌 → 引导在"帮忙拉升"非"带崩",
    对两数据集都正常工作。ImageNet被拉到16就到头, FFHQ能到29。
  · 结论转向: 不是某个bug, 是"无条件ImageNet先验+这个引导 对【这张图】的上限就是16"。
  · 一直只测 00000.png 单张! 原文23.82是整验证集【平均】。若这张是复杂多物体图,
    无条件先验本就难, 单张难图 vs 整集平均不可比。

本脚本: 跑 /home/lzy/imagenet/256x256/ 下前N张(有几张跑几张,最多10)超分ZAPS,
  打印每张 recon PSNR + bicubic + TV, 给出均值/中位/范围。判据:
   · 均值接近原文23.82、方差大(某些图20+某些12) ⇒ 之前16.4只是碰上难图, 复现其实成立!
   · 所有图都卡~16 ⇒ 系统性问题, 与图无关, 需继续查(先验适配/引导强度上限)
  顺带打印每张图的 GT TV(高=复杂高频图, 无条件先验更难)。

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_imagenet_multi.py
"""
import os, sys, glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

TASK = "super_resolution"
SEED = 1000
IMG_DIR = "/home/lzy/imagenet/256x256"
MAX_N = 10


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def tv255(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")) +
                  glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                  glob.glob(os.path.join(IMG_DIR, "*.JPEG")))[:MAX_N]
    if not imgs:
        print(f"[跳过] {IMG_DIR} 下没有图", flush=True)
        return
    print(f"\n{'='*66}\n  ImageNet 超分 多图方差 (共{len(imgs)}张, SEED={SEED}, last_opt)\n{'='*66}", flush=True)

    dm = load_diffusion_model("imagenet", device)
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False, "sampler_mode": "ddim"}

    print(f"  {'图':>16s}{'recon':>8s}{'bicubic':>9s}{'Δvs bic':>9s}{'recon_TV':>10s}{'GT_TV':>7s}", flush=True)
    recs, deltas = [], []
    for img in imgs:
        x0_gt = load_image_as_tensor(img).to(device)
        operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            y_obs = operator(x0_gt)
        y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False)
        obs = to_psnr(((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item())

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
        result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
        x0f = result["x0_final"]
        rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
        recs.append(rec); deltas.append(rec - obs)
        print(f"  {os.path.basename(img):>16s}{rec:8.2f}{obs:9.2f}{rec-obs:+9.2f}"
              f"{tv255(x0f):10.1f}{tv255(x0_gt):7.1f}", flush=True)
        del zaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import statistics as st
    print(f"\n  recon: 均值={st.mean(recs):.2f}  中位={st.median(recs):.2f}  "
          f"范围=[{min(recs):.2f}, {max(recs):.2f}]", flush=True)
    print(f"  Δvs bicubic: 均值={st.mean(deltas):+.2f}  ({'超过' if st.mean(deltas)>0 else '劣于'}bicubic)", flush=True)
    print("\n判读:", flush=True)
    print(f"  · 均值接近原文23.82、方差大 ⇒ 之前16.4是碰上难图, 复现其实成立(原文是整集平均)", flush=True)
    print(f"  · 所有图都卡~16、且都劣于bicubic ⇒ 系统性问题与图无关, 继续查先验适配/引导上限", flush=True)
    print(f"  · GT_TV高的图recon更低 ⇒ 复杂高频图无条件先验更难, 佐证'先验上限'解释", flush=True)


if __name__ == "__main__":
    main()
