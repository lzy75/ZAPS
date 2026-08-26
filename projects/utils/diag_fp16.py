"""
诊断2: ImageNet 超分重建 fp16 vs fp32 消融。

背景: 诊断1(diag_zeta_d.py)显示 FFHQ 与 ImageNet 学到的 ζ/D 量级几乎相同,
排除"引导过冲"。但 ImageNet recon(16.3) < obs(22.8) —— 比 bicubic 还差,
而 data-consistency loss 已很小(0.004)。强先验被正确驱动时不该劣于"什么都不做"。

唯一与失败对应的配置差异: ImageNet use_fp16=True, FFHQ use_fp16=False。
Tweedie x̂₀=(x_t−√(1−ᾱ)·ε)/√ᾱ 在高噪声步 √ᾱ≈0.006, 1/√ᾱ≈160,
会把 ε 的 fp16 误差放大约两个数量级 → x̂₀ 崩 → 引导把解推向零空间里的垃圾。

本实验: 同图同种子(SEED=1000), 用同一 ImageNet 权重, 分别以 fp16 / fp32
跑完整 ZAPS(方案1: eta=1, last_opt), 比 recon PSNR。
  · fp32 行 recon 大幅回升(→~24dB 且 > obs) ⇒ 根因 = fp16 精度, 修复=ImageNet 关 fp16
  · fp32 仍崩(< obs)                        ⇒ 排除 fp16, 下一步查数据预处理/先验适配

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_fp16.py
不改动任何磁盘代码, 只在内存里切换 MODEL_CONFIG 的 use_fp16。
"""
import os, sys, glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE, IMAGENET_CKPT
from modules.main_single import load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS
from modules.diffusion_model import ImageNetDiffusionModel

TASK = "super_resolution"
SEED = 1000
IMG_DIR = "/home/lzy/imagenet/256x256"
IMAGES = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:3]


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()  # 值域[-1,1]幅度2→峰值4


def build_model(use_fp16, device):
    # 只切内存里的类属性, 不写磁盘; _load_model 在构造时按此读取
    ImageNetDiffusionModel.MODEL_CONFIG = {**ImageNetDiffusionModel.MODEL_CONFIG,
                                           "use_fp16": use_fp16}
    return ImageNetDiffusionModel(ckpt_path=IMAGENET_CKPT, device=device)


def run(dm, image_path, device):
    x0_gt = load_image_as_tensor(image_path).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y_obs = operator(x0_gt)

    cfg = {**ZAPS_CONFIG, "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"])}
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
    result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
    x0_final = result["x0_final"]

    y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False) \
        if y_obs.shape[-1] != x0_gt.shape[-1] else y_obs
    obs = to_psnr(((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    rec = to_psnr(((x0_final.clamp(-1, 1) - x0_gt) ** 2).mean().item())

    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, obs, result["loss_history"][-1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [p for p in IMAGES if os.path.exists(p)]
    if not imgs:
        print(f"[跳过] 没有可用 ImageNet 图: {IMG_DIR}/*.png", flush=True)
        return
    print(f"图片({len(imgs)}): {[os.path.basename(p) for p in imgs]}", flush=True)

    summary = {}
    for use_fp16 in (True, False):
        tag = "fp16" if use_fp16 else "fp32"
        print(f"\n{'='*56}\n  ImageNet {tag}\n{'='*56}", flush=True)
        dm = build_model(use_fp16, device)
        recs = []
        for p in imgs:
            rec, obs, loss = run(dm, p, device)
            flag = "✓好" if rec > obs else "✗劣于bicubic"
            print(f"  {os.path.basename(p):14s} recon={rec:6.2f}  obs={obs:6.2f}  "
                  f"loss末={loss:.4f}  {flag}", flush=True)
            recs.append(rec)
        summary[tag] = sum(recs) / len(recs)
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*56}\n  汇总(平均 recon PSNR)\n{'='*56}", flush=True)
    print(f"  fp16 = {summary.get('fp16', float('nan')):.2f}   "
          f"fp32 = {summary.get('fp32', float('nan')):.2f}", flush=True)
    print("\n判读:", flush=True)
    print("  · fp32 大幅高于 fp16 且 > obs ⇒ 根因=fp16 精度(高噪步 Tweedie 放大误差),"
          "修复: ImageNet 配置关 use_fp16", flush=True)
    print("  · fp32 与 fp16 相近且仍 < obs ⇒ 排除 fp16, 下一步查数据预处理/值域/先验适配", flush=True)


if __name__ == "__main__":
    main()
