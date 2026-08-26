"""
诊断6: ImageNet 超分重建——存图直接看失败模式(+ FFHQ 对照)。

背景(截至此步的确定结论):
  诊断1 ζ/D与FFHQ同量级 → 排除引导过冲。
  诊断2 fp16=fp32 → 排除精度。
  诊断4 单步去噪ImageNet正常(小t 38dB) → 先验前向OK。
  查原文: 无条件模型+30步+(15,10,5)+10ep 与我们完全一致, 原文ImageNet超分=23.82dB。
  ⇒ 我们的 16.5dB 是实现bug, 非设定问题。bug 在多步引导轨迹里(单步好、多步崩)。

本实验(看失败模式): 同种子跑 ImageNet 超分(last_opt), 把 GT / bicubic上采观测 / 重建
  三张图都存成png, 并打印 TV 与 PSNR。肉眼判失败类型:
  · 重建=高TV噪声(类似诊断3无条件那张)  ⇒ 先验往零空间灌噪, 查多步后验方差/learn_sigma
  · 重建=貌似合理但内容错             ⇒ 先验跑偏到别的自然图, 查引导方向/算子伴随
  · 重建=偏色/整体平移/低频就错        ⇒ 值域或归一化bug(原文[0,1] vs 我们[-1,1])
  · 重建=过糊(TV远低于GT)             ⇒ 引导过强压掉高频 / 后验方差太小
对照跑一张 FFHQ 超分, 应是干净人脸(recon>29)。

不改磁盘代码, 用 ZAPS 类主路径(同 diag_zeta_d)。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_sr_dump.py
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
OUT_DIR = os.path.join(RESULTS_DIR, "diag_sr_dump")
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


def stats(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    return f"min={x.min():.2f} max={x.max():.2f} mean={x.mean():.2f} std={x.std():.2f}"


def save_png(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)[0]
    arr = ((x + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return path
    except Exception as e:
        np.save(path.replace(".png", ".npy"), arr)
        return path.replace(".png", ".npy") + f" (PIL不可用:{e})"


def run_one(dataset, device):
    print(f"\n{'='*60}\n  {dataset} 超分重建 dump\n{'='*60}", flush=True)
    x0_gt = load_image_as_tensor(IMAGES[dataset]).to(device)
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
    x0_final = result["x0_final"]

    y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False)
    obs = to_psnr(((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    rec = to_psnr(((x0_final.clamp(-1, 1) - x0_gt) ** 2).mean().item())

    os.makedirs(OUT_DIR, exist_ok=True)
    p_gt = save_png(x0_gt, os.path.join(OUT_DIR, f"{dataset}_1_gt.png"))
    p_ob = save_png(y_up,  os.path.join(OUT_DIR, f"{dataset}_2_bicubic.png"))
    p_rc = save_png(x0_final, os.path.join(OUT_DIR, f"{dataset}_3_recon.png"))

    print(f"  recon PSNR={rec:.2f}  bicubic PSNR={obs:.2f}  loss末={result['loss_history'][-1]:.4f}", flush=True)
    print(f"  TV:  GT={tv255(x0_gt):.1f}  bicubic={tv255(y_up):.1f}  recon={tv255(x0_final):.1f}", flush=True)
    print(f"  recon幅度: {stats(x0_final)}", flush=True)
    print(f"  GT   幅度: {stats(x0_gt)}", flush=True)
    print(f"  存图: {os.path.dirname(p_rc)}/  ({os.path.basename(p_gt)}, {os.path.basename(p_ob)}, {os.path.basename(p_rc)})", flush=True)

    del dm, zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dataset, rec, obs, tv255(x0_final), tv255(x0_gt)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [run_one(ds, device) for ds in IMAGES if os.path.exists(IMAGES[ds])]

    print(f"\n{'='*60}\n  汇总\n{'='*60}", flush=True)
    print(f"  {'数据集':10s}{'recon':>8s}{'bicubic':>9s}{'TV_recon':>10s}{'TV_gt':>8s}", flush=True)
    for ds, rec, obs, tvr, tvg in rows:
        print(f"  {ds:10s}{rec:8.2f}{obs:9.2f}{tvr:10.1f}{tvg:8.1f}", flush=True)
    print(f"\n  用图片查看器打开 {OUT_DIR}/ 下的 imagenet_3_recon.png —— 一眼看失败模式:", flush=True)
    print("   · 满屏噪点(TV_recon>>TV_gt)      ⇒ 先验灌噪, 查后验方差/learn_sigma", flush=True)
    print("   · 是另一张清晰自然图但不是原图    ⇒ 引导方向/算子伴随问题", flush=True)
    print("   · 偏色/整体发灰/低频就错          ⇒ 值域归一化bug", flush=True)
    print("   · 糊成一团(TV_recon<<TV_gt)      ⇒ 引导过强/方差太小压掉高频", flush=True)


if __name__ == "__main__":
    main()
