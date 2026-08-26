"""
诊断3: 把先验从反问题机器里剥离 —— ImageNet / FFHQ 纯无条件生成对照。

背景链条:
  诊断1: ImageNet 与 FFHQ 学到的 ζ/D 量级几乎相同 → 排除"引导过冲"。
  诊断2: fp16 与 fp32 recon 完全一致(16.54 vs 16.51) → 排除精度。
  现状: data-consistency loss 已小(0.004), 但 recon(16.5) < bicubic(22~27),
        比"什么都不做"还差 6~10dB → 高频零空间被灌入了错误内容。
        同样的引导/算子/ζ-D, 唯一变量是先验模型本身。

本实验: 令 ζ=0(引导修正恒为0), 复用【真实采样代码路径】(_reverse_diffusion,
        与实验完全一致的 ddpm_posterior_step + Tweedie), 从纯噪声跑完整 30 步,
        得到纯无条件样本。ImageNet vs FFHQ 对照。
  · ImageNet 出连贯自然图(TV 适中, 有结构) ⇒ 先验没问题, 问题在 SR 引导/算子交互
  · ImageNet 出噪声/糊/纯色(TV 异常)         ⇒ 先验调用本身崩(权重虽strict-load成功,
                                             但前向配置与ckpt不匹配) → 查architecture

不改任何磁盘代码; 生成图存 results/diag_uncond_*.png 供肉眼确认。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_uncond.py
"""
import os, sys, glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
from configs.config import ZAPS_CONFIG, TASK_CONFIGS, IMG_SIZE, RESULTS_DIR
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

SEED = 1000
OUT_DIR = os.path.join(RESULTS_DIR, "diag_uncond")
GT = {
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
}


def tv255(x):
    """total-variation, 折算到 [0,255] 尺度(自然图约5~25, 纯噪声>40)。"""
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def save_png(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)[0]           # [C,H,W]
    arr = ((x + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return path
    except Exception as e:
        np.save(path.replace(".png", ".npy"), arr)
        return path.replace(".png", ".npy") + f" (PIL不可用:{e})"


def stats(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    return f"min={x.min():.2f} max={x.max():.2f} mean={x.mean():.2f} std={x.std():.2f}"


def gen_uncond(dataset, device):
    print(f"\n{'='*56}\n  {dataset} 无条件生成 (ζ=0, 纯先验)\n{'='*56}", flush=True)
    # 需要一个 operator 才能构造 ZAPS, 但 ζ=0 时它对轨迹无影响
    operator = get_operator("super_resolution", device=device,
                            **TASK_CONFIGS["super_resolution"])
    dm = load_diffusion_model(dataset, device)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator,
                img_size=IMG_SIZE[0], **ZAPS_CONFIG)
    with torch.no_grad():
        zaps.zeta.zero_()                                  # 关引导 → 纯无条件 DDPM
    # 造一个占位 y(ζ=0 时残差被乘0, 不进轨迹), 形状随便给个降质结果
    dummy_gt = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[0], device=device)
    with torch.no_grad():
        y_dummy = operator(dummy_gt)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    x0, nfe, sec = zaps.sample(y_dummy, eta_override=1.0, init_noise=None, scheduler=None)

    os.makedirs(OUT_DIR, exist_ok=True)
    p = save_png(x0, os.path.join(OUT_DIR, f"uncond_{dataset}.png"))
    print(f"  样本: TV={tv255(x0):.1f}  {stats(x0)}  NFE={nfe} time={sec:.1f}s", flush=True)
    print(f"  已存: {p}", flush=True)

    # 参考: 该数据集一张真实 GT 的 TV
    g = GT.get(dataset)
    if g and os.path.exists(g):
        gt = load_image_as_tensor(g).to(device)
        print(f"  参考(真实GT {os.path.basename(g)}): TV={tv255(gt):.1f}  {stats(gt)}", flush=True)

    del dm, zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tv255(x0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tv_i = gen_uncond("imagenet", device)
    tv_f = gen_uncond("ffhq", device)
    print(f"\n{'='*56}\n  判读\n{'='*56}", flush=True)
    print(f"  无条件样本 TV:  imagenet={tv_i:.1f}   ffhq={tv_f:.1f}", flush=True)
    print("  · 两者都出连贯图(TV 与各自GT接近, 有结构) ⇒ 先验OK, 问题在SR引导/算子交互, 下一步dump逐步x̂₀", flush=True)
    print("  · imagenet TV异常(远高=噪声 / 极低=纯色糊) 而 ffhq 正常 ⇒ ImageNet先验调用崩,"
          " 查前向配置(attention_order/num_heads/channel_mult)与ckpt是否真匹配", flush=True)
    print(f"\n  用图片查看器打开 {OUT_DIR}/uncond_imagenet.png 和 uncond_ffhq.png 肉眼确认。", flush=True)


if __name__ == "__main__":
    main()
