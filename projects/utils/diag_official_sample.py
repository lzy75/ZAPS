"""
诊断23(决定性): 用 guided-diffusion【官方 pipeline】跑 ImageNet 无条件生成。

背景(定位到无条件生成能力):
  · 引导有效性数据: ImageNet与FFHQ引导幅度‖corr‖/‖unc‖相近, 但FFHQ中噪区残差降到7.9/PSNR27,
    ImageNet残差卡11.8/PSNR16 → 引导在动, 但对ImageNet没转化成x̂0变好(高频被无条件先验乱填)。
  · 根子在"无条件模型生成能力"。fork说无条件ImageNet生成差是Dhariwal&Nichol已知现象。
  · 但要100%确认: 用【官方create_model_and_diffusion + 官方p_sample_loop】跑, 绕过我们所有代码。

本脚本完全用官方API(不碰我们的diffusion_model/zaps_algorithm/config):
  官方 create_model_and_diffusion(256x256_uncond的FLAGS) 加载同一权重,
  官方 p_sample_loop 跑标准250步无条件生成, 存图+TV。
  判据:
   · 官方pipeline也生成差图/高TV ⇒ 100%坐实无条件ImageNet生成本就差(官方码不可能有bug),
     是模型本性不可修。超分16dB得靠改进引导补偿(学DPS), 而非修无条件生成。
   · 官方pipeline出好图 ⇒ 我们的采样代码有bug(好消息), 逐行对比揪出来。

256x256_diffusion_uncond.pt 官方FLAGS:
  image_size=256 num_channels=256 num_res_blocks=2 attention_resolutions=32,16,8
  num_head_channels=64 use_scale_shift_norm=True resblock_updown=True learn_sigma=True
  class_cond=False use_fp16=True noise_schedule=linear diffusion_steps=1000
采样用 timestep_respacing=250 (官方标准无条件采样步数)。

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_official_sample.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
from guided_diffusion import script_util as su

CKPT = os.path.join(_ROOT, "modules", "models", "256x256_diffusion_uncond.pt")
OUT_DIR = os.path.join(_ROOT, "results", "diag_official")
SEED = 1000
RESPACINGS = ["250", "1000"]   # 官方标准无条件采样步数


def tv255_uint8(arr):
    a = arr.astype(np.float32)
    dh = np.abs(a[1:, :, :] - a[:-1, :, :]).mean()
    dw = np.abs(a[:, 1:, :] - a[:, :-1, :]).mean()
    return dh + dw


def official_flags(respacing):
    f = su.model_and_diffusion_defaults()
    f.update(dict(
        image_size=256, num_channels=256, num_res_blocks=2,
        attention_resolutions="32,16,8", num_head_channels=64,
        use_scale_shift_norm=True, resblock_updown=True, learn_sigma=True,
        class_cond=False, use_fp16=True, noise_schedule="linear",
        diffusion_steps=1000, timestep_respacing=respacing,
    ))
    return f


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*66}\n  官方 pipeline ImageNet 无条件生成 (SEED={SEED})\n{'='*66}", flush=True)
    print(f"  权重: {CKPT}", flush=True)

    for respacing in RESPACINGS:
        flags = official_flags(respacing)
        model, diffusion = su.create_model_and_diffusion(**flags)
        sd = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)
        if flags["use_fp16"]:
            model.convert_to_fp16()
        model.eval()

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model, (1, 3, 256, 256),
                clip_denoised=True, progress=True,
            )
        # sample 在 [-1,1]
        img = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        tv = tv255_uint8(img)
        print(f"\n  [respacing={respacing}] 官方p_sample_loop: TV={tv:.1f}  "
              f"像素 mean={img.mean():.1f} std={img.std():.1f}", flush=True)
        try:
            from PIL import Image
            p = os.path.join(OUT_DIR, f"official_imagenet_{respacing}.png")
            Image.fromarray(img).save(p)
            print(f"  已存: {p}", flush=True)
        except Exception as e:
            np.save(os.path.join(OUT_DIR, f"official_imagenet_{respacing}.npy"), img)
            print(f"  PIL不可用,存npy: {e}", flush=True)

        del model, diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*66}\n  判读 (自然图 TV≈15-40, 纯噪声>60)\n{'='*66}", flush=True)
    print("  · 官方pipeline也高TV/差图 ⇒ 无条件ImageNet生成本就差(官方码无bug), 模型本性不可修。", flush=True)
    print("    超分16dB靠改进引导补偿, 不指望修无条件生成。", flush=True)
    print("  · 官方pipeline出好图 ⇒ 我们采样代码有bug, 逐行对比揪出。", flush=True)
    print(f"  肉眼看 {OUT_DIR}/official_imagenet_*.png", flush=True)


if __name__ == "__main__":
    main()
