"""使用官方 guided-diffusion 采样器验证 ImageNet 权重。

用法：
    python projects/modules/generate_imagenet.py
    python projects/modules/generate_imagenet.py --steps 1000
"""

import argparse
import os

import torch
from PIL import Image
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "modules", "models", "256x256_diffusion_uncond.pt"
)
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT, "results", "imagenet_generation_smoke.png"
)


def build_model_and_diffusion(steps: int):
    """按官方 ImageNet 256×256 无条件模型参数创建采样器。"""
    if not 1 <= steps <= 1000:
        raise ValueError(f"steps 应在 1~1000 之间，实际为 {steps}")

    config = model_and_diffusion_defaults()
    config.update(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        class_cond=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        resblock_updown=True,
        use_scale_shift_norm=True,
        use_fp16=True,
        use_new_attention_order=False,
        dropout=0.0,
        timestep_respacing=str(steps),
    )
    return create_model_and_diffusion(**config)


def save_sample(sample: torch.Tensor, output: str):
    """将 [-1, 1] 的单张采样结果保存为 PNG。"""
    array = (
        ((sample[0].float().clamp(-1, 1) + 1.0) * 127.5)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype("uint8")
    )
    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(array).save(output)


def main():
    parser = argparse.ArgumentParser(description="ImageNet 扩散权重无条件生成测试")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="ImageNet 权重路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 PNG 路径")
    parser.add_argument("--steps", type=int, default=250, help="采样步数，建议先 250 再 1000")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda 或 cuda:1")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"ImageNet 权重不存在: {args.checkpoint}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("ImageNet 256×256 FP16 生成测试仅支持 CUDA 设备")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，ImageNet 256×256 生成测试需要 GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[1/3] 创建模型：steps={args.steps} device={device}")
    model, diffusion = build_model_and_diffusion(args.steps)

    print(f"[2/3] 严格加载权重：{args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    model.convert_to_fp16()

    print("[3/3] 开始无条件生成")
    with torch.inference_mode():
        sample = diffusion.p_sample_loop(
            model,
            shape=(1, 3, 256, 256),
            clip_denoised=True,
            model_kwargs={},
            device=device,
            progress=True,
        )

    save_sample(sample, args.output)
    print(f"生成成功：{os.path.abspath(args.output)}")
    print(
        f"shape={tuple(sample.shape)} "
        f"min={sample.min().item():.4f} "
        f"max={sample.max().item():.4f} "
        f"mean={sample.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
