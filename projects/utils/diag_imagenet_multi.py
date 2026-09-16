"""Evaluate the paper-faithful ImageNet ZAPS baseline on a small subset.

This is a validation run, not a hyper-parameter sweep. It keeps the ZAPS
parameterization described in the paper:

* 30 irregular DDPM steps x 10 epochs (300 NFE)
* fixed DDPM posterior variance as written in the ZAPS sampling equations
* sigma=0.05 in the model's [-1, 1] value domain
* jointly learn per-step zeta and the wavelet Hessian branch D
* Adam's default learning rate 1e-3

Each image gets a deterministic, distinct seed. Results are appended to CSV
after every image so a partially completed run remains usable.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, METRICS_CONFIG, TASK_CONFIGS, ZAPS_CONFIG
from modules.dataset_loader import tensor_to_image
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS
from utils.metrics import compute_all_metrics, compute_psnr


TASK = "super_resolution"
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_images(data_dir: str, start_index: int, max_images: int) -> list[str]:
    images = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(data_dir, "**", extension), recursive=True))
    images = sorted(set(images))
    return images[start_index:start_index + max_images]


def save_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_image(tensor.squeeze(0), denormalize=True).save(path)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the paper-faithful ImageNet ZAPS baseline on multiple images."
    )
    parser.add_argument("--data-dir", default="/home/lzy/imagenet/256x256")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    images = collect_images(args.data_dir, args.start_index, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECTS_ROOT, "results", "diag_imagenet_multi", timestamp
    )
    recon_dir = os.path.join(output_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    config = {
        **ZAPS_CONFIG,
        "lr": args.learning_rate,
        "zeta_init": 0.1,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }
    diffusion_model = load_diffusion_model("imagenet", args.device)

    fieldnames = [
        "index", "image", "seed", "psnr", "ssim", "lpips", "observed_psnr",
        "final_mse", "final_residual", "zeta_min", "zeta_max", "d_delta_rms",
        "d_delta_max", "nfe", "seconds", "reconstruction",
    ]
    rows = []
    print("\n=== ImageNet multi-image validation ===", flush=True)
    print(
        f"images={len(images)}; lr={args.learning_rate:g}; sigma=0.05; "
        "DDPM; fixed posterior variance; joint zeta+D",
        flush=True,
    )
    print(f"CSV: {csv_path}", flush=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for local_index, image_path in enumerate(images):
            dataset_index = args.start_index + local_index
            seed = args.seed_base + dataset_index
            ground_truth = load_image_as_tensor(image_path).to(args.device)
            operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])

            set_seed(seed)
            with torch.no_grad():
                measurement = operator(ground_truth)

            set_seed(seed)
            zaps = ZAPS(
                diffusion_model=diffusion_model,
                forward_operator=operator,
                img_size=IMG_SIZE[0],
                **config,
            )
            started = time.time()
            losses = zaps.optimize(measurement, verbose=False, x0_gt=ground_truth)
            elapsed = time.time() - started
            reconstruction = zaps._last_opt_x0

            metrics = compute_all_metrics(
                reconstruction, ground_truth, lpips_net=METRICS_CONFIG["lpips_net"]
            )
            observed_up = F.interpolate(
                measurement,
                size=ground_truth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            observed_psnr = compute_psnr(observed_up, ground_truth)
            with torch.no_grad():
                residual = (
                    measurement - operator.H(reconstruction)
                ).flatten().norm().item()
                zeta_min = zaps.zeta.detach().min().item()
                zeta_max = zaps.zeta.detach().max().item()
                d_delta = zaps.D.detach() - config["d_init"]
                d_delta_rms = d_delta.square().mean().sqrt().item()
                d_delta_max = d_delta.abs().max().item()

            recon_path = os.path.join(
                recon_dir, f"{dataset_index:05d}_{os.path.basename(image_path)}"
            )
            save_image(reconstruction, recon_path)
            row = {
                "index": dataset_index,
                "image": image_path,
                "seed": seed,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "observed_psnr": observed_psnr,
                "final_mse": losses[-1],
                "final_residual": residual,
                "zeta_min": zeta_min,
                "zeta_max": zeta_max,
                "d_delta_rms": d_delta_rms,
                "d_delta_max": d_delta_max,
                "nfe": config["num_steps"] * config["num_epochs"],
                "seconds": elapsed,
                "reconstruction": recon_path,
            }
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            print(
                f"[{local_index + 1:02d}/{len(images):02d}] "
                f"{os.path.basename(image_path)} seed={seed} "
                f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} "
                f"LPIPS={metrics['lpips']:.4f} obs={observed_psnr:.3f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            del zaps, operator, ground_truth, measurement, reconstruction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== Summary ===", flush=True)
    for metric in ("psnr", "ssim", "lpips", "observed_psnr"):
        values = [float(row[metric]) for row in rows]
        average, std = mean_std(values)
        print(
            f"{metric:>14}: mean={average:.4f} std={std:.4f} "
            f"median={statistics.median(values):.4f} "
            f"range=[{min(values):.4f}, {max(values):.4f}]",
            flush=True,
        )
    print(f"results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
