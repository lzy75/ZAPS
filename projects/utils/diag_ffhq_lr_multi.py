"""Paired multi-image confirmation of screened FFHQ learning rates.

This is the second stage after the single-image learning-rate sweep.  It keeps
the recovered historical FFHQ convention fixed (legacy bicubic transpose,
fixed skip variance) and compares only the selected learning rates.  For each
image all rates share the same noisy measurement, x_T, and DDPM draws.
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

from configs.config import TASK_CONFIGS
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from utils.diag_ffhq_regression import TransposeMode, tensor_psnr
from utils.diag_learning_rate_ablation import run_learning_rate, set_seed


TASK = "super_resolution"
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp")


def collect_images(data_dir: str, start_index: int, max_images: int) -> list[str]:
    images = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(data_dir, "**", extension), recursive=True))
    images = sorted(set(images))
    return images[start_index:start_index + max_images]


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm screened ZAPS learning rates on multiple FFHQ images."
    )
    parser.add_argument("--data-dir", default="/home/lzy/FFHQ/00000/00000")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.001, 0.005, 0.01]
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if any(rate <= 0 for rate in args.learning_rates):
        raise ValueError("all learning rates must be positive")
    images = collect_images(args.data_dir, args.start_index, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECTS_ROOT, "results", "diag_ffhq_lr_multi", timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    diffusion_model = load_diffusion_model("ffhq", args.device)
    fieldnames = [
        "index", "image", "seed", "lr", "psnr", "observed_psnr",
        "delta_observed", "final_mse", "final_residual", "zeta_min",
        "zeta_max", "d_delta_rms", "d_delta_max", "seconds",
    ]
    rows = []
    print("\n=== Paired FFHQ multi-image learning-rate confirmation ===", flush=True)
    print(
        f"images={len(images)}; rates={args.learning_rates}; "
        "transpose=legacy_bicubic; variance=fixed; schedule=15-10-5",
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
            base_operator = get_operator(
                TASK, device=args.device, **TASK_CONFIGS[TASK]
            )
            operator = TransposeMode(base_operator, "legacy_bicubic").to(args.device)

            set_seed(seed)
            with torch.no_grad():
                measurement = operator(ground_truth)
            observed_up = F.interpolate(
                measurement,
                size=ground_truth.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            observed_psnr = tensor_psnr(ground_truth, observed_up)

            for learning_rate in args.learning_rates:
                result = run_learning_rate(
                    learning_rate,
                    diffusion_model,
                    operator,
                    measurement,
                    ground_truth,
                    seed,
                    False,
                    False,
                )
                row = {
                    "index": dataset_index,
                    "image": image_path,
                    "seed": seed,
                    "lr": learning_rate,
                    "psnr": result["psnr"],
                    "observed_psnr": observed_psnr,
                    "delta_observed": result["psnr"] - observed_psnr,
                    "final_mse": result["losses"][-1],
                    "final_residual": result["residual"],
                    "zeta_min": result["zeta_min"],
                    "zeta_max": result["zeta_max"],
                    "d_delta_rms": result["d_delta_rms"],
                    "d_delta_max": result["d_delta_max"],
                    "seconds": result["seconds"],
                }
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"[{local_index + 1:02d}/{len(images):02d}] "
                    f"{os.path.basename(image_path)} seed={seed} "
                    f"lr={learning_rate:g} PSNR={result['psnr']:.3f} "
                    f"obs={observed_psnr:.3f} "
                    f"delta={result['psnr'] - observed_psnr:+.3f} "
                    f"time={result['seconds']:.1f}s",
                    flush=True,
                )

            del base_operator, operator, ground_truth, measurement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== Summary by learning rate ===", flush=True)
    print(
        f"{'lr':>9} {'PSNR mean':>11} {'PSNR std':>10} {'median':>10} "
        f"{'delta obs':>11} {'improved':>10} {'MSE mean':>11} {'resid mean':>12}",
        flush=True,
    )
    for learning_rate in args.learning_rates:
        selected = [row for row in rows if row["lr"] == learning_rate]
        psnr_values = [float(row["psnr"]) for row in selected]
        delta_values = [float(row["delta_observed"]) for row in selected]
        mse_values = [float(row["final_mse"]) for row in selected]
        residual_values = [float(row["final_residual"]) for row in selected]
        psnr_mean, psnr_std = mean_std(psnr_values)
        print(
            f"{learning_rate:9.4g} {psnr_mean:11.4f} {psnr_std:10.4f} "
            f"{statistics.median(psnr_values):10.4f} "
            f"{statistics.mean(delta_values):+11.4f} "
            f"{sum(delta > 0 for delta in delta_values):>4}/{len(delta_values):<5} "
            f"{statistics.mean(mse_values):11.6f} "
            f"{statistics.mean(residual_values):12.4f}",
            flush=True,
        )
    print(f"\nresults: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
