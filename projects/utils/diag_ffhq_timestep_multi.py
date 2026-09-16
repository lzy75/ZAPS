"""Paired multi-image confirmation of screened FFHQ timestep schedules.

Compares the paper 15-10-5 schedule against the two single-image finalists:
global uniform spacing and Karras/EDM rho=7.  All ZAPS parameters and the
recovered historical FFHQ operator convention remain fixed.
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
from utils.diag_ffhq_regression import TransposeMode, tensor_psnr
from utils.diag_ffhq_timestep_ablation import schedule_variants
from utils.diag_learning_rate_ablation import set_seed
from utils.metrics import compute_all_metrics


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


def save_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_image(tensor.squeeze(0), denormalize=True).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm screened FFHQ timestep schedules on multiple images."
    )
    parser.add_argument("--data-dir", default="/home/lzy/FFHQ/00000/00000")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--karras-rho", type=float, default=7.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    images = collect_images(args.data_dir, args.start_index, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECTS_ROOT, "results", "diag_ffhq_timestep_multi", timestamp
    )
    recon_dir = os.path.join(output_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    diffusion_model = load_diffusion_model("ffhq", args.device)
    variants = schedule_variants(diffusion_model, [], args.karras_rho)
    selected_names = {"paper_15_10_5", "uniform_30", f"karras_rho_{args.karras_rho:g}"}
    variants = [(name, tau) for name, tau in variants if name in selected_names]
    if len(variants) != 3:
        raise RuntimeError(f"expected three schedule variants, found {[name for name, _ in variants]}")

    config = {
        **ZAPS_CONFIG,
        "lr": 0.01,
        "zeta_init": 0.1,
        "d_init": 0.2,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }
    fieldnames = [
        "index", "image", "seed", "schedule", "psnr", "ssim", "lpips",
        "observed_psnr", "delta_observed", "final_mse", "final_residual",
        "zeta_min", "zeta_max", "d_delta_rms", "seconds", "reconstruction",
    ]
    rows = []
    print("\n=== Paired FFHQ multi-image timestep confirmation ===", flush=True)
    print(
        f"images={len(images)}; schedules={[name for name, _ in variants]}; "
        "lr=0.01; zeta_init=0.1; D_init=0.2; legacy transpose; fixed variance",
        flush=True,
    )
    for name, tau in variants:
        print(f"{name}: {tau.tolist()}", flush=True)
    print(f"CSV: {csv_path}", flush=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for local_index, image_path in enumerate(images):
            dataset_index = args.start_index + local_index
            seed = args.seed_base + dataset_index
            ground_truth = load_image_as_tensor(image_path).to(args.device)
            base_operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
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

            for schedule_name, timesteps in variants:
                set_seed(seed)
                zaps = ZAPS(
                    diffusion_model=diffusion_model,
                    forward_operator=operator,
                    img_size=IMG_SIZE[0],
                    **config,
                )
                zaps.tau = timesteps.to(args.device)
                started = time.time()
                losses = zaps.optimize(measurement, verbose=False, x0_gt=ground_truth)
                elapsed = time.time() - started
                reconstruction = zaps._last_opt_x0
                metrics = compute_all_metrics(
                    reconstruction,
                    ground_truth,
                    lpips_net=METRICS_CONFIG["lpips_net"],
                )
                with torch.no_grad():
                    residual = (
                        measurement - operator.H(reconstruction)
                    ).flatten().norm().item()
                    d_delta = zaps.D.detach() - config["d_init"]

                recon_path = os.path.join(
                    recon_dir,
                    schedule_name,
                    f"{dataset_index:05d}_{os.path.basename(image_path)}",
                )
                save_image(reconstruction, recon_path)
                row = {
                    "index": dataset_index,
                    "image": image_path,
                    "seed": seed,
                    "schedule": schedule_name,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                    "observed_psnr": observed_psnr,
                    "delta_observed": metrics["psnr"] - observed_psnr,
                    "final_mse": losses[-1],
                    "final_residual": residual,
                    "zeta_min": zaps.zeta.detach().min().item(),
                    "zeta_max": zaps.zeta.detach().max().item(),
                    "d_delta_rms": d_delta.square().mean().sqrt().item(),
                    "seconds": elapsed,
                    "reconstruction": recon_path,
                }
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"[{local_index + 1:02d}/{len(images):02d}] "
                    f"{os.path.basename(image_path)} seed={seed} "
                    f"{schedule_name} PSNR={metrics['psnr']:.3f} "
                    f"SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f} "
                    f"delta={metrics['psnr'] - observed_psnr:+.3f} "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )
                del zaps, reconstruction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del base_operator, operator, ground_truth, measurement

    print("\n=== Summary by schedule ===", flush=True)
    print(
        f"{'schedule':>20} {'PSNR mean':>11} {'std':>8} {'median':>9} "
        f"{'SSIM':>8} {'LPIPS':>9} {'delta obs':>10} {'improved':>10} "
        f"{'MSE':>10} {'residual':>10}",
        flush=True,
    )
    for schedule_name, _ in variants:
        selected = [row for row in rows if row["schedule"] == schedule_name]
        psnr_values = [float(row["psnr"]) for row in selected]
        psnr_mean, psnr_std = mean_std(psnr_values)
        delta_values = [float(row["delta_observed"]) for row in selected]
        print(
            f"{schedule_name:>20} {psnr_mean:11.4f} {psnr_std:8.4f} "
            f"{statistics.median(psnr_values):9.4f} "
            f"{statistics.mean(float(row['ssim']) for row in selected):8.4f} "
            f"{statistics.mean(float(row['lpips']) for row in selected):9.4f} "
            f"{statistics.mean(delta_values):+10.4f} "
            f"{sum(delta > 0 for delta in delta_values):>4}/{len(delta_values):<5} "
            f"{statistics.mean(float(row['final_mse']) for row in selected):10.6f} "
            f"{statistics.mean(float(row['final_residual']) for row in selected):10.4f}",
            flush=True,
        )
    print(f"\nresults: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
