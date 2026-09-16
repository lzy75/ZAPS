"""Exact paired test of the measurement value-domain convention.

The diffusion state and score model always remain in [-1, 1].  The two paired
variants differ only in the physical measurement coordinates:

* model_domain: A(x)=H(x), y=A(x)+N(0, 0.05^2)
* unit_domain:  A(x)=H((x+1)/2), y=A(x)+N(0, 0.05^2)

For unit_domain, the adjoint of the affine operator's Jacobian is exactly
0.5*H^T.  Thus the observation, residual, loss, and likelihood gradient are all
in [0, 1], while the diffusion model still receives its required [-1, 1]
state.  Both variants jointly learn zeta and D with the paper's Adam defaults.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, TASK_CONFIGS, ZAPS_CONFIG
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS
from utils.metrics import compute_all_metrics, compute_psnr


TASK = "super_resolution"
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp")
VARIANTS = ("model_domain", "unit_domain")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_images(data_dir: str, start_index: int, max_images: int) -> list[str]:
    images = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(data_dir, "**", extension), recursive=True))
    return sorted(set(images))[start_index:start_index + max_images]


class UnitIntervalMeasurement(nn.Module):
    """Apply the physical operator in [0,1] to a diffusion state in [-1,1]."""

    def __init__(self, base_operator: nn.Module):
        super().__init__()
        self.base = base_operator

    def H(self, x_model: torch.Tensor) -> torch.Tensor:
        return self.base.H((x_model + 1.0) * 0.5)

    def forward(self, x_model: torch.Tensor) -> torch.Tensor:
        return self.H(x_model)

    def transpose(self, residual: torch.Tensor, output_size=None) -> torch.Tensor:
        return 0.5 * self.base.transpose(residual, output_size=output_size)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the current model-domain measurement with an exact [0,1] physical domain."
    )
    parser.add_argument("--data-dir", default="/home/lzy/imagenet/256x256")
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    images = collect_images(args.data_dir, args.start_index, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    output_dir = args.output_dir or os.path.join(
        PROJECTS_ROOT,
        "results",
        "diag_value_domain_exact",
        time.strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    config = {
        **ZAPS_CONFIG,
        "lr": 0.001,
        "zeta_init": 0.1,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }
    diffusion_model = load_diffusion_model("imagenet", args.device)
    rows = []
    fieldnames = [
        "index", "image", "seed", "variant", "psnr", "ssim", "lpips",
        "observed_psnr", "final_mse", "final_residual", "zeta_min",
        "zeta_max", "d_delta_rms", "seconds",
    ]

    print("\n=== Exact value-domain ablation ===", flush=True)
    print(
        f"images={len(images)}; lr=0.001; joint zeta+D; DDPM; fixed variance; sigma=0.05",
        flush=True,
    )
    print(f"CSV: {csv_path}", flush=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for local_index, image_path in enumerate(images):
            dataset_index = args.start_index + local_index
            seed = args.seed_base + dataset_index
            ground_truth = load_image_as_tensor(image_path).to(args.device)

            base_for_noise = get_operator(
                TASK,
                device=args.device,
                **{**TASK_CONFIGS[TASK], "noise_sigma": 0.0},
            )
            set_seed(seed)
            standard_noise = torch.randn_like(base_for_noise.H(ground_truth))

            for variant in VARIANTS:
                base = get_operator(
                    TASK,
                    device=args.device,
                    **{**TASK_CONFIGS[TASK], "noise_sigma": 0.0},
                )
                if variant == "model_domain":
                    operator = base
                    with torch.no_grad():
                        measurement = operator.H(ground_truth) + 0.05 * standard_noise
                    metric_measurement = measurement
                else:
                    operator = UnitIntervalMeasurement(base).to(args.device)
                    with torch.no_grad():
                        measurement = operator.H(ground_truth) + 0.05 * standard_noise
                    metric_measurement = measurement * 2.0 - 1.0

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
                metrics = compute_all_metrics(reconstruction, ground_truth)
                observed_up = F.interpolate(
                    metric_measurement,
                    size=ground_truth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                observed_psnr = compute_psnr(observed_up, ground_truth)

                with torch.no_grad():
                    residual = (
                        measurement - operator.H(reconstruction)
                    ).flatten().norm().item()
                    d_delta = zaps.D.detach() - config["d_init"]
                    row = {
                        "index": dataset_index,
                        "image": image_path,
                        "seed": seed,
                        "variant": variant,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                        "lpips": metrics["lpips"],
                        "observed_psnr": observed_psnr,
                        "final_mse": losses[-1],
                        "final_residual": residual,
                        "zeta_min": zaps.zeta.detach().min().item(),
                        "zeta_max": zaps.zeta.detach().max().item(),
                        "d_delta_rms": d_delta.square().mean().sqrt().item(),
                        "seconds": elapsed,
                    }
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"[{local_index + 1:02d}/{len(images):02d}] {os.path.basename(image_path)} "
                    f"{variant:>12} PSNR={metrics['psnr']:.3f} "
                    f"SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f} "
                    f"obs={observed_psnr:.3f} time={elapsed:.1f}s",
                    flush=True,
                )
                del zaps, operator, base, measurement, reconstruction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del base_for_noise, ground_truth, standard_noise

    print("\n=== Summary ===", flush=True)
    summaries = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        summaries[variant] = {}
        for metric in ("psnr", "ssim", "lpips", "observed_psnr"):
            values = [float(row[metric]) for row in subset]
            average, std = mean_std(values)
            summaries[variant][metric] = average
            print(
                f"{variant:>12} {metric:>13}: mean={average:.4f} std={std:.4f}",
                flush=True,
            )
    delta = summaries["unit_domain"]["psnr"] - summaries["model_domain"]["psnr"]
    print(f"PSNR delta (unit_domain - model_domain): {delta:+.4f} dB", flush=True)
    print(f"results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
