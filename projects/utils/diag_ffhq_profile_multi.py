"""Multi-image confirmation for the profile-normalized FFHQ scheduler.

Every image/schedule pair runs an adaptive-path null and the selected refined
configuration with the same seed.  Reported deltas are paired within image and
within base schedule; absolute scores across different schedules are reported
separately.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import TASK_CONFIGS
from modules.dataset_loader import tensor_to_image
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from utils.diag_ffhq_profile_schedule import make_profile_scheduler
from utils.diag_ffhq_regression import TransposeMode, tensor_psnr
from utils.diag_ffhq_state_schedule import finite_json, git_info, run_variant
from utils.diag_ffhq_timestep_ablation import schedule_variants
from utils.diag_learning_rate_ablation import set_seed


TASK = "super_resolution"
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp")


def collect_images(data_dir: str, start_index: int, max_images: int) -> list[str]:
    images = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(
            glob.glob(os.path.join(data_dir, "**", extension), recursive=True)
        )
    images = sorted(set(images))
    return images[start_index:start_index + max_images]


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm profile-normalized FFHQ schedules on multiple images."
    )
    parser.add_argument("--data-dir", default="/home/lzy/FFHQ/00000/00000")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=["uniform_30", "paper_15_10_5", "karras_rho_3"],
    )
    parser.add_argument("--response-strength", type=float, default=0.20)
    parser.add_argument("--profile-center-decay", type=float, default=0.8)
    parser.add_argument("--profile-residual-scale", type=float, default=0.15)
    parser.add_argument("--profile-cosine-scale", type=float, default=0.2)
    parser.add_argument("--profile-cosine-gate", type=float, default=0.35)
    parser.add_argument(
        "--profile-gate-mode",
        choices=("symmetric", "veto_only"),
        default="veto_only",
    )
    parser.add_argument("--mod-min", type=float, default=0.8)
    parser.add_argument("--mod-max", type=float, default=1.2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.max_images < 1:
        raise ValueError("max-images must be positive")
    if args.response_strength < 0:
        raise ValueError("response-strength must be non-negative")
    if not 0 <= args.profile_center_decay < 1:
        raise ValueError("profile-center-decay must be in [0,1)")
    if args.profile_residual_scale <= 0 or args.profile_cosine_scale <= 0:
        raise ValueError("profile scales must be positive")
    if not 0 <= args.profile_cosine_gate <= 1:
        raise ValueError("profile-cosine-gate must be in [0,1]")
    if not 0 < args.mod_min <= 1 <= args.mod_max:
        raise ValueError("modifier bounds must contain 1")

    images = collect_images(args.data_dir, args.start_index, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        PROJECTS_ROOT, "results", "diag_ffhq_profile_multi", timestamp
    )
    recon_dir = output_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"

    diffusion_model = load_diffusion_model("ffhq", args.device)
    available = schedule_variants(
        diffusion_model,
        [2.0, 3.0],
        [3.0, 5.0, 7.0],
        include_noise_grids=True,
    )
    schedule_map = {name: tau for name, tau in available}
    selected_names = list(dict.fromkeys(args.schedules))
    unknown = [name for name in selected_names if name not in schedule_map]
    if unknown:
        raise ValueError(
            f"unknown schedules {unknown}; available={list(schedule_map)}"
        )
    schedules = [(name, schedule_map[name]) for name in selected_names]

    fieldnames = [
        "index", "image", "seed", "base_schedule", "mode", "psnr", "ssim",
        "lpips", "observed_psnr", "mse", "residual", "nfe", "seconds",
        "reconstruction",
    ]
    rows = []
    print("\n=== Multi-image profile schedule confirmation ===", flush=True)
    print(
        f"images={len(images)}; schedules={selected_names}; "
        f"response={args.response_strength}; gate={args.profile_cosine_gate}; "
        f"gate_mode={args.profile_gate_mode}; "
        f"modifier=[{args.mod_min},{args.mod_max}]",
        flush=True,
    )
    print(f"CSV: {csv_path}", flush=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
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

            for base_name, timesteps in schedules:
                pair = []
                for mode, response in (
                    ("null", 0.0),
                    ("refined", args.response_strength),
                ):
                    scheduler = make_profile_scheduler(timesteps, args, response)
                    result, tensors = run_variant(
                        f"{base_name}_{mode}",
                        scheduler,
                        timesteps,
                        diffusion_model,
                        operator,
                        measurement,
                        ground_truth,
                        args.device,
                        seed,
                        verbose=False,
                    )
                    nominal_descending = [
                        int(value) for value in reversed(timesteps.tolist())
                    ]
                    if mode == "null" and result["visited"] != nominal_descending:
                        raise RuntimeError(
                            f"{base_name} null path changed its base grid"
                        )
                    recon_path = (
                        recon_dir / base_name / mode
                        / f"{dataset_index:05d}_{os.path.basename(image_path)}"
                    )
                    recon_path.parent.mkdir(parents=True, exist_ok=True)
                    tensor_to_image(
                        tensors["reconstruction"].squeeze(0), denormalize=True
                    ).save(recon_path)
                    row = {
                        "index": dataset_index,
                        "image": image_path,
                        "seed": seed,
                        "base_schedule": base_name,
                        "mode": mode,
                        "psnr": result["psnr"],
                        "ssim": result["ssim"],
                        "lpips": result["lpips"],
                        "observed_psnr": observed_psnr,
                        "mse": result["mse"],
                        "residual": result["residual"],
                        "nfe": result["nfe"],
                        "seconds": result["seconds"],
                        "reconstruction": str(recon_path),
                    }
                    rows.append(row)
                    pair.append(row)
                    writer.writerow(row)
                    csv_file.flush()

                null, refined = pair
                print(
                    f"[{local_index + 1:02d}/{len(images):02d}] "
                    f"{os.path.basename(image_path)} seed={seed} {base_name} "
                    f"PSNR={null['psnr']:.3f}->{refined['psnr']:.3f} "
                    f"d={refined['psnr'] - null['psnr']:+.3f} "
                    f"dSSIM={refined['ssim'] - null['ssim']:+.4f} "
                    f"dLPIPS={refined['lpips'] - null['lpips']:+.4f}",
                    flush=True,
                )

            del base_operator, operator, ground_truth, measurement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== Paired multi-image summary ===", flush=True)
    print(
        f"{'base':>18} {'null':>9} {'refined':>9} {'dPSNR':>9} "
        f"{'std':>8} {'wins':>7} {'dSSIM':>9} {'dLPIPS':>9}",
        flush=True,
    )
    summaries = []
    for base_name, _ in schedules:
        null_rows = {
            int(row["index"]): row for row in rows
            if row["base_schedule"] == base_name and row["mode"] == "null"
        }
        refined_rows = {
            int(row["index"]): row for row in rows
            if row["base_schedule"] == base_name and row["mode"] == "refined"
        }
        indices = sorted(set(null_rows) & set(refined_rows))
        d_psnr = [
            float(refined_rows[index]["psnr"])
            - float(null_rows[index]["psnr"])
            for index in indices
        ]
        d_ssim = [
            float(refined_rows[index]["ssim"])
            - float(null_rows[index]["ssim"])
            for index in indices
        ]
        d_lpips = [
            float(refined_rows[index]["lpips"])
            - float(null_rows[index]["lpips"])
            for index in indices
        ]
        null_mean = statistics.mean(
            float(null_rows[index]["psnr"]) for index in indices
        )
        refined_mean = statistics.mean(
            float(refined_rows[index]["psnr"]) for index in indices
        )
        delta_mean, delta_std = mean_std(d_psnr)
        summary = {
            "base_schedule": base_name,
            "count": len(indices),
            "null_psnr_mean": null_mean,
            "refined_psnr_mean": refined_mean,
            "delta_psnr_mean": delta_mean,
            "delta_psnr_std": delta_std,
            "delta_psnr_median": statistics.median(d_psnr),
            "psnr_wins": sum(value > 0 for value in d_psnr),
            "delta_ssim_mean": statistics.mean(d_ssim),
            "delta_lpips_mean": statistics.mean(d_lpips),
        }
        summaries.append(summary)
        print(
            f"{base_name:>18} {null_mean:9.4f} {refined_mean:9.4f} "
            f"{delta_mean:+9.4f} {delta_std:8.4f} "
            f"{summary['psnr_wins']:>3}/{len(indices):<3} "
            f"{summary['delta_ssim_mean']:+9.4f} "
            f"{summary['delta_lpips_mean']:+9.4f}",
            flush=True,
        )

    record = {
        "experiment": "ffhq_profile_schedule_multi",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info(),
        "data_dir": os.path.abspath(args.data_dir),
        "images": images,
        "seed_base": args.seed_base,
        "design": {
            "response_strength": args.response_strength,
            "profile_center_decay": args.profile_center_decay,
            "profile_residual_scale": args.profile_residual_scale,
            "profile_cosine_scale": args.profile_cosine_scale,
            "profile_cosine_gate": args.profile_cosine_gate,
            "profile_gate_mode": args.profile_gate_mode,
            "modifier_bounds": [args.mod_min, args.mod_max],
        },
        "summaries": summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_json(record), handle, ensure_ascii=False, indent=2)
    print(f"\nresults: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
