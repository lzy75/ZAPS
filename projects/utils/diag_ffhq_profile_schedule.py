"""Paired FFHQ screen for profile-normalized state schedule refinement.

The experiment keeps both requested state signals but changes their role:

* epoch 1 follows the chosen base schedule exactly and records a pilot profile;
* later epochs compare physical residuals with that schedule-specific profile;
* the physical signal decides whether to shrink/expand the next interval;
* x0-trajectory cosine only gates confidence and can never reverse the sign;
* all arms retain 30 steps x 10 epochs and learn both zeta and D normally.

Three initial schedules are screened independently: paper 15-10-5, uniform-30,
and Karras rho=5.  Each refined arm is paired with an adaptive-path null arm
using the same base schedule and random seed.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, RESULTS_DIR, TASK_CONFIGS
from modules.adaptive_scheduler import (
    BudgetedSchedulerConfig,
    BudgetedStateAwareScheduler,
)
from modules.dataset_loader import tensor_to_image
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from utils.diag_ffhq_regression import TransposeMode
from utils.diag_ffhq_state_schedule import finite_json, git_info, run_variant
from utils.diag_ffhq_timestep_ablation import schedule_variants
from utils.diag_learning_rate_ablation import set_seed


TASK = "super_resolution"
NUM_STEPS = 30


def make_profile_scheduler(timesteps, args, response_strength):
    nominal_descending = [int(value) for value in reversed(timesteps.tolist())]
    return BudgetedStateAwareScheduler(
        nominal_descending,
        BudgetedSchedulerConfig(
            # reference_profile 使用“残差定方向 + 余弦置信门控”，不再把
            # 两项直接线性相加；这里保留 PPT 中残差主导的 0.8/0.2 语义。
            residual_weight=0.8,
            cosine_weight=0.2,
            response_strength=response_strength,
            residual_mode="reference_profile",
            profile_warmup_epochs=1,
            profile_center_decay=args.profile_center_decay,
            profile_residual_scale=args.profile_residual_scale,
            profile_cosine_scale=args.profile_cosine_scale,
            profile_cosine_gate=args.profile_cosine_gate,
            weight_mode="identity",
            mod_min=args.mod_min,
            mod_max=args.mod_max,
        ),
    )


def profile_trace(indicators):
    def values(key):
        return [
            float(item[key])
            for item in indicators
            if item.get(key) is not None
            and float(item[key]) == float(item[key])
        ]

    def mean_std(items):
        if not items:
            return float("nan"), float("nan")
        tensor = torch.tensor(items, dtype=torch.float64)
        return tensor.mean().item(), tensor.std(unbiased=False).item()

    result = {}
    for key in (
        "profile_residual_signal",
        "profile_cosine_signal",
        "profile_confidence",
        "step_modifier",
        "h",
    ):
        result[key] = mean_std(values(key))
    return result


def save_outputs(
    output_dir,
    args,
    observed_psnr,
    schedules,
    results,
    reconstructions,
    ground_truth,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    tensor_to_image(ground_truth.detach().cpu().squeeze(0), denormalize=True).save(
        images_dir / "ground_truth.png"
    )
    for name, tensor in reconstructions.items():
        tensor_to_image(tensor.squeeze(0), denormalize=True).save(
            images_dir / f"{name}.png"
        )

    record = {
        "experiment": "ffhq_reference_profile_schedule_screen",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info(),
        "image": os.path.abspath(args.image),
        "seed": args.seed,
        "observed_psnr": observed_psnr,
        "design": {
            "epochs": 10,
            "steps": NUM_STEPS,
            "lr": 0.01,
            "learn_zeta": True,
            "learn_D": True,
            "pilot_epochs": 1,
            "physical_role": "determines signed schedule response",
            "cosine_role": "confidence gate; cannot reverse physical sign",
            "response_strength": args.response_strength,
            "profile_center_decay": args.profile_center_decay,
            "profile_residual_scale": args.profile_residual_scale,
            "profile_cosine_scale": args.profile_cosine_scale,
            "profile_cosine_gate": args.profile_cosine_gate,
            "modifier_bounds": [args.mod_min, args.mod_max],
        },
        "schedules": {
            name: [int(value) for value in tau.tolist()]
            for name, tau in schedules
        },
        "results": results,
    }
    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_json(record), handle, ensure_ascii=False, indent=2)

    fields = [
        "base_schedule", "mode", "psnr", "delta_psnr", "ssim",
        "delta_ssim", "lpips", "delta_lpips", "mse", "residual",
        "nfe", "seconds",
    ]
    by_name = {item["name"]: item for item in results}
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for base_name, _ in schedules:
            null = by_name[f"{base_name}_null"]
            refined = by_name[f"{base_name}_refined"]
            for mode, item in (("null", null), ("refined", refined)):
                writer.writerow({
                    "base_schedule": base_name,
                    "mode": mode,
                    "psnr": item["psnr"],
                    "delta_psnr": item["psnr"] - null["psnr"],
                    "ssim": item["ssim"],
                    "delta_ssim": item["ssim"] - null["ssim"],
                    "lpips": item["lpips"],
                    "delta_lpips": item["lpips"] - null["lpips"],
                    "mse": item["mse"],
                    "residual": item["residual"],
                    "nfe": item["nfe"],
                    "seconds": item["seconds"],
                })


def main():
    parser = argparse.ArgumentParser(
        description="Refine several FFHQ base schedules with both state metrics."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--response-strength", type=float, default=0.15)
    parser.add_argument("--profile-center-decay", type=float, default=0.8)
    parser.add_argument("--profile-residual-scale", type=float, default=0.15)
    parser.add_argument("--profile-cosine-scale", type=float, default=0.2)
    parser.add_argument("--profile-cosine-gate", type=float, default=0.25)
    parser.add_argument("--mod-min", type=float, default=0.85)
    parser.add_argument("--mod-max", type=float, default=1.15)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        RESULTS_DIR, "diag_ffhq_profile_schedule", timestamp
    )

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    base_operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    operator = TransposeMode(base_operator, "legacy_bicubic").to(args.device)
    with torch.no_grad():
        measurement = operator(ground_truth)
        observation = F.interpolate(
            measurement,
            size=ground_truth.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        observed_mse = (
            observation.clamp(-1.0, 1.0) - ground_truth
        ).square().mean()
        observed_psnr = (
            10.0 * torch.log10(4.0 / observed_mse.clamp_min(1e-12))
        ).item()

    diffusion_model = load_diffusion_model("ffhq", args.device)
    all_schedules = schedule_variants(diffusion_model, [], [5.0])
    selected_names = {"paper_15_10_5", "uniform_30", "karras_rho_5"}
    schedules = [
        (name, tau) for name, tau in all_schedules if name in selected_names
    ]
    if len(schedules) != 3:
        raise RuntimeError(f"unexpected schedules: {[name for name, _ in schedules]}")

    print("\n=== Profile-normalized state schedule screen ===", flush=True)
    print(
        "Physical residual decides direction; x0 cosine gates confidence. "
        "Epoch 1 is the fixed pilot; epochs 2-10 refine the base grid.",
        flush=True,
    )
    print(
        f"response={args.response_strength}; modifier=[{args.mod_min},"
        f"{args.mod_max}]; residual_scale={args.profile_residual_scale}; "
        f"cosine_scale={args.profile_cosine_scale}; "
        f"cosine_gate={args.profile_cosine_gate}",
        flush=True,
    )

    results = []
    reconstructions = {}
    for base_name, timesteps in schedules:
        if timesteps.numel() != NUM_STEPS:
            raise RuntimeError(f"{base_name} must contain {NUM_STEPS} steps")
        print(f"\n### Base: {base_name}", flush=True)
        print(f"timesteps: {timesteps.tolist()}", flush=True)
        pair = []
        for mode, response in (("null", 0.0), ("refined", args.response_strength)):
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
                args.seed,
            )
            nominal_descending = [
                int(value) for value in reversed(timesteps.tolist())
            ]
            if mode == "null" and result["visited"] != nominal_descending:
                raise RuntimeError(f"{base_name} null path changed its base grid")
            results.append(result)
            pair.append(result)
            reconstructions[result["name"]] = tensors["reconstruction"]

        null, refined = pair
        trace = profile_trace(refined["indicators"])
        print(
            f"PAIR {base_name}: PSNR {null['psnr']:.4f} -> "
            f"{refined['psnr']:.4f} ({refined['psnr'] - null['psnr']:+.4f}); "
            f"SSIM {refined['ssim'] - null['ssim']:+.4f}; "
            f"LPIPS {refined['lpips'] - null['lpips']:+.4f}",
            flush=True,
        )
        print(
            "  trace refined: "
            f"res_signal={trace['profile_residual_signal'][0]:+.3f}±"
            f"{trace['profile_residual_signal'][1]:.3f}  "
            f"cos_signal={trace['profile_cosine_signal'][0]:+.3f}±"
            f"{trace['profile_cosine_signal'][1]:.3f}  "
            f"confidence={trace['profile_confidence'][0]:.3f}±"
            f"{trace['profile_confidence'][1]:.3f}  "
            f"modifier={trace['step_modifier'][0]:.3f}±"
            f"{trace['step_modifier'][1]:.3f}  "
            f"h_std={trace['h'][1]:.3f}",
            flush=True,
        )
        print(f"  refined visited t: {refined['visited']}", flush=True)

    save_outputs(
        output_dir,
        args,
        observed_psnr,
        schedules,
        results,
        reconstructions,
        ground_truth,
    )

    print("\n=== Paired summary ===", flush=True)
    print(
        f"{'base':>18} {'null PSNR':>10} {'refined':>10} {'delta':>9} "
        f"{'dSSIM':>9} {'dLPIPS':>9}",
        flush=True,
    )
    by_name = {item["name"]: item for item in results}
    for base_name, _ in schedules:
        null = by_name[f"{base_name}_null"]
        refined = by_name[f"{base_name}_refined"]
        print(
            f"{base_name:>18} {null['psnr']:10.4f} {refined['psnr']:10.4f} "
            f"{refined['psnr'] - null['psnr']:+9.4f} "
            f"{refined['ssim'] - null['ssim']:+9.4f} "
            f"{refined['lpips'] - null['lpips']:+9.4f}",
            flush=True,
        )
    print(f"\nRecords saved to: {output_dir}", flush=True)
    print("Decision: require >+0.05 dB and no LPIPS degradation on one base.", flush=True)


if __name__ == "__main__":
    main()
