"""Paired low-step ImageNet experiment on a self-adjoint inverse problem.

This diagnostic deliberately changes both the image and the inverse task used
in the earlier ImageNet super-resolution investigation.  Random inpainting is
self-adjoint (H^T = H), so any result is independent of the disputed SR
transpose convention.

For every requested step budget it compares four arms with the same
measurement, x_T, and DDPM random draws:

1. paper_fixed:   paper Adam learning rate, fixed nominal schedule;
2. paper_refined: paper Adam learning rate, state-refined schedule;
3. tuned_fixed:   transferred tuned learning rate, fixed nominal schedule;
4. tuned_refined: transferred tuned learning rate, state-refined schedule.

The first comparison separates optimizer adaptation from the paper setting.
The second is the actual state-aware scheduling test.  The state-aware arm
uses epoch 1 as a no-cost pilot profile and preserves the exact NFE budget.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import (
    IMG_SIZE,
    METRICS_CONFIG,
    RESULTS_DIR,
    TASK_CONFIGS,
    ZAPS_CONFIG,
    ZETA_INIT_BY_TASK,
)
from modules.adaptive_scheduler import (
    BudgetedSchedulerConfig,
    BudgetedStateAwareScheduler,
)
from modules.dataset_loader import tensor_to_image
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS
from utils.diag_ffhq_state_schedule import finite_json, git_info, trace_diagnostics
from utils.diag_learning_rate_ablation import set_seed
from utils.metrics import compute_all_metrics, compute_psnr


TASK = "inpainting"


def proportional_schedule(num_steps: int) -> tuple[int, int, int]:
    """Scale paper's 15/10/5 section allocation to a new NFE budget."""
    if num_steps < 6:
        raise ValueError("at least 6 steps are required for three nontrivial sections")
    # Keep the paper's 1/2, 1/3, 1/6 proportions.  The remainder is assigned
    # to the high-noise section, where too few points are most destructive.
    low = max(2, int(round(num_steps * 0.5)))
    middle = max(2, int(round(num_steps / 3.0)))
    high = num_steps - low - middle
    if high < 2:
        deficit = 2 - high
        take_low = min(deficit, low - 2)
        low -= take_low
        deficit -= take_low
        middle -= deficit
        high = 2
    if low + middle + high != num_steps or min(low, middle, high) < 1:
        raise RuntimeError(f"invalid proportional schedule for {num_steps} steps")
    return low, middle, high


def make_scheduler(tau: torch.Tensor, args, response_strength: float):
    nominal_descending = [int(value) for value in reversed(tau.tolist())]
    return BudgetedStateAwareScheduler(
        nominal_descending,
        BudgetedSchedulerConfig(
            residual_weight=0.8,
            cosine_weight=0.2,
            response_strength=response_strength,
            residual_mode="reference_profile",
            profile_warmup_epochs=1,
            profile_center_decay=args.profile_center_decay,
            profile_residual_scale=args.profile_residual_scale,
            profile_cosine_scale=args.profile_cosine_scale,
            profile_cosine_gate=args.profile_cosine_gate,
            profile_gate_mode="veto_only",
            profile_cosine_feature_mode=args.cosine_feature,
            profile_cosine_detail_weight=args.cosine_detail_weight,
            weight_mode="identity",
            mod_min=args.mod_min,
            mod_max=args.mod_max,
        ),
    )


def run_arm(
    name: str,
    num_steps: int,
    learning_rate: float,
    response_strength: float,
    diffusion_model,
    operator,
    measurement: torch.Tensor,
    ground_truth: torch.Tensor,
    args,
) -> tuple[dict, torch.Tensor]:
    section_schedule = proportional_schedule(num_steps)
    config = {
        **ZAPS_CONFIG,
        "num_steps": num_steps,
        "schedule": section_schedule,
        "num_epochs": args.epochs,
        "lr": learning_rate,
        "zeta_init": ZETA_INIT_BY_TASK[TASK],
        "d_init": 0.2,
        # Keep the canonical ImageNet ZAPS convention established by the
        # earlier controlled experiments.
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }

    set_seed(args.seed)
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **config,
    )
    scheduler = make_scheduler(zaps.tau.detach().cpu(), args, response_strength)
    nominal_descending = [int(value) for value in reversed(zaps.tau.tolist())]

    print(
        f"\n--- {num_steps:02d} steps | {name} | lr={learning_rate:g} "
        f"| response={response_strength:g} ---",
        flush=True,
    )
    print(
        f"sections={section_schedule}; nominal t={nominal_descending}",
        flush=True,
    )
    started = time.time()
    losses = zaps.optimize(
        measurement,
        verbose=args.verbose,
        x0_gt=ground_truth,
        scheduler=scheduler,
    )
    elapsed = time.time() - started
    reconstruction = zaps._last_opt_x0.detach()
    metrics = compute_all_metrics(
        reconstruction,
        ground_truth,
        lpips_net=METRICS_CONFIG["lpips_net"],
    )
    indicators = list(getattr(zaps, "_indicator_log", []))
    visited = [int(item["t"]) for item in indicators]
    if response_strength == 0.0 and visited != nominal_descending:
        raise RuntimeError(
            f"null path changed the nominal grid: {visited} != {nominal_descending}"
        )
    with torch.no_grad():
        residual = (measurement - operator.H(reconstruction)).flatten().norm().item()
        d_delta = zaps.D.detach() - config["d_init"]

    diagnostics = trace_diagnostics(
        indicators,
        scheduler.cfg.mod_min,
        scheduler.cfg.mod_max,
        scheduler.cfg.weight_min,
        scheduler.cfg.weight_max,
    )
    result = {
        "name": name,
        "steps": num_steps,
        "sections": list(section_schedule),
        "epochs": args.epochs,
        "learning_rate": learning_rate,
        "response_strength": response_strength,
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "lpips": float(metrics["lpips"]),
        "final_mse": float(losses[-1]),
        "final_residual": float(residual),
        "zeta_min": zaps.zeta.detach().min().item(),
        "zeta_max": zaps.zeta.detach().max().item(),
        "d_delta_rms": d_delta.square().mean().sqrt().item(),
        "nfe": int(zaps._last_nfe),
        "seconds": elapsed,
        "nominal_timesteps": nominal_descending,
        "visited_timesteps": visited,
        "indicators": indicators,
        "trace_diagnostics": diagnostics,
    }
    print(
        f"RESULT {name}: PSNR={result['psnr']:.4f} "
        f"SSIM={result['ssim']:.4f} LPIPS={result['lpips']:.4f} "
        f"residual={residual:.4f} NFE={result['nfe']}",
        flush=True,
    )
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, reconstruction.cpu()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor_to_image(tensor.squeeze(0), denormalize=True).save(path)


def validate_args(args) -> None:
    if not args.step_counts:
        raise ValueError("step-counts cannot be empty")
    if len(set(args.step_counts)) != len(args.step_counts):
        raise ValueError("step-counts must be unique")
    if min(args.step_counts) < 6:
        raise ValueError("all step counts must be >= 6")
    if args.epochs < 2:
        raise ValueError("reference-profile refinement needs at least 2 epochs")
    if args.paper_lr <= 0 or args.tuned_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.response_strength < 0:
        raise ValueError("response-strength must be non-negative")
    if not 0 <= args.profile_center_decay < 1:
        raise ValueError("profile-center-decay must be in [0,1)")
    if args.profile_residual_scale <= 0 or args.profile_cosine_scale <= 0:
        raise ValueError("profile scales must be positive")
    if not 0 <= args.profile_cosine_gate <= 1:
        raise ValueError("profile-cosine-gate must be in [0,1]")
    if not 0 <= args.cosine_detail_weight <= 1:
        raise ValueError("cosine-detail-weight must be in [0,1]")
    if not 0 < args.mod_min <= 1 <= args.mod_max:
        raise ValueError("modifier bounds must contain 1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test ImageNet ZAPS and state-aware scheduling at 10/15/20/30 "
            "steps on random inpainting."
        )
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--step-counts", type=int, nargs="+", default=[10, 15, 20, 30]
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--paper-lr", type=float, default=0.001)
    parser.add_argument("--tuned-lr", type=float, default=0.05)
    parser.add_argument("--response-strength", type=float, default=0.20)
    parser.add_argument("--profile-center-decay", type=float, default=0.8)
    parser.add_argument("--profile-residual-scale", type=float, default=0.15)
    parser.add_argument("--profile-cosine-scale", type=float, default=0.20)
    parser.add_argument("--profile-cosine-gate", type=float, default=0.35)
    parser.add_argument(
        "--cosine-feature",
        choices=("global", "dwt_detail", "multiscale"),
        default="multiscale",
    )
    parser.add_argument("--cosine-detail-weight", type=float, default=0.7)
    parser.add_argument("--mod-min", type=float, default=0.8)
    parser.add_argument("--mod-max", type=float, default=1.2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        RESULTS_DIR, "diag_imagenet_lowstep_state", timestamp
    )
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(image_path).to(args.device)
    task_config = {**TASK_CONFIGS[TASK], "seed": args.seed}
    operator = get_operator(TASK, device=args.device, **task_config)
    with torch.no_grad():
        measurement = operator(ground_truth)
        # Show only measured pixels in the observation preview.  The additive
        # Gaussian noise tensor itself also has entries outside the mask, but
        # H^T masks those entries out of every reconstruction update.
        observed_preview = operator.H(measurement)
        observed_psnr = compute_psnr(observed_preview, ground_truth)

    save_image(ground_truth.detach().cpu(), images_dir / "ground_truth.png")
    save_image(observed_preview.detach().cpu(), images_dir / "observation.png")
    diffusion_model = load_diffusion_model("imagenet", args.device)

    print("\n=== ImageNet low-step paired experiment ===", flush=True)
    print(
        f"task={TASK}; image={image_path}; seed={args.seed}; "
        f"steps={args.step_counts}; epochs={args.epochs}",
        flush=True,
    )
    print(
        "H is self-adjoint. Every arm shares y, x_T, and DDPM draws. "
        f"Observed-preview PSNR={observed_psnr:.4f} dB",
        flush=True,
    )

    results = []
    for num_steps in args.step_counts:
        arms = (
            ("paper_fixed", args.paper_lr, 0.0),
            ("paper_refined", args.paper_lr, args.response_strength),
            ("tuned_fixed", args.tuned_lr, 0.0),
            ("tuned_refined", args.tuned_lr, args.response_strength),
        )
        for arm_name, learning_rate, response_strength in arms:
            result, reconstruction = run_arm(
                arm_name,
                num_steps,
                learning_rate,
                response_strength,
                diffusion_model,
                operator,
                measurement,
                ground_truth,
                args,
            )
            results.append(result)
            save_image(
                reconstruction,
                images_dir / f"steps_{num_steps:02d}_{arm_name}.png",
            )

    rows = []
    by_key = {(item["steps"], item["name"]): item for item in results}
    print("\n=== Paired low-step summary ===", flush=True)
    print(
        f"{'steps':>5} {'paper':>9} {'paper+':>9} {'dStateP':>9} "
        f"{'tuned':>9} {'tuned+':>9} {'dStateT':>9}",
        flush=True,
    )
    for num_steps in args.step_counts:
        paper = by_key[(num_steps, "paper_fixed")]
        paper_refined = by_key[(num_steps, "paper_refined")]
        tuned = by_key[(num_steps, "tuned_fixed")]
        refined = by_key[(num_steps, "tuned_refined")]
        row = {
            "steps": num_steps,
            "paper_psnr": paper["psnr"],
            "paper_refined_psnr": paper_refined["psnr"],
            "paper_state_gain_psnr": paper_refined["psnr"] - paper["psnr"],
            "paper_state_gain_ssim": paper_refined["ssim"] - paper["ssim"],
            "paper_state_gain_lpips": paper_refined["lpips"] - paper["lpips"],
            "tuned_fixed_psnr": tuned["psnr"],
            "tuned_refined_psnr": refined["psnr"],
            "optimizer_gain_psnr": tuned["psnr"] - paper["psnr"],
            "tuned_state_gain_psnr": refined["psnr"] - tuned["psnr"],
            "tuned_state_gain_ssim": refined["ssim"] - tuned["ssim"],
            "tuned_state_gain_lpips": refined["lpips"] - tuned["lpips"],
            "paper_ssim": paper["ssim"],
            "paper_refined_ssim": paper_refined["ssim"],
            "tuned_fixed_ssim": tuned["ssim"],
            "tuned_refined_ssim": refined["ssim"],
            "paper_lpips": paper["lpips"],
            "paper_refined_lpips": paper_refined["lpips"],
            "tuned_fixed_lpips": tuned["lpips"],
            "tuned_refined_lpips": refined["lpips"],
            "paper_nfe": paper["nfe"],
            "paper_refined_nfe": paper_refined["nfe"],
            "tuned_fixed_nfe": tuned["nfe"],
            "tuned_refined_nfe": refined["nfe"],
        }
        rows.append(row)
        print(
            f"{num_steps:5d} {paper['psnr']:9.4f} "
            f"{paper_refined['psnr']:9.4f} "
            f"{row['paper_state_gain_psnr']:+9.4f} "
            f"{tuned['psnr']:9.4f} {refined['psnr']:9.4f} "
            f"{row['tuned_state_gain_psnr']:+9.4f}",
            flush=True,
        )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "experiment": "imagenet_lowstep_state_inpainting",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info(),
        "image": image_path,
        "task": TASK,
        "task_config": task_config,
        "seed": args.seed,
        "observed_preview_psnr": observed_psnr,
        "design": {
            "step_counts": args.step_counts,
            "epochs": args.epochs,
            "paper_lr": args.paper_lr,
            "tuned_lr": args.tuned_lr,
            "response_strength": args.response_strength,
            "profile_gate_mode": "veto_only",
            "cosine_feature": args.cosine_feature,
            "cosine_detail_weight": args.cosine_detail_weight,
            "modifier_bounds": [args.mod_min, args.mod_max],
            "success_rule": (
                "state PSNR gain > 0 at >=3/4 budgets in a predeclared LR "
                "branch, including at least one budget <=15; no systematic "
                "SSIM/LPIPS degradation"
            ),
        },
        "summary": rows,
        "results": results,
    }
    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_json(record), handle, ensure_ascii=False, indent=2)

    paper_wins = sum(row["paper_state_gain_psnr"] > 0 for row in rows)
    tuned_wins = sum(row["tuned_state_gain_psnr"] > 0 for row in rows)
    paper_low_wins = sum(
        row["paper_state_gain_psnr"] > 0 and row["steps"] <= 15
        for row in rows
    )
    tuned_low_wins = sum(
        row["tuned_state_gain_psnr"] > 0 and row["steps"] <= 15
        for row in rows
    )
    paper_perceptual_ok = sum(
        row["paper_state_gain_ssim"] >= 0
        and row["paper_state_gain_lpips"] <= 0
        for row in rows
    )
    tuned_perceptual_ok = sum(
        row["tuned_state_gain_ssim"] >= 0
        and row["tuned_state_gain_lpips"] <= 0
        for row in rows
    )
    print(
        "\nDecision counters: "
        f"paper-lr PSNR wins={paper_wins}/{len(rows)}, "
        f"low-step={paper_low_wins}/2, perceptual={paper_perceptual_ok}/{len(rows)}; "
        f"tuned-lr PSNR wins={tuned_wins}/{len(rows)}, "
        f"low-step={tuned_low_wins}/2, perceptual={tuned_perceptual_ok}/{len(rows)}",
        flush=True,
    )
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
