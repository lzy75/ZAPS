"""Gate and screen PPT-aligned state-aware FFHQ timestep schedules.

The experiment deliberately has two stages:

1. Code gate: a state-aware scheduler with response_strength=0 must reproduce
   the tuned fixed uniform-30 baseline exactly (timesteps, output, learned
   zeta/D, and NFE). A failed gate aborts the experiment.
2. Indicator screen: change timestep placement only. ZAPS still learns both
   zeta and D with the selected FFHQ baseline settings, while the scheduler's
   ``adapt_weight`` is the identity. Residual-only, cosine-only, and
   residual-dominant combinations are compared under paired randomness.

Every run writes a machine-readable JSON record, a compact CSV summary, the
per-step indicator trace, and reconstruction images.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECTS_ROOT)
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, METRICS_CONFIG, RESULTS_DIR, TASK_CONFIGS, ZAPS_CONFIG
from modules.adaptive_scheduler import (
    BudgetedSchedulerConfig,
    BudgetedStateAwareScheduler,
)
from modules.dataset_loader import tensor_to_image
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS
from utils.diag_ffhq_regression import TransposeMode
from utils.diag_ffhq_timestep_ablation import rounded_spacing
from utils.diag_learning_rate_ablation import set_seed
from utils.metrics import compute_all_metrics


TASK = "super_resolution"
NUM_STEPS = 30
NUM_EPOCHS = 10


def git_info() -> dict:
    def run(*args):
        return subprocess.check_output(
            args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()

    try:
        status = run("git", "status", "--porcelain")
        return {
            "commit": run("git", "rev-parse", "--short", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
        }
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def finite_json(value):
    """Convert tensors/non-finite floats into strict JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return finite_json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.double() - right.double()).norm()
    denominator = left.double().norm().clamp_min(1e-12)
    return (numerator / denominator).item()


def make_scheduler(
    nominal_descending: list[int],
    residual_weight: float,
    cosine_weight: float,
    args,
    *,
    response_strength: float | None = None,
):
    return BudgetedStateAwareScheduler(
        nominal_descending,
        cfg=BudgetedSchedulerConfig(
            residual_weight=residual_weight,
            cosine_weight=cosine_weight,
            response_strength=(
                args.response_strength
                if response_strength is None
                else response_strength
            ),
            residual_target_drop=args.residual_target_drop,
            mod_min=args.mod_min,
            mod_max=args.mod_max,
        ),
    )


def run_variant(
    name: str,
    scheduler,
    uniform_tau: torch.Tensor,
    diffusion_model,
    operator,
    measurement: torch.Tensor,
    ground_truth: torch.Tensor,
    device: str,
    seed: int,
) -> tuple[dict, dict]:
    config = {
        **ZAPS_CONFIG,
        "num_steps": NUM_STEPS,
        "num_epochs": NUM_EPOCHS,
        "lr": 0.01,
        "zeta_init": 0.1,
        "d_init": 0.2,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }
    set_seed(seed)
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **config,
    )
    zaps.tau = uniform_tau.to(device)
    print(f"\n--- {name} ---", flush=True)
    started = time.time()
    losses = zaps.optimize(
        measurement,
        verbose=True,
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
    with torch.no_grad():
        residual = (
            measurement - operator.H(reconstruction)
        ).flatten().norm().item()
        d_delta = zaps.D.detach() - config["d_init"]

    if scheduler is None:
        visited = [int(value) for value in reversed(uniform_tau.tolist())]
        steps = [
            visited[index] - visited[index + 1]
            for index in range(len(visited) - 1)
        ] + [1]
        indicators = []
    else:
        indicators = list(getattr(zaps, "_indicator_log", []))
        visited = [int(item["t"]) for item in indicators]
        steps = [int(item["h"]) for item in indicators]

    result = {
        "name": name,
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "lpips": float(metrics["lpips"]),
        "mse": float(losses[-1]),
        "residual": float(residual),
        "zeta_min": zaps.zeta.detach().min().item(),
        "zeta_max": zaps.zeta.detach().max().item(),
        "d_delta_rms": d_delta.square().mean().sqrt().item(),
        "seconds": elapsed,
        "visited": visited,
        "steps": steps,
        "nfe": int(zaps._last_nfe),
        "indicators": indicators,
    }
    tensors = {
        "reconstruction": reconstruction.cpu(),
        "zeta": zaps.zeta.detach().cpu().clone(),
        "D": zaps.D.detach().cpu().clone(),
    }
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, tensors


def evaluate_gate(
    fixed: dict,
    fixed_tensors: dict,
    null: dict,
    null_tensors: dict,
) -> dict:
    checks = {
        "timesteps_equal": fixed["visited"] == null["visited"],
        "nfe_equal": fixed["nfe"] == null["nfe"] == NUM_STEPS * NUM_EPOCHS,
        "output_relative_error": relative_error(
            fixed_tensors["reconstruction"], null_tensors["reconstruction"]
        ),
        "zeta_relative_error": relative_error(
            fixed_tensors["zeta"], null_tensors["zeta"]
        ),
        "D_relative_error": relative_error(fixed_tensors["D"], null_tensors["D"]),
        "psnr_absolute_delta": abs(fixed["psnr"] - null["psnr"]),
    }
    tolerance = 1e-7
    passed = (
        checks["timesteps_equal"]
        and checks["nfe_equal"]
        and checks["output_relative_error"] <= tolerance
        and checks["zeta_relative_error"] <= tolerance
        and checks["D_relative_error"] <= tolerance
        and checks["psnr_absolute_delta"] <= 1e-6
    )
    return {"passed": passed, "tolerance": tolerance, **checks}


def save_records(
    output_dir: Path,
    args,
    uniform_descending: list[int],
    gate: dict,
    results: list[dict],
    reconstructions: dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
    measurement: torch.Tensor,
    observed_psnr: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    tensor_to_image(ground_truth.detach().cpu().squeeze(0), denormalize=True).save(
        images_dir / "ground_truth.png"
    )
    tensor_to_image(measurement.detach().cpu().squeeze(0), denormalize=True).save(
        images_dir / "measurement.png"
    )
    for name, reconstruction in reconstructions.items():
        tensor_to_image(reconstruction.squeeze(0), denormalize=True).save(
            images_dir / f"{name}.png"
        )

    record = {
        "experiment": "ffhq_state_aware_schedule_gate_and_screen",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info(),
        "image": os.path.abspath(args.image),
        "seed": args.seed,
        "observed_psnr": observed_psnr,
        "fixed_baseline": {
            "task": TASK,
            "transpose": "legacy_bicubic",
            "timesteps": "uniform_30",
            "nominal_descending": uniform_descending,
            "lr": 0.01,
            "zeta_init": 0.1,
            "d_init": 0.2,
            "learn_zeta": True,
            "learn_D": True,
            "use_learned_var": False,
            "sampler": "ddpm",
            "epochs": NUM_EPOCHS,
            "NFE": NUM_STEPS * NUM_EPOCHS,
        },
        "state_design": {
            "residual_indicator": (
                "E_r=clip(1-relative_residual_drop/residual_target_drop,0,1)"
            ),
            "cosine_indicator": "E_c=clip((1-cosine_delta_x0)/2,0,1)",
            "combined_score": "E=w_r*E_r+w_c*E_c",
            "step_rule": "h=nominal_h*[1+strength*(0.5-E)*2]",
            "residual_target_drop": args.residual_target_drop,
            "response_strength": args.response_strength,
            "modulation_bounds": [args.mod_min, args.mod_max],
            "zeta_D_rule": "unchanged; scheduler adapt_weight is identity",
        },
        "gate": gate,
        "results": results,
    }
    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_json(record), handle, ensure_ascii=False, indent=2)

    summary_fields = [
        "name", "psnr", "ssim", "lpips", "mse", "residual", "zeta_min",
        "zeta_max", "d_delta_rms", "nfe", "seconds", "psnr_delta_vs_fixed",
    ]
    baseline_psnr = results[0]["psnr"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in summary_fields if key in result}
            row["psnr_delta_vs_fixed"] = result["psnr"] - baseline_psnr
            writer.writerow(row)

    trace_fields = [
        "variant", "step", "t", "t_prev", "h", "parameter_index",
        "residual_norm", "relative_residual_drop", "residual_error",
        "cosine_sim_x0", "cosine_error", "state_score", "base_step",
        "step_modifier",
    ]
    with (output_dir / "indicator_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        for result in results:
            for item in result.get("indicators", []):
                row = {"variant": result["name"]}
                row.update({key: item.get(key) for key in trace_fields if key != "variant"})
                writer.writerow(finite_json(row))


def parse_weight_pairs(values: list[str]) -> list[tuple[float, float]]:
    pairs = []
    for value in values:
        try:
            residual, cosine = (float(item) for item in value.split(":", 1))
        except Exception as exc:
            raise ValueError(f"权重应写成 residual:cosine，收到 {value!r}") from exc
        if residual <= cosine:
            raise ValueError(f"组合权重必须保持残差为主，收到 {value!r}")
        if abs(residual + cosine - 1.0) > 1e-8:
            raise ValueError(f"组合权重之和必须为 1，收到 {value!r}")
        pairs.append((residual, cosine))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate and screen PPT-aligned FFHQ state-aware schedules."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--response-strength", type=float, default=0.5)
    parser.add_argument("--residual-target-drop", type=float, default=0.05)
    parser.add_argument("--mod-min", type=float, default=0.75)
    parser.add_argument("--mod-max", type=float, default=1.25)
    parser.add_argument(
        "--weight-pairs",
        nargs="*",
        default=["0.7:0.3", "0.8:0.2", "0.9:0.1"],
        help="Residual:cosine pairs; residual must be dominant.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional exact output directory; default is a timestamped results directory.",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Run only fixed_uniform and adaptive_null_parity.",
    )
    args = parser.parse_args()

    if args.response_strength < 0:
        raise ValueError("response-strength must be non-negative")
    if args.residual_target_drop <= 0:
        raise ValueError("residual-target-drop must be positive")
    if not 0 < args.mod_min <= 1 <= args.mod_max:
        raise ValueError("modulation bounds must satisfy 0 < min <= 1 <= max")
    weight_pairs = parse_weight_pairs(args.weight_pairs)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        RESULTS_DIR, "diag_ffhq_state_schedule", timestamp
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
        observed_mse = (observation.clamp(-1.0, 1.0) - ground_truth).square().mean()
        observed_psnr = (10.0 * torch.log10(4.0 / observed_mse.clamp_min(1e-12))).item()
    diffusion_model = load_diffusion_model("ffhq", args.device)
    uniform_tau = rounded_spacing(diffusion_model.num_steps, NUM_STEPS, 1.0)
    nominal_descending = [int(value) for value in reversed(uniform_tau.tolist())]

    print("\n=== Stage 1: strict code gate ===", flush=True)
    print(f"nominal timesteps: {nominal_descending}", flush=True)
    fixed, fixed_tensors = run_variant(
        "fixed_uniform", None, uniform_tau, diffusion_model, operator,
        measurement, ground_truth, args.device, args.seed,
    )
    null_scheduler = make_scheduler(
        nominal_descending, 0.8, 0.2, args, response_strength=0.0
    )
    null, null_tensors = run_variant(
        "adaptive_null_parity", null_scheduler, uniform_tau, diffusion_model,
        operator, measurement, ground_truth, args.device, args.seed,
    )
    gate = evaluate_gate(fixed, fixed_tensors, null, null_tensors)
    print("\n=== Gate result ===", flush=True)
    print(json.dumps(finite_json(gate), ensure_ascii=False, indent=2), flush=True)

    results = [fixed, null]
    reconstructions = {
        fixed["name"]: fixed_tensors["reconstruction"],
        null["name"]: null_tensors["reconstruction"],
    }
    if not gate["passed"]:
        save_records(
            output_dir, args, nominal_descending, gate, results,
            reconstructions, ground_truth, measurement, observed_psnr,
        )
        raise RuntimeError(
            f"代码门控失败，已停止状态实验；诊断记录位于 {output_dir}"
        )

    if not args.gate_only:
        print("\n=== Stage 2: paired state-indicator screen ===", flush=True)
        variants = [
            ("residual_only", 1.0, 0.0),
            ("cosine_only", 0.0, 1.0),
            *[
                (f"combined_r{residual:g}_c{cosine:g}", residual, cosine)
                for residual, cosine in weight_pairs
            ],
        ]
        for name, residual_weight, cosine_weight in variants:
            scheduler = make_scheduler(
                nominal_descending, residual_weight, cosine_weight, args
            )
            result, tensors = run_variant(
                name, scheduler, uniform_tau, diffusion_model, operator,
                measurement, ground_truth, args.device, args.seed,
            )
            results.append(result)
            reconstructions[name] = tensors["reconstruction"]

    save_records(
        output_dir, args, nominal_descending, gate, results,
        reconstructions, ground_truth, measurement, observed_psnr,
    )

    baseline_psnr = fixed["psnr"]
    print("\n=== Summary ===", flush=True)
    print(f"observed PSNR: {observed_psnr:.4f} dB", flush=True)
    print(
        f"{'variant':>26} {'PSNR':>9} {'delta':>9} {'SSIM':>8} {'LPIPS':>9} "
        f"{'MSE':>11} {'residual':>10} {'NFE':>6} {'seconds':>9}",
        flush=True,
    )
    for result in results:
        print(
            f"{result['name']:>26} {result['psnr']:9.4f} "
            f"{result['psnr'] - baseline_psnr:+9.4f} "
            f"{result['ssim']:8.4f} {result['lpips']:9.4f} "
            f"{result['mse']:11.6f} {result['residual']:10.4f} "
            f"{result['nfe']:6d} {result['seconds']:9.1f}",
            flush=True,
        )
        print(f"  visited t: {result['visited']}", flush=True)

    print(f"\nRecords saved to: {output_dir}", flush=True)
    print("Decision order:", flush=True)
    print("  1) gate must PASS before interpreting any adaptive result", flush=True)
    print("  2) residual_only vs cosine_only identifies the useful signal", flush=True)
    print("  3) only if a combined variant beats fixed_uniform do we tune strength", flush=True)


if __name__ == "__main__":
    main()
