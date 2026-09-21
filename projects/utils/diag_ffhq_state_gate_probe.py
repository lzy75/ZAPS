"""One-unroll parity probe for the FFHQ state-schedule code gate.

This separates forward-path equivalence from optimization drift.  It runs the
fixed path twice and the null state-aware path once with identical seeds,
initial noise, timesteps, and DDPM draws, then compares outputs and raw
zeta/D gradients before Adam or gradient clipping can amplify small errors.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, RESULTS_DIR, TASK_CONFIGS, ZAPS_CONFIG
from modules.adaptive_scheduler import (
    BudgetedSchedulerConfig,
    BudgetedStateAwareScheduler,
)
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS
from utils.diag_ffhq_regression import TransposeMode
from utils.diag_ffhq_timestep_ablation import rounded_spacing
from utils.diag_learning_rate_ablation import set_seed


TASK = "super_resolution"
NUM_STEPS = 30


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = (reference.double() - candidate.double()).norm()
    denominator = reference.double().norm().clamp_min(1e-12)
    return (numerator / denominator).item()


def run_probe(
    name,
    scheduler,
    uniform_tau,
    diffusion_model,
    operator,
    measurement,
    device,
    seed,
):
    config = {
        **ZAPS_CONFIG,
        "num_steps": NUM_STEPS,
        "num_epochs": 10,
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
    fixed_noise = torch.randn(
        measurement.shape[0], 3, IMG_SIZE[0], IMG_SIZE[1], device=device
    )
    nfe = [0]
    started = time.time()
    if scheduler is None:
        output = zaps._reverse_diffusion(
            measurement,
            nfe_counter=nfe,
            eta_override=zaps.eta,
            init_noise=fixed_noise,
        )
        visited = [int(value) for value in reversed(uniform_tau.tolist())]
        parameter_indices = list(range(NUM_STEPS - 1, -1, -1))
    else:
        output = zaps._reverse_diffusion_adaptive(
            measurement,
            scheduler,
            nfe_counter=nfe,
            eta_override=zaps.eta,
            init_noise=fixed_noise,
            record_indicators=True,
        )
        visited = [int(item["t"]) for item in zaps._indicator_log]
        parameter_indices = [
            int(item["parameter_index"]) for item in zaps._indicator_log
        ]
    loss = F.mse_loss(operator.H(output), measurement)
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - started

    tensors = {
        "output": output.detach().cpu(),
        "zeta_grad": zaps.zeta.grad.detach().cpu().clone(),
        "D_grad": zaps.D.grad.detach().cpu().clone(),
    }
    metadata = {
        "name": name,
        "loss": loss.item(),
        "nfe": nfe[0],
        "seconds": elapsed,
        "visited": visited,
        "parameter_indices": parameter_indices,
        "output_norm": tensors["output"].double().norm().item(),
        "zeta_grad_norm": tensors["zeta_grad"].double().norm().item(),
        "D_grad_norm": tensors["D_grad"].double().norm().item(),
    }
    del zaps, output, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, tensors


def compare(reference, candidate):
    ref_meta, ref_tensors = reference
    cand_meta, cand_tensors = candidate
    return {
        "timesteps_equal": ref_meta["visited"] == cand_meta["visited"],
        "parameter_indices_equal": (
            ref_meta["parameter_indices"] == cand_meta["parameter_indices"]
        ),
        "nfe_equal": ref_meta["nfe"] == cand_meta["nfe"] == NUM_STEPS,
        "loss_absolute_delta": abs(ref_meta["loss"] - cand_meta["loss"]),
        "output_relative_error": relative_error(
            ref_tensors["output"], cand_tensors["output"]
        ),
        "zeta_grad_relative_error": relative_error(
            ref_tensors["zeta_grad"], cand_tensors["zeta_grad"]
        ),
        "D_grad_relative_error": relative_error(
            ref_tensors["D_grad"], cand_tensors["D_grad"]
        ),
    }


def classify(repeat, adaptive, tolerance=1e-7):
    error_keys = (
        "output_relative_error",
        "zeta_grad_relative_error",
        "D_grad_relative_error",
    )
    exact_repeat = all(repeat[key] <= tolerance for key in error_keys)
    exact_adaptive = all(adaptive[key] <= tolerance for key in error_keys)
    discrete_equal = all(
        adaptive[key]
        for key in ("timesteps_equal", "parameter_indices_equal", "nfe_equal")
    )
    within_floor = all(
        adaptive[key] <= max(tolerance, 2.0 * repeat[key] + tolerance)
        for key in error_keys
    )
    if exact_repeat and exact_adaptive and discrete_equal:
        return "PASS_EXACT"
    if exact_repeat and not exact_adaptive:
        return "PATH_MISMATCH"
    if not exact_repeat and within_floor and discrete_equal:
        return "GPU_NONDETERMINISM_FLOOR"
    return "PATH_MISMATCH_ABOVE_REPEAT_FLOOR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    base_operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    operator = TransposeMode(base_operator, "legacy_bicubic").to(args.device)
    with torch.no_grad():
        measurement = operator(ground_truth)
    diffusion_model = load_diffusion_model("ffhq", args.device)
    uniform_tau = rounded_spacing(diffusion_model.num_steps, NUM_STEPS, 1.0)
    nominal = [int(value) for value in reversed(uniform_tau.tolist())]

    null_scheduler = BudgetedStateAwareScheduler(
        nominal,
        BudgetedSchedulerConfig(
            residual_weight=0.8,
            cosine_weight=0.2,
            response_strength=0.0,
            residual_target_drop=0.05,
            mod_min=0.75,
            mod_max=1.25,
        ),
    )

    print("=== One-unroll forward/backward parity probe ===", flush=True)
    fixed_a = run_probe(
        "fixed_A", None, uniform_tau, diffusion_model, operator,
        measurement, args.device, args.seed,
    )
    fixed_b = run_probe(
        "fixed_B", None, uniform_tau, diffusion_model, operator,
        measurement, args.device, args.seed,
    )
    adaptive = run_probe(
        "adaptive_null", null_scheduler, uniform_tau, diffusion_model, operator,
        measurement, args.device, args.seed,
    )

    repeat_comparison = compare(fixed_a, fixed_b)
    adaptive_comparison = compare(fixed_a, adaptive)
    classification = classify(repeat_comparison, adaptive_comparison)
    result = {
        "classification": classification,
        "fixed_repeat": repeat_comparison,
        "adaptive_null": adaptive_comparison,
        "runs": [fixed_a[0], fixed_b[0], adaptive[0]],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    output_dir = Path(RESULTS_DIR, "diag_ffhq_state_schedule")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
