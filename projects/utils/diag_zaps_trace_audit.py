"""Trace initial fixed ZAPS trajectories using a saved baseline observation.

No optimizer runs and no parameters change. Each traced trajectory is checked
against ZAPS.sample with the same x_T and posterior draws before interpretation.
This isolates early guidance, clipping and grid effects from adaptation.
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

from configs.config import IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS, ddpm_posterior_step
from modules.dataset_loader import tensor_to_image
from utils.diag_ffhq_regression import TransposeMode, tensor_psnr
from utils.diag_ffhq_state_schedule import finite_json, git_info
from utils.diag_learning_rate_ablation import set_seed


@torch.no_grad()
def trace(zaps, y, ground_truth, init_noise):
    x = init_noise.clone()
    ab = zaps.dm.alphas_cumprod
    rows = []
    for position, index in enumerate(range(len(zaps.tau) - 1, -1, -1)):
        t = int(zaps.tau[index])
        previous_t = int(zaps.tau[index - 1]) if index > 0 else -1
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        if zaps.use_learned_var:
            eps, variance = zaps.dm._predict_eps_var(x, tb)
        else:
            eps, variance = zaps.dm._predict_eps(x, tb), None
        # Unclipped estimate is recorded only; sampling uses the exact core
        # Tweedie helper and posterior helper, including final t=0 behavior.
        raw_x0 = (x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt().clamp_min(1e-8)
        x0 = zaps._tweedie_estimate(x, eps, ab[t], index)
        log_var = (
            zaps._learned_log_var(variance, t, previous_t)
            if variance is not None and previous_t >= 0 else None
        )
        uncond = ddpm_posterior_step(
            x, x0, t, previous_t, ab, eta=zaps.eta,
            learned_log_var=log_var, mode=zaps.sampler_mode,
        )
        residual = y - zaps.A.H(x0)
        v = zaps.A.transpose(residual)
        hv = zaps.dwt.synthesis(zaps.D[index] * zaps.dwt.analysis(v))
        correction = zaps.zeta[index] * (
            v + (1 - ab[t]) * hv
        ) / ab[t].sqrt().clamp_min(1e-8)
        next_x = uncond + correction
        rows.append({
            "step": position,
            "t": t,
            "t_prev": previous_t,
            "jump": t - previous_t,
            "zeta": zaps.zeta[index].item(),
            "sqrt_alpha_bar": ab[t].sqrt().item(),
            "residual_norm": residual.norm().item(),
            "adjoint_residual_norm": v.norm().item(),
            "correction_norm": correction.norm().item(),
            "uncond_norm": uncond.norm().item(),
            "correction_over_uncond": (correction.norm() / uncond.norm().clamp_min(1e-12)).item(),
            "correction_over_uncond_increment": (correction.norm() / (uncond - x).norm().clamp_min(1e-12)).item(),
            "raw_x0_clip_fraction": (raw_x0.abs() > 1).float().mean().item(),
            "x0_psnr": tensor_psnr(ground_truth, x0),
            "next_x_rms": next_x.square().mean().sqrt().item(),
            "input_x_rms": x.square().mean().sqrt().item(),
        })
        x = next_x
    return x, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", choices=("gaussian_deblur", "super_resolution"), default="gaussian_deblur")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = Path(args.run_dir)
    record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run_args = record["arguments"]
    seed = int(run_args["seed"])
    dataset = run_args["dataset"]
    measurement = torch.load(
        source / args.task / "measurement.pt", map_location=args.device, weights_only=True
    )
    ground_truth = load_image_as_tensor(run_args["image"]).to(args.device)
    model = load_diffusion_model(dataset, args.device)
    operator = get_operator(args.task, device=args.device, **record["task_configs"][args.task])
    if args.task == "super_resolution":
        operator = TransposeMode(operator, run_args["sr_transpose"]).to(args.device)
    output_dir = source / ("initial_trace_" + args.task + "_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # There can be more than one LR, but all arms have the same initialization.
    definitions = {}
    for result in record["results"]:
        if result["task"] == args.task:
            definitions.setdefault(result["schedule"], result)
    if len(definitions) != 2:
        raise RuntimeError("source run must contain both fixed schedules for this task")

    traces = {}
    for name, result in definitions.items():
        set_seed(seed)
        zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **result["config"])
        zaps.tau = torch.tensor(result["timesteps_ascending"], device=args.device)
        # Matches optimize(): one x_T draw, followed by each posterior noise.
        init_noise = torch.randn(ground_truth.shape, device=args.device)
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        reconstruction, rows = trace(zaps, measurement, ground_truth, init_noise)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        reference, nfe, _ = zaps.sample(measurement, init_noise=init_noise)
        relative_error = (
            (reconstruction.double() - reference.double()).norm()
            / reference.double().norm().clamp_min(1e-12)
        ).item()
        if relative_error > 1e-5 or nfe != len(rows):
            raise RuntimeError(f"trace/core parity failed: {relative_error=}, {nfe=}")
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tensor_to_image(reconstruction.cpu().squeeze(0), denormalize=True).save(output_dir / f"{name}.png")
        traces[name] = {
            "parity_relative_error": relative_error,
            "config": result["config"],
            "initial_final_psnr": tensor_psnr(ground_truth, reconstruction),
            "source_optimized_psnr": result["psnr"],
            "source_learning_rate": result["learning_rate"],
            "source_loss_history": result["loss_history"],
            "source_zeta_ascending": result["zeta_ascending"],
            "traced_nfe": len(rows),
            "parity_reference_nfe": nfe,
            "rows": rows,
        }
        print(f"\n--- {name} (initial parameters, no training) ---", flush=True)
        print(f"core parity relative error: {relative_error:.3e}", flush=True)
        print(f"{'k':>3} {'t':>4} {'jump':>5} {'corr/unc':>9} {'corr/dunc':>10} {'resid':>9} {'clip%':>7} {'x0PSNR':>8} {'nextRMS':>8}", flush=True)
        for row in rows:
            print(
                f"{row['step']:3d} {row['t']:4d} {row['jump']:5d} "
                f"{row['correction_over_uncond']:9.3f} "
                f"{row['correction_over_uncond_increment']:10.3f} "
                f"{row['residual_norm']:9.2f} "
                f"{100 * row['raw_x0_clip_fraction']:7.1f} "
                f"{row['x0_psnr']:8.2f} {row['next_x_rms']:8.3f}", flush=True,
            )
        print(f"initial final PSNR: {traces[name]['initial_final_psnr']:.4f}", flush=True)
        print(
            f"saved optimized PSNR: {result['psnr']:.4f}; "
            f"loss {result['loss_history'][0]:.6f} -> {result['loss_history'][-1]:.6f}; "
            f"zeta range [{min(result['zeta_ascending']):.5f}, "
            f"{max(result['zeta_ascending']):.5f}]",
            flush=True,
        )
        selected = [
            (int(t), float(zeta))
            for t, zeta in zip(result["timesteps_ascending"], result["zeta_ascending"])
            if int(t) >= 667
        ]
        print(f"saved high-noise zeta: {list(reversed(selected))}", flush=True)
        del zaps, reconstruction, reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (output_dir / "trace.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_json({"git": git_info(), "source": str(source), "task": args.task, "traces": traces}), handle, ensure_ascii=False, indent=2)
    print("\nTrace PSNR is initialization behavior, not the optimized baseline.", flush=True)
    print("No single norm threshold proves overshoot; inspect the shared first-step response and subsequent residual/clipping/PSNR together.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
