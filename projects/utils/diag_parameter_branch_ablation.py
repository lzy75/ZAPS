"""Paired ablation of the learnable ZAPS parameter branches.

Runs the same 30-step x 10-epoch trajectory three times while changing only
which parameters Adam may update:

* both: learn per-timestep zeta and wavelet Hessian diagonals D.
* zeta_only: learn zeta while D remains fixed at its 0.2 initialization.
* d_only: learn D while zeta remains fixed at its 0.1 initialization.

The observation, x_T, and every DDPM transition-noise draw are paired across
all variants.  Ground truth is used only for reporting PSNR, never for fitting.
"""

import argparse
import os
import sys
import time

import torch


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, TASK_CONFIGS, ZAPS_CONFIG, ZETA_INIT_BY_TASK
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS


TASK = "super_resolution"
MODES = ("both", "zeta_only", "d_only")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = (reference - estimate.clamp(-1.0, 1.0)).square().mean()
    return (10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))).item()


def run_mode(
    mode: str,
    learning_rate: float,
    diffusion_model,
    operator,
    measurement: torch.Tensor,
    ground_truth: torch.Tensor,
    seed: int,
) -> dict:
    cfg = {
        **ZAPS_CONFIG,
        "lr": learning_rate,
        "zeta_init": ZETA_INIT_BY_TASK[TASK],
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }
    set_seed(seed)
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **cfg,
    )
    if mode == "zeta_only":
        zaps.D.requires_grad_(False)
    elif mode == "d_only":
        zaps.zeta.requires_grad_(False)
    elif mode != "both":
        raise ValueError(f"unknown mode: {mode}")

    started = time.time()
    losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
    elapsed = time.time() - started
    reconstruction = zaps._last_opt_x0

    with torch.no_grad():
        result_psnr = psnr(ground_truth, reconstruction)
        residual = (
            measurement - operator.H(reconstruction)
        ).flatten().norm().item()
        zeta = zaps.zeta.detach().float().cpu()
        d_values = zaps.D.detach().float().cpu()
        d_delta = d_values - cfg["d_init"]
        d_delta_rms = d_delta.square().mean().sqrt().item()
        d_delta_max = d_delta.abs().max().item()
        zeta_delta_rms = (
            zeta - cfg["zeta_init"]
        ).square().mean().sqrt().item()

    reverse_zeta = [
        (int(zaps.tau[index].item()), float(zeta[index]))
        for index in range(len(zaps.tau) - 1, -1, -1)
    ]
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "mode": mode,
        "losses": losses,
        "psnr": result_psnr,
        "residual": residual,
        "seconds": elapsed,
        "zeta": reverse_zeta,
        "zeta_min": float(zeta.min()),
        "zeta_max": float(zeta.max()),
        "zeta_delta_rms": zeta_delta_rms,
        "d_delta_rms": d_delta_rms,
        "d_delta_max": d_delta_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate zeta and D learning branches in ZAPS."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(MODES)
    )
    args = parser.parse_args()

    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    with torch.no_grad():
        measurement = operator(ground_truth)
    diffusion_model = load_diffusion_model("imagenet", args.device)

    print("\n=== Paired ZAPS parameter-branch ablation ===")
    print(f"learning rate={args.learning_rate:g}")
    print("All runs share y, x_T, and every DDPM transition-noise draw.")
    results = []
    for mode in args.modes:
        print(f"\n--- {mode} ---")
        results.append(
            run_mode(
                mode,
                args.learning_rate,
                diffusion_model,
                operator,
                measurement,
                ground_truth,
                args.seed,
            )
        )

    print("\n=== Summary ===")
    print(
        f"{'mode':>12} {'PSNR':>10} {'final MSE':>12} {'final ||r||':>14} "
        f"{'zeta dRMS':>12} {'D dRMS':>11} {'D dMax':>11} {'seconds':>10}"
    )
    for result in results:
        print(
            f"{result['mode']:>12} {result['psnr']:10.4f} "
            f"{result['losses'][-1]:12.6f} {result['residual']:14.4f} "
            f"{result['zeta_delta_rms']:12.6f} "
            f"{result['d_delta_rms']:11.6f} "
            f"{result['d_delta_max']:11.6f} {result['seconds']:10.1f}"
        )

    selected_positions = (0, 1, 2, 3, 4, 14, 23, 29)
    print("\nLearned zeta in reverse order (t: zeta):")
    for result in results:
        selected = [result["zeta"][position] for position in selected_positions]
        print(result["mode"])
        print("  " + ", ".join(
            f"{t}: {zeta:.5f}" for t, zeta in selected
        ))

    print("\nInterpretation:")
    print("  zeta_only ~= both: learned zeta supplies nearly all improvement")
    print("  both > zeta_only and d_only: the Hessian branch contributes materially")
    print("  d_only ~= fixed-zeta baseline: D cannot compensate for unadapted zeta")


if __name__ == "__main__":
    main()
