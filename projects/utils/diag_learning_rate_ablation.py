"""Paired learning-rate ablation for ZAPS zero-shot adaptation.

The paper uses only ten optimizer updates for the 30-step x 10-epoch setup.
With Adam lr=1e-3, each per-timestep zeta in the current implementation moves
by at most roughly 0.01 from its 0.1 initialization.  This diagnostic tests
whether that narrow adaptation range is preventing the distinct zeta values
from converging.

Every learning-rate run shares the same observation, x_T, and sequence of
per-epoch DDPM transition noises.  Only Adam's learning rate changes.
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = (reference - estimate.clamp(-1.0, 1.0)).square().mean()
    return (10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))).item()


def run_learning_rate(
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

    reverse_zeta = [
        (int(zaps.tau[index].item()), float(zeta[index]))
        for index in range(len(zaps.tau) - 1, -1, -1)
    ]
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "lr": learning_rate,
        "losses": losses,
        "psnr": result_psnr,
        "residual": residual,
        "seconds": elapsed,
        "zeta": reverse_zeta,
        "zeta_min": float(zeta.min()),
        "zeta_max": float(zeta.max()),
        "d_delta_rms": d_delta_rms,
        "d_delta_max": d_delta_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ZAPS zero-shot optimization learning rates."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-3, 1e-2],
    )
    args = parser.parse_args()

    if any(rate <= 0 for rate in args.learning_rates):
        raise ValueError("all learning rates must be positive")

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    with torch.no_grad():
        measurement = operator(ground_truth)
    diffusion_model = load_diffusion_model("imagenet", args.device)

    print("\n=== Paired ZAPS learning-rate ablation ===")
    print("All runs share y, x_T, and every DDPM transition-noise draw.")
    results = []
    for learning_rate in args.learning_rates:
        print(f"\n--- lr={learning_rate:g} ---")
        results.append(
            run_learning_rate(
                learning_rate,
                diffusion_model,
                operator,
                measurement,
                ground_truth,
                args.seed,
            )
        )

    print("\n=== Summary ===")
    print(
        f"{'lr':>10} {'PSNR':>10} {'final MSE':>12} {'final ||r||':>14} "
        f"{'zeta min':>11} {'zeta max':>11} {'D delta RMS':>13} "
        f"{'D delta max':>13} {'seconds':>10}"
    )
    for result in results:
        print(
            f"{result['lr']:10.4g} {result['psnr']:10.4f} "
            f"{result['losses'][-1]:12.6f} {result['residual']:14.4f} "
            f"{result['zeta_min']:11.5f} {result['zeta_max']:11.5f} "
            f"{result['d_delta_rms']:13.6f} {result['d_delta_max']:13.6f} "
            f"{result['seconds']:10.1f}"
        )

    selected_positions = (0, 1, 2, 3, 4, 14, 23, 29)
    print("\nLearned zeta in reverse order (t: zeta):")
    for result in results:
        selected = [result["zeta"][position] for position in selected_positions]
        print(f"lr={result['lr']:g}")
        print("  " + ", ".join(
            f"{t}: {zeta:.5f}" for t, zeta in selected
        ))

    print("\nInterpretation:")
    print("  a higher lr improves PSNR/loss strongly: adaptation range is the bottleneck")
    print("  higher lr only lowers measurement loss or hurts PSNR: optimizer speed is not the root cause")


if __name__ == "__main__":
    main()
