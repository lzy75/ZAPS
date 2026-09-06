"""Test whether Eq.21's Hessian approximation is missing from unroll backward.

Both variants have identical forward equations, observations, x_T, and DDPM
noise draws.  The only difference is the gradient used through later Tweedie
steps while optimizing zeta/D:

* identity_only: current stop-gradient behavior, dx0_hat/dx = I/sqrt(alpha).
* eq21_surrogate: dx0_hat/dx is replaced by the paper's
  (I + (1-alpha) W D W^T)/sqrt(alpha) approximation.

The surrogate is forward-zero and therefore cannot directly change a sample;
it only changes the zero-shot optimization gradients.
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


def run_variant(
    name: str,
    use_surrogate: bool,
    diffusion_model,
    operator,
    measurement: torch.Tensor,
    ground_truth: torch.Tensor,
    seed: int,
) -> dict:
    cfg = {
        **ZAPS_CONFIG,
        "zeta_init": ZETA_INIT_BY_TASK[TASK],
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": use_surrogate,
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
        d_per_step = d_values.abs().flatten(1).mean(1)
        d_delta_rms = d_delta.square().mean().sqrt().item()
        d_delta_max = d_delta.abs().max().item()

    reverse_parameters = [
        (
            int(zaps.tau[index].item()),
            float(zeta[index]),
            float(d_per_step[index]),
        )
        for index in range(len(zaps.tau) - 1, -1, -1)
    ]
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "losses": losses,
        "psnr": result_psnr,
        "residual": residual,
        "seconds": elapsed,
        "parameters": reverse_parameters,
        "d_delta_rms": d_delta_rms,
        "d_delta_max": d_delta_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate the Eq.21 score-Jacobian surrogate in unroll backward."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    with torch.no_grad():
        measurement = operator(ground_truth)
    diffusion_model = load_diffusion_model("imagenet", args.device)

    results = []
    print("\n=== Unroll score-Jacobian backward ablation ===")
    print("Forward values and random draws are paired; only backward differs.")
    for name, flag in (("identity_only", False), ("eq21_surrogate", True)):
        print(f"\n--- {name} ---")
        results.append(
            run_variant(
                name,
                flag,
                diffusion_model,
                operator,
                measurement,
                ground_truth,
                args.seed,
            )
        )

    print("\n=== Summary ===")
    print(
        f"{'variant':>17} {'PSNR':>10} {'final MSE':>12} {'final ||r||':>14} "
        f"{'D delta RMS':>13} {'D delta max':>13} {'seconds':>10}"
    )
    for result in results:
        print(
            f"{result['name']:>17} {result['psnr']:10.4f} "
            f"{result['losses'][-1]:12.6f} {result['residual']:14.4f} "
            f"{result['d_delta_rms']:13.6f} {result['d_delta_max']:13.6f} "
            f"{result['seconds']:10.1f}"
        )

    selected_positions = (0, 1, 2, 3, 4, 14, 23, 29)
    print("\nLearned parameters in reverse order (t: zeta / mean|D|):")
    for result in results:
        selected = [result["parameters"][position] for position in selected_positions]
        print(result["name"])
        print("  " + ", ".join(
            f"{t}: {zeta:.5f}/{d_value:.5f}"
            for t, zeta, d_value in selected
        ))

    print("\nInterpretation:")
    print("  surrogate improves clearly: the missing Eq.21 unroll-backward path is a core bug")
    print("  variants remain alike: identity stop-gradient is adequate and this hypothesis is rejected")


if __name__ == "__main__":
    main()
