"""Ablate DDPM transition-noise reuse during ZAPS zero-shot optimization.

The current optimizer fixes x_T but draws fresh per-step DDPM noise in every
epoch.  This script compares that behavior with a paired variant that captures
the RNG state at the first reverse trajectory and restores it before every
later epoch.  Consequently, both variants have exactly the same measurement,
initial x_T, and first-epoch transition noises; only epochs 2-10 differ.

No core implementation or model weights are changed.
"""

import argparse
import os
import sys
import time
import types

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


def install_fixed_transition_rng(zaps: ZAPS) -> None:
    """Replay the first epoch's transition-noise RNG state in later epochs."""
    original_reverse = zaps._reverse_diffusion
    captured = {}

    def fixed_reverse(self, *args, **kwargs):
        if not captured:
            captured["cpu"] = torch.get_rng_state()
            if torch.cuda.is_available():
                captured["cuda"] = torch.cuda.get_rng_state_all()
        else:
            torch.set_rng_state(captured["cpu"])
            if "cuda" in captured:
                torch.cuda.set_rng_state_all(captured["cuda"])
        return original_reverse(*args, **kwargs)

    zaps._reverse_diffusion = types.MethodType(fixed_reverse, zaps)


def run_variant(
    name: str,
    reuse_transition_noise: bool,
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
    }
    set_seed(seed)
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **cfg,
    )
    if reuse_transition_noise:
        install_fixed_transition_rng(zaps)

    started = time.time()
    losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
    elapsed = time.time() - started
    reconstruction = zaps._last_opt_x0

    with torch.no_grad():
        final_residual = (
            measurement - operator.H(reconstruction)
        ).flatten().norm().item()
        final_psnr = psnr(ground_truth, reconstruction)
        zeta = zaps.zeta.detach().float().cpu()
        d_per_step = zaps.D.detach().float().cpu().abs().flatten(1).mean(1)

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
        "psnr": final_psnr,
        "residual": final_residual,
        "seconds": elapsed,
        "parameters": reverse_parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fresh versus fixed DDPM transition noise across epochs."
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

    print("\n=== Transition-noise reuse ablation ===")
    print("Both variants share y, x_T, and all first-epoch transition noises.")
    results = []
    for name, reuse in (("fresh_each_epoch", False), ("fixed_across_epochs", True)):
        print(f"\n--- {name} ---")
        results.append(
            run_variant(
                name,
                reuse,
                diffusion_model,
                operator,
                measurement,
                ground_truth,
                args.seed,
            )
        )

    print("\n=== Summary ===")
    print(f"{'variant':>20} {'PSNR':>10} {'final MSE':>12} {'final ||r||':>14} {'seconds':>10}")
    for result in results:
        print(
            f"{result['name']:>20} {result['psnr']:10.4f} "
            f"{result['losses'][-1]:12.6f} {result['residual']:14.4f} "
            f"{result['seconds']:10.1f}"
        )

    selected_positions = (0, 1, 2, 3, 4, 14, 23, 29)
    print("\nLearned parameters in reverse order (t: zeta / mean|D|):")
    for result in results:
        print(result["name"])
        selected = [result["parameters"][position] for position in selected_positions]
        print("  " + ", ".join(
            f"{t}: {zeta:.5f}/{d_value:.5f}"
            for t, zeta, d_value in selected
        ))

    print("\nInterpretation:")
    print("  fixed noise lowers loss and raises PSNR: stochastic transition gradients are blocking adaptation")
    print("  both remain nearly identical: transition-noise redraw is not the cause")


if __name__ == "__main__":
    main()
