"""Locate the shared-code regression in the current FFHQ SR baseline.

The historically successful FFHQ baseline used a bicubic-upsample map for
``A.transpose`` and fixed skip-posterior variance.  The current baseline uses
the exact adjoint of the DPS Resizer and the diffusion model's learned-range
variance.  This paired 2x2 ablation changes only those two choices while
sharing H, y, x_T, timestep schedule, optimizer, and DDPM noise draws.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def tensor_psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = (reference - estimate.clamp(-1.0, 1.0)).square().mean()
    return (10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))).item()


class TransposeMode(nn.Module):
    """Keep the current forward Resizer H and switch only A.transpose."""

    def __init__(self, base: nn.Module, mode: str):
        super().__init__()
        if mode not in ("exact", "legacy_bicubic"):
            raise ValueError(f"unknown transpose mode: {mode}")
        self.base = base
        self.mode = mode
        self.scale_factor = base.scale_factor

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.H(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.H(x)

    def transpose(self, y: torch.Tensor, output_size=None) -> torch.Tensor:
        if self.mode == "exact":
            return self.base.transpose(y, output_size=output_size)
        if output_size is None:
            output_size = (
                y.shape[-2] * self.scale_factor,
                y.shape[-1] * self.scale_factor,
            )
        return F.interpolate(
            y, size=output_size, mode="bicubic", align_corners=False
        )


def run_variant(
    name: str,
    transpose_mode: str,
    use_learned_var: bool,
    diffusion_model,
    base_operator,
    measurement: torch.Tensor,
    ground_truth: torch.Tensor,
    device: str,
    seed: int,
) -> dict:
    operator = TransposeMode(base_operator, transpose_mode).to(device)
    config = {
        **ZAPS_CONFIG,
        "lr": 0.001,
        "zeta_init": ZETA_INIT_BY_TASK[TASK],
        "use_learned_var": use_learned_var,
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
    print(
        f"\n--- {name}: transpose={transpose_mode}, "
        f"learned_var={use_learned_var} ---",
        flush=True,
    )
    started = time.time()
    losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
    elapsed = time.time() - started
    reconstruction = zaps._last_opt_x0

    with torch.no_grad():
        residual = (
            measurement - operator.H(reconstruction)
        ).flatten().norm().item()
        d_delta = zaps.D.detach() - config["d_init"]
        result = {
            "name": name,
            "transpose": transpose_mode,
            "learned_var": use_learned_var,
            "psnr": tensor_psnr(ground_truth, reconstruction),
            "mse": losses[-1],
            "residual": residual,
            "zeta_min": zaps.zeta.detach().min().item(),
            "zeta_max": zaps.zeta.detach().max().item(),
            "d_delta_rms": d_delta.square().mean().sqrt().item(),
            "seconds": elapsed,
        }
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired FFHQ regression ablation: transpose x variance."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    ground_truth = load_image_as_tensor(args.image).to(args.device)
    base_operator = get_operator(
        TASK,
        device=args.device,
        **{**TASK_CONFIGS[TASK], "noise_sigma": 0.0},
    )

    set_seed(args.seed)
    with torch.no_grad():
        clean_measurement = base_operator.H(ground_truth)
        measurement = clean_measurement + 0.05 * torch.randn_like(clean_measurement)
    observation = F.interpolate(
        measurement,
        size=ground_truth.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    observed_psnr = tensor_psnr(ground_truth, observation)

    diffusion_model = load_diffusion_model("ffhq", args.device)
    variants = (
        ("current", "exact", True),
        ("variance_revert", "exact", False),
        ("transpose_revert", "legacy_bicubic", True),
        ("historical_pair", "legacy_bicubic", False),
    )

    print("\n=== FFHQ shared-code regression: paired 2x2 ablation ===", flush=True)
    print(
        "All variants share current H, y, x_T, 15-10-5 timesteps, and every "
        "DDPM transition-noise draw.",
        flush=True,
    )
    print(f"Observed bicubic PSNR: {observed_psnr:.4f} dB", flush=True)

    results = [
        run_variant(
            name,
            transpose_mode,
            use_learned_var,
            diffusion_model,
            base_operator,
            measurement,
            ground_truth,
            args.device,
            args.seed,
        )
        for name, transpose_mode, use_learned_var in variants
    ]

    print("\n=== Summary ===", flush=True)
    print(
        f"{'variant':>18} {'transpose':>16} {'learned':>8} {'PSNR':>9} "
        f"{'delta_obs':>10} {'final MSE':>12} {'final ||r||':>13} "
        f"{'zeta range':>21} {'D dRMS':>10} {'seconds':>9}",
        flush=True,
    )
    for result in results:
        zeta_range = f"{result['zeta_min']:.4f}..{result['zeta_max']:.4f}"
        print(
            f"{result['name']:>18} {result['transpose']:>16} "
            f"{str(result['learned_var']):>8} {result['psnr']:9.4f} "
            f"{result['psnr'] - observed_psnr:+10.4f} "
            f"{result['mse']:12.6f} {result['residual']:13.4f} "
            f"{zeta_range:>21} {result['d_delta_rms']:10.6f} "
            f"{result['seconds']:9.1f}",
            flush=True,
        )

    lookup = {result["name"]: result for result in results}
    variance_effect_exact = (
        lookup["current"]["psnr"] - lookup["variance_revert"]["psnr"]
    )
    transpose_effect_fixed = (
        lookup["historical_pair"]["psnr"] - lookup["variance_revert"]["psnr"]
    )
    interaction = (
        lookup["transpose_revert"]["psnr"]
        - lookup["current"]["psnr"]
        - transpose_effect_fixed
    )
    print("\nPaired effects (positive means the first named choice is better):", flush=True)
    print(
        f"  learned variance effect with exact H^T: {variance_effect_exact:+.4f} dB",
        flush=True,
    )
    print(
        f"  legacy transpose effect with fixed variance: {transpose_effect_fixed:+.4f} dB",
        flush=True,
    )
    print(f"  transpose x variance interaction: {interaction:+.4f} dB", flush=True)
    print("\nDecision rule:", flush=True)
    print(
        "  historical_pair recovers ~29 dB: the regression is fully explained "
        "by these shared-code changes.",
        flush=True,
    )
    print(
        "  only transpose_revert/historical_pair recover: the exact-adjoint "
        "normalization is incompatible with the historical zeta scale.",
        flush=True,
    )
    print(
        "  only fixed-variance variants recover: learned-range variance on "
        "large skipped steps is the regression.",
        flush=True,
    )
    print(
        "  none recover: next isolate the old forward H and old timestep grid; "
        "do not tune learning rates yet.",
        flush=True,
    )


if __name__ == "__main__":
    main()
