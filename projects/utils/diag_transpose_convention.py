"""Paired single-image ablation of the super-resolution transpose convention.

This diagnostic changes exactly one component of the current paper-setting
ImageNet ZAPS baseline: the map used as ``A.transpose`` in the likelihood
correction.  The forward operator, noisy observation, x_T, DDPM draws,
optimizer, zeta/D initialization, and diffusion model are shared.

Variants:

* exact: the mathematical adjoint of the DPS Resizer downsampler.
* exact_x16: the same direction multiplied by scale_factor**2.  This isolates
  normalization from direction.
* dps_nearest: ``F.interpolate`` with its default nearest mode, exactly as in
  DPS/guided_diffusion/measurements.py.  This tests reference-code convention,
  not mathematical correctness.
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
VARIANTS = ("exact", "exact_x16", "dps_nearest")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = (reference - estimate.clamp(-1.0, 1.0)).square().mean()
    return (10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))).item()


class TransposeConvention(nn.Module):
    """Keep H fixed while selecting the map used for A.transpose."""

    def __init__(self, base: nn.Module, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown transpose variant: {variant}")
        self.base = base
        self.variant = variant
        self.scale_factor = base.scale_factor

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.H(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.H(x)

    def transpose(self, y: torch.Tensor, output_size=None) -> torch.Tensor:
        if self.variant == "dps_nearest":
            if output_size is None:
                return F.interpolate(y, scale_factor=self.scale_factor)
            return F.interpolate(y, size=output_size)

        exact = self.base.transpose(y, output_size=output_size)
        if self.variant == "exact_x16":
            return exact * float(self.scale_factor ** 2)
        return exact


def transpose_diagnostics(
    operator: TransposeConvention,
    image_shape: tuple[int, ...],
    device: str,
    seed: int,
) -> tuple[float, float]:
    """Return ||T(y)||/||H^T(y)|| and normalized inner-product error."""
    set_seed(seed + 9173)
    x = torch.randn(image_shape, device=device)
    y = torch.randn_like(operator.H(x))
    exact = operator.base.transpose(y, output_size=x.shape[-2:])
    candidate = operator.transpose(y, output_size=x.shape[-2:])
    norm_ratio = (
        candidate.flatten().norm() / exact.flatten().norm().clamp_min(1e-12)
    ).item()
    lhs = (operator.H(x) * y).sum()
    rhs = (x * candidate).sum()
    ip_error = (
        (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12)
    ).item()
    return norm_ratio, ip_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact, scaled-exact, and DPS nearest SR transpose conventions."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    config = {
        **ZAPS_CONFIG,
        "lr": 0.001,
        "zeta_init": ZETA_INIT_BY_TASK[TASK],
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }

    ground_truth = load_image_as_tensor(args.image).to(args.device)
    base = get_operator(
        TASK,
        device=args.device,
        **{**TASK_CONFIGS[TASK], "noise_sigma": 0.0},
    )
    set_seed(args.seed)
    with torch.no_grad():
        measurement = base.H(ground_truth) + 0.05 * torch.randn_like(base.H(ground_truth))

    diffusion_model = load_diffusion_model("imagenet", args.device)
    results = []

    print("\n=== Paired SR transpose-convention ablation ===", flush=True)
    print(
        "one image; lr=0.001; joint zeta+D; DDPM; fixed variance; "
        "model-domain sigma=0.05",
        flush=True,
    )
    print(
        "All variants share H, y, x_T, and every DDPM transition-noise draw.",
        flush=True,
    )

    for variant in VARIANTS:
        operator = TransposeConvention(base, variant).to(args.device)
        norm_ratio, ip_error = transpose_diagnostics(
            operator, tuple(ground_truth.shape), args.device, args.seed
        )
        set_seed(args.seed)
        zaps = ZAPS(
            diffusion_model=diffusion_model,
            forward_operator=operator,
            img_size=IMG_SIZE[0],
            **config,
        )
        print(
            f"\n--- {variant} (||T y||/||H^T y||={norm_ratio:.4f}, "
            f"adjoint error={ip_error:.6f}) ---",
            flush=True,
        )
        started = time.time()
        losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
        elapsed = time.time() - started
        reconstruction = zaps._last_opt_x0
        with torch.no_grad():
            residual = (measurement - operator.H(reconstruction)).flatten().norm().item()
            d_delta = zaps.D.detach() - config["d_init"]
            result = {
                "variant": variant,
                "psnr": psnr(ground_truth, reconstruction),
                "mse": losses[-1],
                "residual": residual,
                "zeta_min": zaps.zeta.detach().min().item(),
                "zeta_max": zaps.zeta.detach().max().item(),
                "d_delta_rms": d_delta.square().mean().sqrt().item(),
                "norm_ratio": norm_ratio,
                "ip_error": ip_error,
                "seconds": elapsed,
            }
        results.append(result)
        del zaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Summary ===", flush=True)
    print(
        f"{'variant':>13} {'PSNR':>9} {'final MSE':>12} {'final ||r||':>13} "
        f"{'zeta min':>10} {'zeta max':>10} {'D dRMS':>10} "
        f"{'T/H^T':>9} {'adj.err':>9} {'seconds':>9}",
        flush=True,
    )
    for result in results:
        print(
            f"{result['variant']:>13} {result['psnr']:9.4f} "
            f"{result['mse']:12.6f} {result['residual']:13.4f} "
            f"{result['zeta_min']:10.5f} {result['zeta_max']:10.5f} "
            f"{result['d_delta_rms']:10.6f} {result['norm_ratio']:9.4f} "
            f"{result['ip_error']:9.6f} {result['seconds']:9.1f}",
            flush=True,
        )

    exact_psnr = results[0]["psnr"]
    print("\nPSNR delta versus exact:", flush=True)
    for result in results[1:]:
        print(
            f"  {result['variant']}: {result['psnr'] - exact_psnr:+.4f} dB",
            flush=True,
        )
    print("\nDecision rule:", flush=True)
    print(
        "  exact_x16 improves: transpose normalization is the missing convention.",
        flush=True,
    )
    print(
        "  dps_nearest improves beyond exact_x16: direction plus DPS convention matters.",
        flush=True,
    )
    print(
        "  neither improves: reject transpose convention and audit the Eq.21 unroll next.",
        flush=True,
    )


if __name__ == "__main__":
    main()
