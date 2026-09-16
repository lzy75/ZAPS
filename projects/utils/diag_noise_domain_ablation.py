"""Paired test of the ImageNet measurement-noise value-domain convention.

The diffusion trajectory lives in [-1, 1].  If the paper's sigma=0.05 noise
is defined in image space [0, 1] and the observation is then converted to
[-1, 1], its equivalent standard deviation is 0.10.  This diagnostic compares
that convention with the repository's current sigma=0.05 convention while
holding the clean observation, normalized noise direction, x_T, DDPM draws,
optimizer, and timesteps fixed.

Only zeta is optimized because the paired parameter-branch ablation showed
that learning D does not improve PSNR on this ImageNet example.
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
    label: str,
    sigma: float,
    clean_measurement: torch.Tensor,
    standard_noise: torch.Tensor,
    ground_truth: torch.Tensor,
    diffusion_model,
    operator,
    learning_rate: float,
    seed: int,
) -> dict:
    measurement = clean_measurement + sigma * standard_noise
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
    zaps.D.requires_grad_(False)

    started = time.time()
    losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
    elapsed = time.time() - started
    reconstruction = zaps._last_opt_x0

    with torch.no_grad():
        residual = (measurement - operator.H(reconstruction)).flatten().norm().item()
        result = {
            "label": label,
            "sigma": sigma,
            "psnr": psnr(ground_truth, reconstruction),
            "loss": losses[-1],
            "residual": residual,
            "zeta_min": zaps.zeta.detach().min().item(),
            "zeta_max": zaps.zeta.detach().max().item(),
            "seconds": elapsed,
        }

    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sigma=0.05 in [-1,1] with the [0,1]-equivalent sigma=0.10."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    operator_kwargs = {**TASK_CONFIGS[TASK], "noise_sigma": 0.0}
    operator = get_operator(TASK, device=args.device, **operator_kwargs)
    with torch.no_grad():
        clean_measurement = operator.H(ground_truth)

    # Match the earlier branch ablation's observation-noise realization.  The
    # same standard-normal tensor is then scaled in both variants.
    set_seed(args.seed)
    standard_noise = torch.randn_like(clean_measurement)
    diffusion_model = load_diffusion_model("imagenet", args.device)

    print("\n=== Paired measurement-noise domain ablation ===", flush=True)
    print(f"learning rate={args.learning_rate:g}; D frozen; zeta learned", flush=True)
    print("Both runs share H(x), normalized observation noise, x_T, and DDPM draws.", flush=True)

    variants = (
        ("current_[-1,1]_sigma0.05", 0.05),
        ("paper_[0,1]_equiv_sigma0.10", 0.10),
    )
    results = []
    for label, sigma in variants:
        print(f"\n--- {label} ---", flush=True)
        results.append(
            run_variant(
                label,
                sigma,
                clean_measurement,
                standard_noise,
                ground_truth,
                diffusion_model,
                operator,
                args.learning_rate,
                args.seed,
            )
        )

    print("\n=== Summary ===", flush=True)
    print(
        f"{'variant':>32} {'sigma':>8} {'PSNR':>10} {'final MSE':>12} "
        f"{'final ||r||':>14} {'zeta min':>11} {'zeta max':>11} {'seconds':>10}",
        flush=True,
    )
    for result in results:
        print(
            f"{result['label']:>32} {result['sigma']:8.3f} "
            f"{result['psnr']:10.4f} {result['loss']:12.6f} "
            f"{result['residual']:14.4f} {result['zeta_min']:11.5f} "
            f"{result['zeta_max']:11.5f} {result['seconds']:10.1f}",
            flush=True,
        )

    delta = results[1]["psnr"] - results[0]["psnr"]
    print(f"\nPSNR delta (sigma0.10 - sigma0.05): {delta:+.4f} dB", flush=True)
    print("Interpretation:", flush=True)
    print("  clear positive delta: adopt the [0,1]-calibrated noise convention", flush=True)
    print("  near-zero or negative delta: value-domain noise scaling is not the missing cause", flush=True)


if __name__ == "__main__":
    main()
