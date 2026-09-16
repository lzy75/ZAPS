"""Paired single-image FFHQ timestep-schedule screening.

The tuned fixed-schedule ZAPS configuration is held constant:

* legacy bicubic transpose and fixed skip variance;
* Adam lr=0.01, zeta_init=0.1, D_init=0.2;
* 30 DDPM steps x 10 epochs (300 NFE).

Only the set of 30 diffusion timesteps changes.  The variants are the paper's
15-10-5 schedule, global uniform spacing, uniform grids in sigma/log-sigma/
log-SNR coordinates, optional global power-law spacing, and multiple
Karras/EDM sigma schedules mapped to the pretrained DDPM's discrete timesteps.
"""

import argparse
import os
import sys
import time

import torch


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, METRICS_CONFIG, TASK_CONFIGS, ZAPS_CONFIG
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS, build_irregular_timesteps
from utils.diag_ffhq_regression import TransposeMode
from utils.diag_learning_rate_ablation import set_seed
from utils.metrics import compute_all_metrics


TASK = "super_resolution"
NUM_STEPS = 30


def rounded_spacing(total_steps: int, count: int, power: float) -> torch.Tensor:
    """Return strictly increasing global linear/power-law timestep indices."""
    u = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    targets = (total_steps - 1) * u.pow(power)
    training_indices = torch.arange(total_steps, dtype=torch.float64)
    # Strong powers can map several low-noise targets to the same integer.
    # Resolve those collisions by choosing the nearest available index while
    # preserving both endpoints and a strictly increasing sequence.
    return nearest_strict_indices(training_indices, targets)


def nearest_strict_indices(
    training_values: torch.Tensor, target_values: torch.Tensor
) -> torch.Tensor:
    """Map sorted targets to sorted values while preserving a unique index each."""
    total = training_values.numel()
    count = target_values.numel()
    selected = []
    previous = -1
    for position, target in enumerate(target_values):
        insertion = int(torch.searchsorted(training_values, target).item())
        candidates = [max(0, min(total - 1, insertion))]
        if insertion > 0:
            candidates.append(insertion - 1)
        nearest = min(
            candidates,
            key=lambda index: abs(float(training_values[index] - target)),
        )
        lower = previous + 1
        upper = total - (count - position)
        nearest = max(lower, min(upper, nearest))
        selected.append(nearest)
        previous = nearest
    selected[0] = 0
    selected[-1] = total - 1
    return torch.tensor(selected, dtype=torch.long)


def karras_timesteps(
    alphas_cumprod: torch.Tensor, count: int, rho: float
) -> torch.Tensor:
    """Build a Karras sigma schedule and map it to discrete DDPM indices."""
    alpha_bar = alphas_cumprod.detach().double().cpu()
    training_sigma = torch.sqrt((1.0 - alpha_bar) / alpha_bar.clamp_min(1e-30))
    sigma_min = training_sigma[0]
    sigma_max = training_sigma[-1]
    ramp = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    targets = (
        sigma_min.pow(1.0 / rho)
        + ramp * (sigma_max.pow(1.0 / rho) - sigma_min.pow(1.0 / rho))
    ).pow(rho)
    return nearest_strict_indices(training_sigma, targets)


def noise_coordinate_timesteps(
    alphas_cumprod: torch.Tensor,
    count: int,
    coordinate: str,
) -> torch.Tensor:
    """Return a grid uniform in a monotone diffusion-noise coordinate."""
    alpha_bar = alphas_cumprod.detach().double().cpu().clamp(1e-30, 1.0 - 1e-15)
    sigma = torch.sqrt((1.0 - alpha_bar) / alpha_bar)
    if coordinate == "sigma":
        training_values = sigma
    elif coordinate == "log_sigma":
        training_values = sigma.clamp_min(1e-30).log()
    elif coordinate == "negative_log_snr":
        # log-SNR decreases with training timestep, so negate it to obtain the
        # ascending coordinate expected by nearest_strict_indices().
        training_values = -(alpha_bar.log() - torch.log1p(-alpha_bar))
    else:
        raise ValueError(f"unknown noise coordinate: {coordinate}")
    targets = torch.linspace(
        float(training_values[0]),
        float(training_values[-1]),
        count,
        dtype=torch.float64,
    )
    return nearest_strict_indices(training_values, targets)


def schedule_variants(
    diffusion_model,
    powers: list[float],
    rho: float | list[float],
    *,
    include_noise_grids: bool = False,
):
    total_steps = int(diffusion_model.alphas_cumprod.numel())
    alpha_bar = diffusion_model.alphas_cumprod
    variants = [
        (
            "paper_15_10_5",
            build_irregular_timesteps(
                total_steps=total_steps,
                schedule=(15, 10, 5),
                spacing="linear",
            ),
        ),
        ("uniform_30", rounded_spacing(total_steps, NUM_STEPS, 1.0)),
    ]
    if include_noise_grids:
        variants.extend(
            [
                (
                    "uniform_sigma",
                    noise_coordinate_timesteps(alpha_bar, NUM_STEPS, "sigma"),
                ),
                (
                    "uniform_logsigma",
                    noise_coordinate_timesteps(alpha_bar, NUM_STEPS, "log_sigma"),
                ),
                (
                    "uniform_logsnr",
                    noise_coordinate_timesteps(
                        alpha_bar, NUM_STEPS, "negative_log_snr"
                    ),
                ),
            ]
        )
    variants.extend(
        (
            f"power_{power:g}",
            rounded_spacing(total_steps, NUM_STEPS, power),
        )
        for power in powers
    )
    rhos = [rho] if isinstance(rho, (int, float)) else list(rho)
    variants.extend(
        (
            f"karras_rho_{value:g}",
            karras_timesteps(alpha_bar, NUM_STEPS, value),
        )
        for value in rhos
    )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed FFHQ ZAPS timestep schedules at 300 NFE."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--powers",
        type=float,
        nargs="*",
        default=[],
        help="Optional global-t power grids (power 2 and 3 were already rejected).",
    )
    parser.add_argument(
        "--karras-rhos",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 7.0],
        help="Karras rho values to screen (default: 3 5 7).",
    )
    parser.add_argument(
        "--karras-rho",
        type=float,
        default=None,
        help="Deprecated single-rho override retained for old commands.",
    )
    args = parser.parse_args()

    if any(power <= 0 for power in args.powers):
        raise ValueError("power values must be positive")
    karras_rhos = [args.karras_rho] if args.karras_rho is not None else args.karras_rhos
    if any(rho <= 0 for rho in karras_rhos):
        raise ValueError("Karras rho values must be positive")

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    base_operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    operator = TransposeMode(base_operator, "legacy_bicubic").to(args.device)
    with torch.no_grad():
        measurement = operator(ground_truth)

    diffusion_model = load_diffusion_model("ffhq", args.device)
    variants = schedule_variants(
        diffusion_model,
        args.powers,
        karras_rhos,
        include_noise_grids=True,
    )
    config = {
        **ZAPS_CONFIG,
        "lr": 0.01,
        "zeta_init": 0.1,
        "d_init": 0.2,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }

    print("\n=== Paired FFHQ timestep-schedule screening ===", flush=True)
    print(
        "lr=0.01; zeta_init=0.1; D_init=0.2; legacy bicubic transpose; "
        "fixed variance; 30 steps x 10 epochs",
        flush=True,
    )
    print("All variants share y, x_T, and DDPM draws by reverse-step position.", flush=True)

    results = []
    for name, timesteps in variants:
        if timesteps.numel() != NUM_STEPS or timesteps.unique().numel() != NUM_STEPS:
            raise RuntimeError(f"{name} does not contain {NUM_STEPS} unique timesteps")
        set_seed(args.seed)
        zaps = ZAPS(
            diffusion_model=diffusion_model,
            forward_operator=operator,
            img_size=IMG_SIZE[0],
            **config,
        )
        zaps.tau = timesteps.to(args.device)
        print(f"\n--- {name} ---", flush=True)
        print(f"timesteps (ascending): {timesteps.tolist()}", flush=True)
        started = time.time()
        losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
        elapsed = time.time() - started
        reconstruction = zaps._last_opt_x0
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
            results.append(
                {
                    "name": name,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                    "mse": losses[-1],
                    "residual": residual,
                    "zeta_min": zaps.zeta.detach().min().item(),
                    "zeta_max": zaps.zeta.detach().max().item(),
                    "d_delta_rms": d_delta.square().mean().sqrt().item(),
                    "seconds": elapsed,
                }
            )
        del zaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Summary ===", flush=True)
    print(
        f"{'schedule':>20} {'PSNR':>10} {'SSIM':>9} {'LPIPS':>9} {'final MSE':>12} "
        f"{'final ||r||':>14} {'zeta min':>11} {'zeta max':>11} "
        f"{'D dRMS':>10} {'seconds':>10}",
        flush=True,
    )
    for result in results:
        print(
            f"{result['name']:>20} {result['psnr']:10.4f} "
            f"{result['ssim']:9.4f} {result['lpips']:9.4f} "
            f"{result['mse']:12.6f} {result['residual']:14.4f} "
            f"{result['zeta_min']:11.5f} {result['zeta_max']:11.5f} "
            f"{result['d_delta_rms']:10.6f} {result['seconds']:10.1f}",
            flush=True,
        )
    print(
        "\nSelect by the PSNR/SSIM/LPIPS trade-off; lower measurement MSE alone "
        "is not sufficient.",
        flush=True,
    )


if __name__ == "__main__":
    main()
