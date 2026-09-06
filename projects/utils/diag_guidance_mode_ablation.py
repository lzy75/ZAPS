"""Fair 30-step ablation of three likelihood-guidance implementations.

All modes share the same ImageNet model, noisy SR observation, initial x_T,
timestep grid, and DDPM transition noises:

* ``zaps_raw``: the current ZAPS equation, zeta * J_approx^T A^T r.
* ``zaps_l2``: the same approximate direction normalized by ||r||_2 and
  scaled like the official DPS super-resolution configuration.
* ``dps_exact``: the exact DPS direction obtained by differentiating the L2
  residual through the denoiser.

This script does not optimize or alter model weights.  Model parameters retain
``requires_grad=True`` because guided-diffusion's custom checkpoint backward
expects differentiable parameter inputs.
"""

import argparse
import os
import sys
import time

import torch


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, RESULTS_DIR, TASK_CONFIGS, ZAPS_CONFIG
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS, ddpm_posterior_step


TASK = "super_resolution"
MODES = ("zaps_raw", "zaps_l2", "dps_exact")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = (reference - estimate.clamp(-1.0, 1.0)).square().mean()
    return (10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))).item()


def save_image(image: torch.Tensor, path: str) -> None:
    from PIL import Image

    array = (
        ((image.detach().float().cpu()[0].clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array).save(path)


def approximate_direction(
    zaps: ZAPS,
    residual: torch.Tensor,
    timestep_index: int,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    adjoint_residual = zaps.A.transpose(residual)
    hessian_term = zaps.dwt.synthesis(
        zaps.D[timestep_index] * zaps.dwt.analysis(adjoint_residual)
    )
    return (
        adjoint_residual + (1.0 - alpha_bar) * hessian_term
    ) / alpha_bar.sqrt().clamp_min(1e-8)


def run_mode(
    mode: str,
    diffusion_model,
    zaps: ZAPS,
    measurement: torch.Tensor,
    ground_truth: torch.Tensor,
    seed: int,
    zeta: float,
    dps_scale: float,
) -> dict:
    set_seed(seed + 7)
    x_t = torch.randn(
        1, 3, IMG_SIZE[0], IMG_SIZE[1], device=ground_truth.device
    )
    timesteps = zaps.tau
    alphas_cumprod = diffusion_model.alphas_cumprod
    residual_trace = []
    started = time.time()

    for timestep_index in range(len(timesteps) - 1, -1, -1):
        t_current = int(timesteps[timestep_index].item())
        t_previous = (
            int(timesteps[timestep_index - 1].item())
            if timestep_index > 0
            else -1
        )
        t_batch = torch.tensor(
            [t_current], device=ground_truth.device, dtype=torch.long
        )
        alpha_bar = alphas_cumprod[t_current]

        if mode == "dps_exact":
            x_probe = x_t.detach().requires_grad_(True)
            epsilon = diffusion_model._predict_eps(x_probe, t_batch)
            x0_graph = (
                x_probe - (1.0 - alpha_bar).sqrt() * epsilon
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            x0_graph = x0_graph.clamp(-1.0, 1.0)
            residual_graph = measurement - zaps.A.H(x0_graph)
            residual_l2 = residual_graph.flatten().norm()
            correction = -dps_scale * torch.autograd.grad(
                residual_l2,
                x_probe,
                create_graph=False,
                retain_graph=False,
            )[0].detach()
            x0_pred = x0_graph.detach()
            residual_value = residual_l2.detach().item()
            del x_probe, epsilon, x0_graph, residual_graph, residual_l2
        else:
            with torch.no_grad():
                epsilon = diffusion_model._predict_eps(x_t, t_batch)
                x0_pred = (
                    x_t - (1.0 - alpha_bar).sqrt() * epsilon
                ) / alpha_bar.sqrt().clamp_min(1e-8)
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                residual = measurement - zaps.A.H(x0_pred)
                residual_l2 = residual.flatten().norm()
                direction = approximate_direction(
                    zaps, residual, timestep_index, alpha_bar
                )
                if mode == "zaps_raw":
                    correction = zeta * direction
                elif mode == "zaps_l2":
                    correction = (
                        dps_scale
                        * direction
                        / residual_l2.clamp_min(1e-8)
                    )
                else:
                    raise ValueError(f"unknown mode: {mode}")
                residual_value = residual_l2.item()

        residual_trace.append(residual_value)
        with torch.no_grad():
            unconditional = ddpm_posterior_step(
                x_t,
                x0_pred,
                t_current,
                t_previous,
                alphas_cumprod,
                eta=1.0,
                mode="ddpm",
            )
            x_t = unconditional + correction

    with torch.no_grad():
        final = x_t.clamp(-1.0, 1.0)
        final_residual = (measurement - zaps.A.H(final)).flatten().norm().item()
        result_psnr = psnr(ground_truth, final)

    return {
        "image": final,
        "psnr": result_psnr,
        "residual": final_residual,
        "trace": residual_trace,
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw ZAPS, normalized ZAPS, and exact DPS at 30 steps."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--zeta", type=float, default=0.1)
    parser.add_argument("--d-init", type=float, default=0.2)
    parser.add_argument("--dps-scale", type=float, default=0.3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(RESULTS_DIR, "diag_guidance_modes"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    with torch.no_grad():
        measurement = operator(ground_truth)

    diffusion_model = load_diffusion_model("imagenet", args.device)
    diffusion_model.model.eval()
    if "dps_exact" in args.modes and not all(
        parameter.requires_grad for parameter in diffusion_model.model.parameters()
    ):
        raise RuntimeError(
            "dps_exact requires model parameter requires_grad flags to remain enabled "
            "for guided-diffusion checkpoint backward"
        )

    cfg = {
        **ZAPS_CONFIG,
        "zeta_init": args.zeta,
        "d_init": args.d_init,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
    }
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **cfg,
    )
    with torch.no_grad():
        zaps.zeta.fill_(args.zeta)
        zaps.D.fill_(args.d_init)

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n=== Fair 30-step guidance-mode ablation ===")
    print(f"timesteps (ascending): {[int(t) for t in zaps.tau.tolist()]}")
    print(f"zeta={args.zeta}, D={args.d_init}, DPS scale={args.dps_scale}")
    print(f"{'mode':>12} {'PSNR':>10} {'final ||r||':>14} {'seconds':>10}")

    results = {}
    for mode in args.modes:
        result = run_mode(
            mode,
            diffusion_model,
            zaps,
            measurement,
            ground_truth,
            args.seed,
            args.zeta,
            args.dps_scale,
        )
        results[mode] = result
        save_image(result["image"], os.path.join(args.output_dir, f"{mode}.png"))
        print(
            f"{mode:>12} {result['psnr']:10.4f} "
            f"{result['residual']:14.4f} {result['seconds']:10.1f}"
        )

    print("\nSelected pre-correction residuals:")
    print(f"{'mode':>12} {'t=999':>10} {'t=667':>10} {'t=334':>10} {'t=143':>10} {'t=0':>10}")
    positions = (0, 4, 14, 23, 29)
    for mode, result in results.items():
        values = [result["trace"][position] for position in positions]
        print(f"{mode:>12}" + "".join(f"{value:10.2f}" for value in values))

    print(f"\nImages saved to: {args.output_dir}")
    print("Interpretation:")
    print("  zaps_l2 improves strongly: raw-residual guidance scaling is the primary issue")
    print("  dps_exact improves but zaps_l2 does not: approximate Jacobian direction is the issue")
    print("  dps_exact is also poor: the remaining gap is mainly the 30-step sampler/prior path")


if __name__ == "__main__":
    main()
