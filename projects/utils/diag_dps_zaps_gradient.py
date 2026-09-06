"""Compare the exact DPS likelihood direction with the ZAPS approximation.

This is a diagnostic, not a reconstruction baseline.  It replays one 30-step
ZAPS trajectory and, at a few representative timesteps, computes

    DPS direction  = -d ||y - A(x0_hat(x_t))||_2 / d x_t

by back-propagating through the denoiser.  It then compares that direction to
the Hessian/Jacobian approximation used by ZAPS.

Important: guided-diffusion's custom gradient-checkpoint implementation
expects the model parameters passed to it to require gradients.  Therefore we
must not call ``model.requires_grad_(False)`` here.  ``torch.autograd.grad`` is
asked only for the gradient with respect to x_t, so this script neither
accumulates parameter gradients nor changes model weights.

Run from ``projects``:

    python utils/diag_dps_zaps_gradient.py \
      --image ../DPS/data/samples/00000.png --device cuda
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, TASK_CONFIGS, ZAPS_CONFIG, ZETA_INIT_BY_TASK
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS, ddpm_posterior_step


TASK = "super_resolution"
DEFAULT_CHECK_STEPS = (0, 4, 14, 23, 29)


def tensor_norm(value: torch.Tensor) -> float:
    return value.detach().float().flatten().norm().item()


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(
        left.detach().float().flatten(),
        right.detach().float().flatten(),
        dim=0,
    ).item()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact DPS and approximate ZAPS likelihood gradients."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--zeta", type=float, default=0.1)
    parser.add_argument("--d-init", type=float, default=0.2)
    parser.add_argument("--dps-scale", type=float, default=0.3)
    parser.add_argument(
        "--check-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECK_STEPS),
        help="Reverse-trajectory positions to inspect (0 is t=999 for the paper grid).",
    )
    args = parser.parse_args()

    device = args.device
    set_seed(args.seed)
    x0_gt = load_image_as_tensor(args.image).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    with torch.no_grad():
        measurement = operator(x0_gt)

    diffusion_model = load_diffusion_model("imagenet", device)
    diffusion_model.model.eval()
    if not all(parameter.requires_grad for parameter in diffusion_model.model.parameters()):
        raise RuntimeError(
            "guided-diffusion checkpoint backward requires all model parameters "
            "to keep requires_grad=True for this diagnostic"
        )

    cfg = {
        **ZAPS_CONFIG,
        "zeta_init": args.zeta,
        "d_init": args.d_init,
        "use_learned_var": False,
        "sampler_mode": "ddpm",
    }
    set_seed(args.seed)
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **cfg,
    )
    with torch.no_grad():
        zaps.zeta.fill_(args.zeta)
        zaps.D.fill_(args.d_init)

    set_seed(args.seed + 7)
    x_t = torch.randn(
        1, 3, IMG_SIZE[0], IMG_SIZE[1], device=device
    )

    timesteps = zaps.tau
    alphas_cumprod = diffusion_model.alphas_cumprod
    number_of_steps = len(timesteps)
    check_steps = set(args.check_steps)
    invalid_steps = sorted(
        step for step in check_steps if step < 0 or step >= number_of_steps
    )
    if invalid_steps:
        raise ValueError(
            f"check steps must be in [0, {number_of_steps - 1}], got {invalid_steps}"
        )

    print("\n=== DPS exact gradient vs ZAPS approximate gradient ===")
    print(f"timesteps (ascending): {[int(t) for t in timesteps.tolist()]}")
    print(
        f"{'step':>4} {'t':>4} {'cos(ZAPS,DPS)':>15} "
        f"{'norm(approx)/norm(exact)':>25} "
        f"{'norm(ZAPS correction)/norm(DPS correction)':>42} "
        f"{'norm(correction)/norm(uncond increment)':>39}"
    )

    for reverse_step, timestep_index in enumerate(
        range(number_of_steps - 1, -1, -1)
    ):
        t_current = int(timesteps[timestep_index].item())
        t_previous = (
            int(timesteps[timestep_index - 1].item())
            if timestep_index > 0
            else -1
        )
        t_batch = torch.tensor([t_current], device=device, dtype=torch.long)
        alpha_bar = alphas_cumprod[t_current]

        if reverse_step in check_steps:
            # Keep model parameter requires_grad flags intact: the custom
            # gradient-checkpoint backward expects them to be differentiable.
            x_probe = x_t.detach().requires_grad_(True)
            epsilon = diffusion_model._predict_eps(x_probe, t_batch)
            x0_graph = (
                x_probe - (1.0 - alpha_bar).sqrt() * epsilon
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            x0_graph = x0_graph.clamp(-1.0, 1.0)
            residual_graph = measurement - operator.H(x0_graph)
            residual_l2 = residual_graph.flatten().norm()

            # DPS performs x <- x - scale * d(loss)/d(x_t).
            exact_dps_direction = -torch.autograd.grad(
                residual_l2, x_probe, create_graph=False, retain_graph=False
            )[0].detach()
            x0_pred = x0_graph.detach()
            residual = residual_graph.detach()

            with torch.no_grad():
                adjoint_residual = operator.transpose(residual)
                hessian_term = zaps.dwt.synthesis(
                    zaps.D[timestep_index]
                    * zaps.dwt.analysis(adjoint_residual)
                )
                zaps_raw_direction = (
                    adjoint_residual
                    + (1.0 - alpha_bar) * hessian_term
                ) / alpha_bar.sqrt().clamp_min(1e-8)

                # DPS differentiates an L2 norm, whereas the ZAPS equation uses
                # the raw residual. Divide by ||r|| for a direction/scale
                # comparison independent of that objective convention.
                zaps_l2_direction = (
                    zaps_raw_direction / residual_l2.detach().clamp_min(1e-8)
                )
                unconditional = ddpm_posterior_step(
                    x_t,
                    x0_pred,
                    t_current,
                    t_previous,
                    alphas_cumprod,
                    eta=1.0,
                    mode="ddpm",
                )
                correction = args.zeta * zaps_raw_direction
                unconditional_increment = unconditional - x_t

                direction_cosine = cosine_similarity(
                    zaps_l2_direction, exact_dps_direction
                )
                normalized_norm_ratio = tensor_norm(zaps_l2_direction) / (
                    tensor_norm(exact_dps_direction) + 1e-12
                )
                applied_norm_ratio = tensor_norm(correction) / (
                    args.dps_scale * tensor_norm(exact_dps_direction) + 1e-12
                )
                increment_ratio = tensor_norm(correction) / (
                    tensor_norm(unconditional_increment) + 1e-12
                )

                print(
                    f"{reverse_step:4d} {t_current:4d} "
                    f"{direction_cosine:15.6f} "
                    f"{normalized_norm_ratio:25.4f} "
                    f"{applied_norm_ratio:42.4f} "
                    f"{increment_ratio:39.4f}"
                )
                x_t = unconditional + correction

            del (
                x_probe,
                epsilon,
                x0_graph,
                residual_graph,
                residual_l2,
                exact_dps_direction,
            )
        else:
            with torch.no_grad():
                epsilon = diffusion_model._predict_eps(x_t, t_batch)
                x0_pred = (
                    x_t - (1.0 - alpha_bar).sqrt() * epsilon
                ) / alpha_bar.sqrt().clamp_min(1e-8)
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                residual = measurement - operator.H(x0_pred)
                adjoint_residual = operator.transpose(residual)
                hessian_term = zaps.dwt.synthesis(
                    zaps.D[timestep_index]
                    * zaps.dwt.analysis(adjoint_residual)
                )
                zaps_raw_direction = (
                    adjoint_residual
                    + (1.0 - alpha_bar) * hessian_term
                ) / alpha_bar.sqrt().clamp_min(1e-8)
                unconditional = ddpm_posterior_step(
                    x_t,
                    x0_pred,
                    t_current,
                    t_previous,
                    alphas_cumprod,
                    eta=1.0,
                    mode="ddpm",
                )
                x_t = unconditional + args.zeta * zaps_raw_direction

    print("\nInterpretation:")
    print("  cosine < 0.5 or negative: the approximate Jacobian direction is the primary suspect")
    print("  cosine > 0.9 but a large norm mismatch: guidance scaling/objective normalization is suspect")
    print("  correction/unconditional-increment > 1 at high noise: high-noise guidance is dominating")


if __name__ == "__main__":
    main()
