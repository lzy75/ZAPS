"""Paired FFHQ ablation of zeta and D initialization values.

Use one varying list at a time to keep the experiment interpretable:

* zeta screening: multiple --zeta-inits with one --d-inits value;
* D screening: one --zeta-inits value with multiple --d-inits.

The recovered historical FFHQ convention is fixed throughout: legacy bicubic
transpose, fixed skip variance, DDPM, 15-10-5 schedule, and joint zeta+D.
"""

import argparse
import os
import sys

import torch


PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import TASK_CONFIGS
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from utils.diag_ffhq_regression import TransposeMode
from utils.diag_learning_rate_ablation import run_learning_rate, set_seed


TASK = "super_resolution"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired FFHQ zeta/D initialization ablation."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--zeta-inits", type=float, nargs="+", default=[0.05, 0.1, 0.2]
    )
    parser.add_argument("--d-inits", type=float, nargs="+", default=[0.2])
    args = parser.parse_args()

    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if any(value < 0 for value in args.zeta_inits + args.d_inits):
        raise ValueError("initialization values must be non-negative")
    if len(args.zeta_inits) > 1 and len(args.d_inits) > 1:
        raise ValueError(
            "vary only one branch at a time: use one zeta value or one D value"
        )

    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    base_operator = get_operator(TASK, device=args.device, **TASK_CONFIGS[TASK])
    operator = TransposeMode(base_operator, "legacy_bicubic").to(args.device)
    with torch.no_grad():
        measurement = operator(ground_truth)
    diffusion_model = load_diffusion_model("ffhq", args.device)

    print("\n=== Paired FFHQ initialization ablation ===", flush=True)
    print(
        f"lr={args.learning_rate:g}; transpose=legacy_bicubic; "
        "variance=fixed; schedule=15-10-5",
        flush=True,
    )
    print("All runs share y, x_T, and every DDPM transition-noise draw.", flush=True)

    results = []
    for zeta_init in args.zeta_inits:
        for d_init in args.d_inits:
            print(
                f"\n--- zeta_init={zeta_init:g}, D_init={d_init:g} ---",
                flush=True,
            )
            result = run_learning_rate(
                args.learning_rate,
                diffusion_model,
                operator,
                measurement,
                ground_truth,
                args.seed,
                False,
                True,
                zeta_init,
                d_init,
            )
            result["zeta_init"] = zeta_init
            result["d_init"] = d_init
            results.append(result)

    print("\n=== Summary ===", flush=True)
    print(
        f"{'zeta init':>11} {'D init':>9} {'PSNR':>10} {'final MSE':>12} "
        f"{'final ||r||':>14} {'zeta min':>11} {'zeta max':>11} "
        f"{'D delta RMS':>13} {'D delta max':>13} {'seconds':>10}",
        flush=True,
    )
    for result in results:
        print(
            f"{result['zeta_init']:11.4g} {result['d_init']:9.4g} "
            f"{result['psnr']:10.4f} {result['losses'][-1]:12.6f} "
            f"{result['residual']:14.4f} {result['zeta_min']:11.5f} "
            f"{result['zeta_max']:11.5f} {result['d_delta_rms']:13.6f} "
            f"{result['d_delta_max']:13.6f} {result['seconds']:10.1f}",
            flush=True,
        )

    print("\nInterpretation:", flush=True)
    print(
        "  choose by reconstruction PSNR, not by the lowest measurement MSE alone",
        flush=True,
    )
    print(
        "  if the best value is an endpoint, expand only in that direction once",
        flush=True,
    )
    print(
        "  after zeta is selected, keep it fixed and run the D-init screen",
        flush=True,
    )


if __name__ == "__main__":
    main()
