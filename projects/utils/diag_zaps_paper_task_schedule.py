"""Compare fixed ZAPS grids on the two quantitatively reported ImageNet tasks.

This is a baseline audit, without a state scheduler. Each task/LR pair shares
one observation and resets the sampling RNG before each grid. The defaults are
30 DDPM steps x 10 epochs, lr=0.001, joint zeta+D, fixed skip variance, and the
current exact adjoint. These are the repository's reference settings, not a
claim that every implementation detail matches the authors' implementation.

The same script can audit the historical FFHQ SR schedule comparison by using
--dataset ffhq --tasks super_resolution --sr-transpose legacy_bicubic and
--learning-rates 0.001 0.01. The legacy map is explicitly recorded as such.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import (
    DATASET_MODEL_MAP, IMG_SIZE, METRICS_CONFIG, RESULTS_DIR,
    TASK_CONFIGS, ZAPS_CONFIG, ZETA_INIT_BY_TASK,
)
from modules.dataset_loader import tensor_to_image
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS, build_irregular_timesteps
from utils.diag_ffhq_regression import TransposeMode, tensor_psnr
from utils.diag_ffhq_state_schedule import finite_json, git_info
from utils.diag_ffhq_timestep_ablation import rounded_spacing
from utils.diag_learning_rate_ablation import set_seed
from utils.metrics import compute_all_metrics


def save_image(tensor, destination):
    tensor_to_image(tensor.detach().cpu().squeeze(0), denormalize=True).save(destination)


def run_arm(args, model, operator, measurement, ground_truth, task, lr, name, tau):
    config = {
        **ZAPS_CONFIG,
        "num_steps": 30,
        "schedule": (15, 10, 5),
        "num_epochs": 10,
        "lr": lr,
        "zeta_init": ZETA_INIT_BY_TASK[task],
        "d_init": 0.2,
        "eta": 1.0,
        "timestep_spacing": "linear",
        "use_learned_var": args.use_learned_var,
        "sampler_mode": "ddpm",
        "surrogate_score_jacobian": False,
    }
    # Model loading and metric evaluation may consume random numbers. Reset
    # immediately before every arm so x_T and transition draws are paired.
    set_seed(args.seed)
    zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **config)
    zaps.tau = tau.to(args.device)
    print(f"\n--- {task} | lr={lr:g} | {name} ---", flush=True)
    print(f"timesteps ascending: {tau.tolist()}", flush=True)
    started = time.time()
    losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
    elapsed = time.time() - started
    reconstruction = zaps._last_opt_x0.detach()
    metrics = compute_all_metrics(
        reconstruction, ground_truth, lpips_net=METRICS_CONFIG["lpips_net"]
    )
    with torch.no_grad():
        residual = (measurement - operator.H(reconstruction)).norm().item()
        delta_d = zaps.D.detach() - config["d_init"]
    if zaps._last_nfe != 300:
        raise RuntimeError(f"expected 300 NFE, got {zaps._last_nfe}")
    result = {
        "dataset": args.dataset,
        "image": os.path.abspath(args.image),
        "task": task,
        "learning_rate": lr,
        "schedule": name,
        **metrics,
        "float_psnr": tensor_psnr(ground_truth, reconstruction),
        "measurement_mse": losses[-1],
        "residual": residual,
        "zeta_min": zaps.zeta.detach().min().item(),
        "zeta_max": zaps.zeta.detach().max().item(),
        "d_delta_rms": delta_d.square().mean().sqrt().item(),
        "nfe": zaps._last_nfe,
        "seconds": elapsed,
    }
    detail = {
        **result,
        "config": config,
        "timesteps_ascending": tau.tolist(),
        "loss_history": losses,
        "zeta_ascending": zaps.zeta.detach().cpu().tolist(),
    }
    reconstruction_cpu = reconstruction.cpu()
    del zaps, reconstruction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, detail, reconstruction_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", choices=("imagenet", "ffhq"), default="imagenet")
    parser.add_argument(
        "--tasks", nargs="+", choices=("gaussian_deblur", "super_resolution"),
        default=["gaussian_deblur", "super_resolution"],
    )
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--sr-transpose", choices=("exact", "legacy_bicubic"), default="exact"
    )
    parser.add_argument("--use-learned-var", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if not Path(args.image).is_file():
        raise FileNotFoundError(os.path.abspath(args.image))
    if any(lr <= 0 for lr in args.learning_rates):
        parser.error("learning rates must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must be unique")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("learning rates must be unique")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        RESULTS_DIR, "diag_zaps_paper_task_schedule",
        args.dataset + "_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    ground_truth = load_image_as_tensor(args.image).to(args.device)
    model = load_diffusion_model(args.dataset, args.device)
    total_steps = int(model.num_steps)
    grids = {
        "irregular_15_10_5": build_irregular_timesteps(
            total_steps=total_steps, schedule=(15, 10, 5), spacing="linear"
        ),
        "uniform_30": rounded_spacing(total_steps, 30, 1.0),
    }
    for name, tau in grids.items():
        if tau.numel() != 30 or tau.unique().numel() != 30:
            raise RuntimeError(f"invalid grid: {name}")
        if int(tau[0]) != 0 or int(tau[-1]) != total_steps - 1:
            raise RuntimeError(f"grid must preserve both endpoints: {name}")

    record = {
        "experiment": "zaps_two_task_fixed_schedule_audit",
        "git": git_info(),
        "arguments": vars(args),
        "checkpoint_path": DATASET_MODEL_MAP[args.dataset],
        "task_configs": {task: TASK_CONFIGS[task] for task in args.tasks},
        "final_mode": "last_opt",
        "metric_convention": "psnr/ssim use existing uint8 conversion; float_psnr also saved",
        "paired_scope": "within each task, all arms share y, x_T and transition noise draws",
        "results": [],
    }
    rows = []
    print("\n=== Fixed ZAPS task x schedule audit ===", flush=True)
    print(
        f"dataset={args.dataset}; 30 steps x 10 epochs=300 NFE; "
        f"lr={args.learning_rates}; SR transpose={args.sr_transpose}; "
        f"learned_var={args.use_learned_var}; joint zeta+D; sigma=0.05",
        flush=True,
    )
    print(f"Records: {output_dir}", flush=True)
    for task in args.tasks:
        task_dir = output_dir / task
        task_dir.mkdir(exist_ok=True)
        base = get_operator(task, device=args.device, **TASK_CONFIGS[task])
        operator = (
            TransposeMode(base, args.sr_transpose).to(args.device)
            if task == "super_resolution" else base
        )
        set_seed(args.seed)
        with torch.no_grad():
            measurement = operator(ground_truth)
            observation = measurement
            if measurement.shape[-2:] != ground_truth.shape[-2:]:
                observation = F.interpolate(
                    measurement, size=ground_truth.shape[-2:],
                    mode="bicubic", align_corners=False,
                )
        save_image(ground_truth, task_dir / "ground_truth.png")
        save_image(observation, task_dir / "observation.png")
        torch.save(measurement.cpu(), task_dir / "measurement.pt")
        for lr in args.learning_rates:
            for name, tau in grids.items():
                result, detail, reconstruction = run_arm(
                    args, model, operator, measurement, ground_truth, task, lr, name, tau
                )
                rows.append(result)
                record["results"].append(detail)
                save_image(reconstruction, task_dir / f"lr_{lr:g}_{name}.png")
                # Persist after every run: interrupted experiments retain the
                # completed arms, their loss histories, and exact measurement.
                with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
                    json.dump(finite_json(record), handle, ensure_ascii=False, indent=2)
                with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"RESULT {task} {name} lr={lr:g}: "
                    f"PSNR={result['psnr']:.4f} SSIM={result['ssim']:.4f} "
                    f"LPIPS={result['lpips']:.4f} NFE={result['nfe']}", flush=True,
                )

    print("\n=== Irregular minus uniform (matched task and LR) ===", flush=True)
    print(f"{'task':>18} {'lr':>7} {'irregular':>10} {'uniform':>10} {'dPSNR':>9} {'dSSIM':>9} {'dLPIPS':>9}", flush=True)
    for task in args.tasks:
        for lr in args.learning_rates:
            pair = {row["schedule"]: row for row in rows if row["task"] == task and row["learning_rate"] == lr}
            irregular, uniform = pair["irregular_15_10_5"], pair["uniform_30"]
            print(
                f"{task:>18} {lr:7g} {irregular['psnr']:10.4f} {uniform['psnr']:10.4f} "
                f"{irregular['psnr'] - uniform['psnr']:+9.4f} "
                f"{irregular['ssim'] - uniform['ssim']:+9.4f} "
                f"{irregular['lpips'] - uniform['lpips']:+9.4f}", flush=True,
            )
    print("Positive dPSNR/dSSIM and negative dLPIPS favor irregular.", flush=True)
    print("A single-image result checks the implementation/trend; it is not the paper's 1000-image mean.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
