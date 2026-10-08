"""Matched train-policy x replay-policy audit for one fixed late-noise scale.

Keep the archived rho=1 baseline; only train rho=0.75 (or the explicitly chosen
single scale). This is a modified sampling setting, NOT original-paper recovery
or state-aware scheduling. Both zeta and D learn with the original optimizer.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, METRICS_CONFIG
from modules import zaps_algorithm
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.dataset_loader import tensor_to_image
from utils.diag_ffhq_regression import TransposeMode
from utils.diag_ffhq_state_schedule import finite_json, git_info
from utils.diag_learning_rate_ablation import set_seed
from utils.diag_zaps_trace_audit import relative_error, classify_parity
from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS, restore_rng
from utils.diag_zaps_late_component import rng_equal, rng_snapshot
from utils.metrics import compute_all_metrics


class LateNoiseZAPS(LastUnrollSnapshotZAPS):
    """Apply the same local DDPM policy in original optimization and sample."""

    def __init__(self, *args, late_start=333, late_noise_scale=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sampler_mode != "ddpm":
            raise ValueError("noise-only policy requires DDPM")
        self.late_start = late_start
        self.late_noise_scale = late_noise_scale

    def _reverse_diffusion(self, *args, **kwargs):
        original = zaps_algorithm.ddpm_posterior_step

        def scheduled_step(x_t, x0_pred, t_curr, t_prev, alphas_cumprod,
                           eta=1.0, learned_log_var=None, mode="ddpm"):
            if t_curr <= self.late_start and self.late_noise_scale != 1.0:
                eta = eta * self.late_noise_scale
            return original(x_t, x0_pred, t_curr, t_prev, alphas_cumprod,
                            eta=eta, learned_log_var=learned_log_var, mode=mode)

        # This diagnostic process is single-threaded. The core implementation,
        # loss, gradient path and optimizer remain unchanged; finally restores
        # the helper even if inference/backward raises. No permanent mutation.
        with patch.object(zaps_algorithm, "ddpm_posterior_step", scheduled_step):
            return super()._reverse_diffusion(*args, **kwargs)


def load_unroll(zaps, state):
    zaps.tau = state["tau"].to(zaps.device)
    with torch.no_grad():
        zaps.zeta.copy_(state["zeta"].to(zaps.device))
        zaps.D.copy_(state["D"].to(zaps.device))
    return state["init_noise"].to(zaps.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--late-start", type=int, default=333)
    parser.add_argument("--noise-scale", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not math.isfinite(args.noise_scale) or not 0 < args.noise_scale < 1:
        parser.error("choose one nonzero noise scale below 1; no sweep")
    if not 0 <= args.late_start < 999:
        parser.error("late-start must be in [0,998]")
    trace_dir = Path(args.trace_dir)
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source = Path(saved["source"])
    source_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    task = saved["arguments"]["task"]
    run_args = source_record["arguments"]
    if set(saved["results"]) != {"irregular_15_10_5", "uniform_30"}:
        raise RuntimeError("source must contain both schedules")
    if any(not arm["gates"]["passed"] for arm in saved["results"].values()):
        raise RuntimeError("source replay gate must have passed")
    gt = load_image_as_tensor(saved["image"]).to(args.device)
    y = torch.load(source / task / "measurement.pt", map_location=args.device, weights_only=True)
    model = load_diffusion_model(run_args["dataset"], args.device)
    operator = get_operator(task, device=args.device, **source_record["task_configs"][task])
    if task == "super_resolution":
        operator = TransposeMode(operator, run_args["sr_transpose"]).to(args.device)
    output_dir = trace_dir / ("late_noise_training_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"git": git_info(), "arguments": vars(args), "source_trace": str(trace_dir),
              "semantics": "train and evaluate with same fixed late-noise policy; original baseline retained",
              "formula": "DDPM mean + (eta*rho)*sigma*z + original correction at t<=late_start",
              "ground_truth_usage": "report metrics only; no output selection or early stopping",
              "results": [], "gates": {}, "pairing": {}, "optimization_nfe": {}}

    def save_records():
        (output_dir / "run.json").write_text(json.dumps(finite_json(record), ensure_ascii=False, indent=2), encoding="utf-8")
        if record["results"]:
            with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record["results"][0]))
                writer.writeheader()
                writer.writerows(record["results"])

    print(f"\n=== Train/evaluate late noise consistency; rho={args.noise_scale:g}, t<={args.late_start} ===", flush=True)
    print("Original settings remain archived. One new training per schedule; lr/zeta/D/observation unchanged.", flush=True)
    print(f"Records: {output_dir}", flush=True)
    for name, arm in saved["results"].items():
        old_state = torch.load(trace_dir / f"{name}_last_unroll_state.pt", map_location="cpu", weights_only=True)
        baseline_metrics = None
        baseline_end_rng = None
        for train_scale in (1.0, args.noise_scale):
            set_seed(int(saved["seed"]))
            zaps = LateNoiseZAPS(model, operator, img_size=IMG_SIZE[0],
                                 late_start=args.late_start, late_noise_scale=train_scale, **arm["config"])
            zaps.tau = old_state["tau"].to(args.device)
            expected_output = None
            if train_scale == 1.0:
                state = old_state
                optimization_nfe = 0
            else:
                print(f"\n{name}: optimize rho={train_scale:g} (joint zeta+D)...", flush=True)
                losses = zaps.optimize(y, verbose=True, x0_gt=gt)
                expected_output = zaps._last_opt_x0.detach()
                state = zaps.audit_state
                optimization_nfe = zaps._last_nfe
                if state is None or optimization_nfe != len(zaps.tau) * zaps.num_epochs:
                    raise RuntimeError("training snapshot or NFE invalid")
                pairing = {"x_T_equal": torch.equal(old_state["init_noise"], state["init_noise"]),
                           "epoch10_rng_equal": rng_equal(old_state, state)}
                record["pairing"][name] = pairing
                record["optimization_nfe"][name] = optimization_nfe
                record.setdefault("new_loss_history", {})[name] = losses
                torch.save(state, output_dir / f"{name}_rho_{train_scale:g}_last_unroll_state.pt")
                save_records()
                if not all(pairing.values()):
                    raise RuntimeError(f"training random draws not paired: {pairing}; records: {output_dir}")
            init_noise = load_unroll(zaps, state)
            for eval_scale in (train_scale, args.noise_scale if train_scale == 1.0 else 1.0):
                zaps.late_noise_scale = eval_scale
                restore_rng(state)
                output, nfe, _ = zaps.sample(y, init_noise=init_noise)
                end_rng = rng_snapshot()
                if baseline_end_rng is None:
                    baseline_end_rng = end_rng
                if nfe != len(zaps.tau) or not rng_equal(baseline_end_rng, end_rng):
                    record.setdefault("evaluation_pairing_failed", []).append(
                        {"schedule": name, "train": train_scale, "eval": eval_scale})
                    save_records()
                    raise RuntimeError(f"evaluation NFE/RNG sequence mismatch; records: {output_dir}")
                if eval_scale == train_scale:
                    restore_rng(state)
                    repeat, repeat_nfe, _ = zaps.sample(y, init_noise=init_noise)
                    repeat_error = relative_error(output, repeat)
                    expected_error = relative_error(output, expected_output) if expected_output is not None else 0.0
                    gate = classify_parity(repeat_error, expected_error,
                                           nfe == repeat_nfe == len(zaps.tau))
                    key = f"{name}_train_{train_scale:g}"
                    record["gates"][key] = gate
                    print(f"{key} gate: {json.dumps(finite_json(gate))}", flush=True)
                    save_records()
                    if not gate["passed"]:
                        raise RuntimeError(f"matched training replay failed; records: {output_dir}")
                    del repeat
                metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
                if train_scale == eval_scale == 1.0:
                    baseline_metrics = metrics
                    delta = abs(metrics["psnr"] - arm["metrics"]["psnr"])
                    record["gates"][key]["archived_psnr_absolute_delta"] = delta
                    if delta > 1e-4:
                        record["gates"][key]["passed"] = False
                        save_records()
                        raise RuntimeError(f"archived baseline differs: {delta}; records: {output_dir}")
                row = {"schedule": name, "train_noise_scale": train_scale, "eval_noise_scale": eval_scale,
                       "matched_policy": train_scale == eval_scale, **metrics,
                       "dPSNR": metrics["psnr"] - baseline_metrics["psnr"],
                       "dSSIM": metrics["ssim"] - baseline_metrics["ssim"],
                       "dLPIPS": metrics["lpips"] - baseline_metrics["lpips"],
                       "residual": (y - operator.H(output)).detach().norm().item(),
                       "eval_nfe": nfe}
                record["results"].append(row)
                tensor_to_image(output.cpu().squeeze(0), denormalize=True).save(
                    output_dir / f"{name}_train_{train_scale:g}_eval_{eval_scale:g}.png")
                save_records()
                print(f"{name} train={train_scale:g} eval={eval_scale:g}: PSNR={metrics['psnr']:.4f} "
                      f"SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}", flush=True)
            del zaps, state, output, init_noise, expected_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del old_state
    print("\n=== Paired training-policy summary ===", flush=True)
    print(f"{'schedule':>20} {'train':>7} {'eval':>7} {'PSNR':>9} {'delta':>9} {'SSIM':>8} {'LPIPS':>8} {'dLPIPS':>9}", flush=True)
    for row in record["results"]:
        print(f"{row['schedule']:>20} {row['train_noise_scale']:7.2f} {row['eval_noise_scale']:7.2f} "
              f"{row['psnr']:9.4f} {row['dPSNR']:+9.4f} {row['ssim']:8.4f} "
              f"{row['lpips']:8.4f} {row['dLPIPS']:+9.4f}", flush=True)
    print("Matched rho<1 is a modified setting, not proof of original-paper reproduction or state-aware gain.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
