"""Audit the actual last optimization unroll, not a new post-update sample.

Replay saved baseline settings/measurement. A diagnostic-only subclass snapshots
parameters and RNG immediately BEFORE the last unroll; optimize itself remains
unchanged. The final Adam update must not be used to replay _last_opt_x0.
Ground truth is used only for reporting, never selecting an output or schedule.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch

PROJECTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTS_ROOT)

from configs.config import IMG_SIZE, METRICS_CONFIG
from modules.degradations import get_operator
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.zaps_algorithm import ZAPS
from modules.dataset_loader import tensor_to_image
from utils.diag_ffhq_regression import TransposeMode, tensor_psnr
from utils.diag_ffhq_state_schedule import finite_json, git_info
from utils.diag_learning_rate_ablation import set_seed
from utils.diag_zaps_trace_audit import trace, relative_error, classify_parity
from utils.metrics import compute_all_metrics


class LastUnrollSnapshotZAPS(ZAPS):
    """Observe the call boundary without replacing training or its graph."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_calls = 0
        self.audit_state = None

    def _reverse_diffusion(self, *args, **kwargs):
        self.audit_calls += 1
        if self.audit_calls == self.num_epochs:
            init_noise = kwargs.get("init_noise")
            if init_noise is None:
                raise RuntimeError("last-unroll audit requires optimize's fixed x_T")
            self.audit_state = {
                "zeta": self.zeta.detach().cpu().clone(),
                "D": self.D.detach().cpu().clone(),
                "tau": self.tau.detach().cpu().clone(),
                "init_noise": init_noise.detach().cpu().clone(),
                "cpu_rng": torch.get_rng_state().clone(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        return super()._reverse_diffusion(*args, **kwargs)


def restore_rng(state):
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def late_summary(rows, final_psnr):
    # A common time domain, not equal step indices on unequal grids.
    late_rows = [row for row in rows if row["t"] <= 400]
    peak = max(late_rows, key=lambda row: row["x0_psnr"])
    return {
        "late_domain": "t <= 400; x0 estimates before correction",
        "late_peak_t": peak["t"],
        "late_peak_x0_psnr": peak["x0_psnr"],
        "t0_x0_psnr": rows[-1]["x0_psnr"],
        "late_peak_to_t0_drop": peak["x0_psnr"] - rows[-1]["x0_psnr"],
        "residual_at_late_peak": peak["residual_norm"],
        "t0_pre_correction_residual": rows[-1]["residual_norm"],
        "final_float_psnr": final_psnr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", choices=("gaussian_deblur", "super_resolution"), default="gaussian_deblur")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = Path(args.run_dir)
    record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run_args = record["arguments"]
    definitions = {}
    for result in record["results"]:
        if result["task"] == args.task and abs(result["learning_rate"] - args.learning_rate) < 1e-12:
            if result["schedule"] in definitions:
                raise RuntimeError("duplicate source arm for task/LR/schedule")
            definitions[result["schedule"]] = result
    if set(definitions) != {"irregular_15_10_5", "uniform_30"}:
        raise RuntimeError("source must contain both schedules at the selected task/LR")
    ground_truth = load_image_as_tensor(run_args["image"]).to(args.device)
    measurement = torch.load(source / args.task / "measurement.pt", map_location=args.device, weights_only=True)
    model = load_diffusion_model(run_args["dataset"], args.device)
    operator = get_operator(args.task, device=args.device, **record["task_configs"][args.task])
    if args.task == "super_resolution":
        operator = TransposeMode(operator, run_args["sr_transpose"]).to(args.device)
    output_dir = source / ("optimized_trace_" + args.task + "_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_record = {
        "experiment": "last_optimization_unroll_trace",
        "git": git_info(), "source": str(source), "arguments": vars(args),
        "image": run_args["image"], "seed": run_args["seed"],
        "parameter_semantics": "before final unroll/final Adam update; matches last_opt output",
        "ground_truth_usage": "metrics only; no early stopping or best-PSNR output selection",
        "paired_scope": "same observation, x_T and per-epoch transition draws by seed reset",
        "results": {},
    }

    def save_records():
        (output_dir / "audit.json").write_text(json.dumps(finite_json(output_record), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Last optimization unroll audit (joint zeta+D, no parameter sweep) ===", flush=True)
    print(f"Records: {output_dir}", flush=True)
    for name, result in definitions.items():
        set_seed(int(run_args["seed"]))
        zaps = LastUnrollSnapshotZAPS(model, operator, img_size=IMG_SIZE[0], **result["config"])
        zaps.tau = torch.tensor(result["timesteps_ascending"], device=args.device)
        print(f"\n--- {name}: replay original optimization ---", flush=True)
        losses = zaps.optimize(measurement, verbose=True, x0_gt=ground_truth)
        optimized = zaps._last_opt_x0.detach()
        state = zaps.audit_state
        if state is None or zaps.audit_calls != zaps.num_epochs:
            raise RuntimeError("last-unroll snapshot missing or call count invalid")
        optimization_nfe = zaps._last_nfe
        # Saved after-update zeta/D do not generate the final optimized output.
        # Restore the actual parameters used to produce epoch 10, including D.
        with torch.no_grad():
            zaps.zeta.copy_(state["zeta"].to(zaps.device))
            zaps.D.copy_(state["D"].to(zaps.device))
        init_noise = state["init_noise"].to(zaps.device)
        torch.save(state, output_dir / f"{name}_last_unroll_state.pt")
        print(f"{name}: tracing epoch {zaps.num_epochs} and checking replay...", flush=True)
        restore_rng(state)
        reconstruction, rows = trace(zaps, measurement, ground_truth, init_noise)
        restore_rng(state)
        reference, nfe, _ = zaps.sample(measurement, init_noise=init_noise)
        reference_t = [row["t"] for row in zaps._indicator_log]
        restore_rng(state)
        repeat, repeat_nfe, _ = zaps.sample(measurement, init_noise=init_noise)
        repeat_t = [row["t"] for row in zaps._indicator_log]
        repeat_error = relative_error(reference, repeat)
        metadata_equal = (
            nfe == repeat_nfe == len(rows) == len(zaps.tau)
            and optimization_nfe == len(zaps.tau) * zaps.num_epochs
            and reference_t == repeat_t == [row["t"] for row in rows]
        )
        trace_gate = classify_parity(repeat_error, relative_error(reference, reconstruction), metadata_equal)
        optimized_gate = classify_parity(repeat_error, relative_error(reference, optimized), metadata_equal)
        gates = {"passed": trace_gate["passed"] and optimized_gate["passed"],
                 "trace_vs_core": trace_gate, "last_opt_vs_core_replay": optimized_gate}
        final_psnr = tensor_psnr(ground_truth, optimized)
        metrics = compute_all_metrics(optimized, ground_truth, lpips_net=METRICS_CONFIG["lpips_net"])
        summary = late_summary(rows, final_psnr)
        arm = {
            "config": result["config"], "gates": gates, "summary": summary,
            "metrics": metrics, "archived_optimized_psnr": result["psnr"],
            "loss_history": losses, "archived_loss_history": result["loss_history"],
            "optimization_nfe": optimization_nfe,
            "audit_replay_nfe": len(rows) + nfe + repeat_nfe,
            "total_nfe": optimization_nfe + len(rows) + nfe + repeat_nfe,
            "epoch_used_zeta_ascending": state["zeta"].tolist(), "rows": rows,
        }
        output_record["results"][name] = arm
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tensor_to_image(optimized.cpu().squeeze(0), denormalize=True).save(output_dir / f"{name}.png")
        save_records()
        print("Gate:", json.dumps(finite_json(gates), ensure_ascii=False, indent=2), flush=True)
        if not gates["passed"]:
            raise RuntimeError(f"optimized trace gate failed; records saved to {output_dir}")
        print(f"{'k':>3} {'t':>4} {'zeta':>8} {'corr/dunc':>10} {'resid':>9} {'clip%':>7} {'x0PSNR':>8}", flush=True)
        for row in rows:
            print(f"{row['step']:3d} {row['t']:4d} {row['zeta']:8.5f} "
                  f"{row['correction_over_uncond_increment']:10.3f} {row['residual_norm']:9.2f} "
                  f"{100 * row['raw_x0_clip_fraction']:7.1f} {row['x0_psnr']:8.2f}", flush=True)
        print(f"Final optimized PSNR: {metrics['psnr']:.4f}; archived: {result['psnr']:.4f}", flush=True)
        del zaps, optimized, state, reconstruction, reference, repeat, init_noise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Paired optimized trace summary ===", flush=True)
    print(f"{'schedule':>20} {'finalPSNR':>10} {'SSIM':>8} {'LPIPS':>8} {'peak_t':>7} {'latePeak':>9} {'t0x0':>9} {'drop':>8} {'MSE':>10}", flush=True)
    for name, arm in output_record["results"].items():
        s, m = arm["summary"], arm["metrics"]
        print(f"{name:>20} {m['psnr']:10.4f} {m['ssim']:8.4f} {m['lpips']:8.4f} "
              f"{s['late_peak_t']:7d} {s['late_peak_x0_psnr']:9.4f} {s['t0_x0_psnr']:9.4f} "
              f"{s['late_peak_to_t0_drop']:8.4f} {arm['loss_history'][-1]:10.6f}", flush=True)
    irregular = output_record["results"]["irregular_15_10_5"]
    uniform = output_record["results"]["uniform_30"]
    print(
        f"Irregular - uniform: final dPSNR={irregular['metrics']['psnr'] - uniform['metrics']['psnr']:+.4f}; "
        f"late-drop difference={irregular['summary']['late_peak_to_t0_drop'] - uniform['summary']['late_peak_to_t0_drop']:+.4f}",
        flush=True,
    )
    print("Late peak is a GT-only diagnostic, NOT an early-stopping result. Both outputs are the original last_opt output.", flush=True)
    print("Replay uses epoch-10 inputs/noises and pre-update parameters; it is NOT a fresh final sample.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
