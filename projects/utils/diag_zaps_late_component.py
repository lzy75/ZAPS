"""Saved-last-unroll DDPM noise x guidance intervention, without reoptimizing.

These are causal diagnostics, not a new trained/paper baseline. Below the fixed
low-third boundary t<=333, discard noise and/or guidance. Original posterior
draws are still consumed so disabling noise cannot shift later RNG draws.
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
from utils.diag_zaps_trace_audit import trace, relative_error, classify_parity
from utils.diag_zaps_optimized_trace import restore_rng, late_summary
from utils.metrics import compute_all_metrics


def rng_snapshot():
    return {"cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def rng_equal(first, second):
    if not torch.equal(first["cpu_rng"], second["cpu_rng"]):
        return False
    a, b = first["cuda_rng"], second["cuda_rng"]
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--late-start", type=int, default=333)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.late_start < 999:
        parser.error("late-start must be between 0 and 998")
    trace_dir = Path(args.trace_dir)
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source = Path(saved["source"])
    source_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    task = saved["arguments"]["task"]
    run_args = source_record["arguments"]
    if set(saved["results"]) != {"irregular_15_10_5", "uniform_30"}:
        raise RuntimeError("saved optimized trace must contain both schedules")
    for arm in saved["results"].values():
        if not arm["gates"]["passed"]:
            raise RuntimeError("source last-opt trace gate did not pass")
        if arm["config"]["sampler_mode"] != "ddpm":
            raise RuntimeError("this diagnostic separates noise from drift only for DDPM")
    gt = load_image_as_tensor(saved["image"]).to(args.device)
    y = torch.load(source / task / "measurement.pt", map_location=args.device, weights_only=True)
    model = load_diffusion_model(run_args["dataset"], args.device)
    operator = get_operator(task, device=args.device, **source_record["task_configs"][task])
    if task == "super_resolution":
        operator = TransposeMode(operator, run_args["sr_transpose"]).to(args.device)
    output_dir = trace_dir / ("late_components_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"git": git_info(), "arguments": vars(args), "source_trace": str(trace_dir),
              "semantics": "frozen trained parameters; interventions only at t<=late_start; no reoptimization",
              "results": [], "gates": {}}
    variants = (("baseline", 1.0, 1.0), ("late_noise_off", 0.0, 1.0),
                ("late_guidance_off", 1.0, 0.0), ("late_both_off", 0.0, 0.0))

    def save_records():
        (output_dir / "run.json").write_text(json.dumps(finite_json(record), ensure_ascii=False, indent=2), encoding="utf-8")
        if record["results"]:
            with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record["results"][0]))
                writer.writeheader()
                writer.writerows(record["results"])

    print(f"\n=== Late DDPM noise x guidance diagnostic; t<={args.late_start} ===", flush=True)
    print("No optimizer runs; same saved zeta/D, x_T and posterior draws.", flush=True)
    print(f"Records: {output_dir}", flush=True)
    for name, arm in saved["results"].items():
        state = torch.load(trace_dir / f"{name}_last_unroll_state.pt", map_location="cpu", weights_only=True)
        zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **arm["config"])
        zaps.tau = state["tau"].to(args.device)
        with torch.no_grad():
            zaps.zeta.copy_(state["zeta"].to(zaps.device))
            zaps.D.copy_(state["D"].to(zaps.device))
        init_noise = state["init_noise"].to(zaps.device)
        print(f"\n{name}: baseline replay gate...", flush=True)
        restore_rng(state)
        reference, nfe, _ = zaps.sample(y, init_noise=init_noise)
        reference_t = [row["t"] for row in zaps._indicator_log]
        restore_rng(state)
        repeat, repeat_nfe, _ = zaps.sample(y, init_noise=init_noise)
        repeat_error = relative_error(reference, repeat)
        baseline_rng = None
        baseline_psnr = None
        for variant, noise_scale, guidance_scale in variants:
            restore_rng(state)
            output, rows = trace(zaps, y, gt, init_noise, late_start=args.late_start,
                                 late_noise_scale=noise_scale, late_guidance_scale=guidance_scale)
            end_rng = rng_snapshot()
            if variant == "baseline":
                baseline_rng = end_rng
                gate = classify_parity(repeat_error, relative_error(reference, output),
                                       nfe == repeat_nfe == len(rows)
                                       and reference_t == [row["t"] for row in rows])
                record["gates"][name] = gate
                print(json.dumps(finite_json(gate), indent=2), flush=True)
                save_records()
                if not gate["passed"]:
                    raise RuntimeError(f"baseline replay failed; records: {output_dir}")
            if not rng_equal(baseline_rng, end_rng):
                raise RuntimeError(f"posterior RNG draw sequence shifted in {name}/{variant}")
            metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
            if variant == "baseline":
                baseline_psnr = metrics["psnr"]
                archive_delta = abs(baseline_psnr - arm["metrics"]["psnr"])
                record["gates"][name]["archived_psnr_absolute_delta"] = archive_delta
                if archive_delta > 1e-4:
                    record["gates"][name]["passed"] = False
                    save_records()
                    raise RuntimeError(f"saved baseline metrics not reproduced: delta={archive_delta}; records: {output_dir}")
            summary = late_summary(rows, tensor_psnr(gt, output))
            result = {"schedule": name, "variant": variant, **metrics,
                      "dPSNR": metrics["psnr"] - baseline_psnr,
                      "late_peak_x0_psnr": summary["late_peak_x0_psnr"],
                      "t0_x0_psnr": summary["t0_x0_psnr"],
                      "late_drop": summary["late_peak_to_t0_drop"],
                      "residual": (y - operator.H(output)).detach().norm().item(),
                      "nfe": len(rows), "rng_draws_equal": True}
            record["results"].append(result)
            with (output_dir / f"{name}_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            tensor_to_image(output.cpu().squeeze(0), denormalize=True).save(output_dir / f"{name}_{variant}.png")
            save_records()
            print(f"{name} {variant}: PSNR={metrics['psnr']:.4f} delta={result['dPSNR']:+.4f} "
                  f"SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}", flush=True)
        del zaps, state, reference, repeat, output, init_noise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n=== Paired late-component summary ===", flush=True)
    print(f"{'schedule':>20} {'variant':>19} {'PSNR':>9} {'delta':>9} {'SSIM':>8} {'LPIPS':>8} {'drop':>8} {'resid':>9}", flush=True)
    for row in record["results"]:
        print(f"{row['schedule']:>20} {row['variant']:>19} {row['psnr']:9.4f} {row['dPSNR']:+9.4f} "
              f"{row['ssim']:8.4f} {row['lpips']:8.4f} {row['late_drop']:8.4f} {row['residual']:9.3f}", flush=True)
    print("Interventions test frozen-path sensitivity; they are not a newly trained ZAPS baseline or proof of an implementation bug.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
