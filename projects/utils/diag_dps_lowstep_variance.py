"""Four controlled DPS runs: two saved 30-step grids x two variance policies.

Reuse the completed same-observation DPS-1000 result; do not rerun it or ZAPS.
All four new arms share H/y/image/model/x_T and draw-by-step sampling randomness.
Use original DPS DDPM and PS conditioning, scale=0.3, with original timestep
mapping via SpacedDiffusion. No optimizer, scale sweep, noise scaling or GT selection.
"""

import argparse
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.diag_dps_same_observation import (
    PureOperatorAdapter, file_sha256, set_seed, validate_dps_config,
)
from utils.diag_zaps_sampler_parity import load_dps_reference, rng_state, same_rng


def build_variants(grids):
    if set(grids) != {"irregular_15_10_5", "uniform_30"}:
        raise ValueError("Need both archived schedule grids")
    variants = []
    for schedule, grid in grids.items():
        if len(grid) != 30 or grid != sorted(set(grid)) or grid[0] != 0 or grid[-1] != 999:
            raise ValueError(f"Invalid original 30-step grid: {schedule}")
        for variance in ("fixed_small", "learned_range"):
            variants.append({"schedule": schedule, "variance": variance, "grid": list(grid)})
    return variants


def baseline_mismatches(record, expected):
    failures = []
    if record.get("status") != "complete" or record.get("dps", {}).get("nfe") != 1000:
        failures.append("not a completed 1000-step DPS run")
    for key in ("psnr", "ssim", "lpips"):
        value = record.get("dps", {}).get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            failures.append("missing/invalid DPS metric:" + key)
    for key, value in expected.items():
        if record.get(key) != value:
            failures.append(key)
    try:
        validate_dps_config(record["diffusion_config"], record["conditioning_config"])
    except (KeyError, ValueError, TypeError):
        failures.append("DPS baseline configuration")
    gates = record.get("gates", {})
    for key in ("source_xT_equal", "seed_xT_equal", "pure_H_equal", "full_timestep_grid",
                "model_grad_not_accumulated"):
        if gates.get(key) is not True:
            failures.append("gate:" + key)
    return failures


def find_baseline(trace_dir, explicit_dir, expected):
    """Use only completed, identity-matched archives in this exact trace directory."""
    candidates = ([Path(explicit_dir).resolve()] if explicit_dir else
                  sorted(Path(trace_dir).glob("dps_same_observation_*"), reverse=True))
    rejected = []
    for directory in candidates:
        try:
            record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            rejected.append(f"{directory.name}: {error}")
            continue
        failures = baseline_mismatches(record, expected)
        if not failures:
            return directory, record
        rejected.append(f"{directory.name}: {', '.join(failures)}")
    raise RuntimeError("No compatible completed DPS-1000 archive. No baseline will be rerun silently. "
                       "Use --dps-baseline-dir if needed.\n" + "\n".join(rejected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--dps-baseline-dir", default=None,
                        help="optional explicit completed same-observation DPS run; otherwise auto-find")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dps-root", default=str(REPO_ROOT / "DPS"))
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    original = json.loads((Path(saved["source"]) / "run.json").read_text(encoding="utf-8"))
    run_args = original["arguments"]
    task = saved["arguments"]["task"]
    if task != "gaussian_deblur" or run_args["dataset"] != "imagenet":
        parser.error("Use the existing ImageNet Gaussian-deblurring last-opt trace")
    for arm in saved["results"].values():
        if not arm["gates"]["passed"] or arm["config"]["eta"] != 1.0:
            raise RuntimeError("Source must be the original eta=1 baseline with passed last-opt replay")

    import torch
    import yaml
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.degradations import get_operator
    from modules.dataset_loader import tensor_to_image
    from utils.metrics import compute_all_metrics
    from utils.diag_ffhq_regression import tensor_psnr

    states = {name: torch.load(trace_dir / f"{name}_last_unroll_state.pt",
                               map_location="cpu", weights_only=True)
              for name in saved["results"]}
    variants = build_variants({name: state["tau"].tolist() for name, state in states.items()})
    first_state = states["irregular_15_10_5"]
    if not torch.equal(first_state["init_noise"], states["uniform_30"]["init_noise"]):
        raise RuntimeError("Archived schedules do not share x_T")
    dps_root = Path(args.dps_root).resolve()
    diffusion = yaml.safe_load((dps_root / "configs/diffusion_config.yaml").read_text())
    conditioning_config = yaml.safe_load((dps_root / "configs/gaussian_deblur_config.yaml").read_text())["conditioning"]
    validate_dps_config(diffusion, conditioning_config)
    model = load_diffusion_model("imagenet", args.device)
    model.model.eval()
    if not all(parameter.requires_grad for parameter in model.model.parameters()):
        raise RuntimeError("Input-gradient guidance needs model parameters' requires_grad intact; no optimizer is used")
    dps, reference_metadata = load_dps_reference(dps_root)
    condition_module = importlib.import_module(dps.__package__ + ".condition_methods")
    reference_metadata["source_sha256"][condition_module.__file__] = file_sha256(condition_module.__file__)
    operator = get_operator(task, device=args.device, **original["task_configs"][task])
    condition = condition_module.get_conditioning_method(
        "ps", operator=PureOperatorAdapter(operator),
        noiser=types.SimpleNamespace(__name__="gaussian", sigma=operator.noise.sigma), scale=0.3,
    )
    measurement_path = Path(saved["source"]) / task / "measurement.pt"
    y = torch.load(measurement_path, map_location=args.device, weights_only=True)
    gt = load_image_as_tensor(run_args["image"]).to(args.device)
    init_noise = first_state["init_noise"].to(args.device)
    del states, first_state
    if tuple(y.shape) != tuple(gt.shape) or tuple(init_noise.shape) != tuple(gt.shape):
        raise RuntimeError("Unexpected saved image/observation/x_T shapes")
    if not torch.isfinite(y).all() or not torch.isfinite(init_noise).all():
        raise RuntimeError("Non-finite source inputs")
    seed = int(run_args["seed"])
    set_seed(seed)
    regenerated = torch.randn(tuple(init_noise.shape), device=args.device, dtype=init_noise.dtype)
    if not torch.equal(regenerated, init_noise):
        raise RuntimeError("Source seed does not reproduce x_T")
    del regenerated
    start_rng = rng_state(args.device)
    print("\n=== Paired DPS 30-step grid x variance audit ===", flush=True)
    print("Four runs only; scale=0.3; no ZAPS retraining/no new y/no noise scaling.", flush=True)
    print("Fingerprinting checkpoint and locating matching completed DPS-1000 record...", flush=True)
    expected = {
        "source_trace": str(trace_dir), "task": task, "seed": seed,
        "measurement_sha256": file_sha256(measurement_path),
        "image_sha256": file_sha256(run_args["image"]),
        "checkpoint_sha256": file_sha256(model.ckpt_path),
        "operator_config": original["task_configs"][task],
        "operator_source_sha256": file_sha256(PROJECTS_ROOT / "modules/degradations.py"),
        "reference": reference_metadata,
    }
    baseline_dir, baseline = find_baseline(trace_dir, args.dps_baseline_dir, expected)
    print(f"Reusing DPS-1000: {baseline_dir}; PSNR={baseline['dps']['psnr']:.4f}", flush=True)
    output_dir = trace_dir / ("dps_lowstep_variance_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    record = {
        "experiment": "DPS_30step_schedule_x_variance", "status": "running",
        "arguments": vars(args), "git_revision": revision, **expected,
        "baseline_dir": str(baseline_dir), "DPS_1000_archived": baseline["dps"],
        "ZAPS_archived": baseline["zaps_archived"], "variants": variants, "results": [],
        "semantics": "same y/H/x_T; original DPS loop/PS; scale=0.3; no optimizer; source original-time grids",
        "pairing": "reset post-x_T CPU/CUDA RNG per arm; same draw indices, not same physical-time draws across grids",
        "limits": "DPS30 fixed vs ZAPS30 fixed compares guidance/optimization package, NOT Jacobian alone; "
                  "DPS1000 vs DPS30 also changes posterior timestep spacing and noise path",
    }

    def save_record():
        (output_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                        allow_nan=False), encoding="utf-8")

    first_end_rng = None
    save_record()
    try:
        for variant in variants:
            grid, variance, schedule = variant["grid"], variant["variance"], variant["schedule"]
            name = schedule + "_" + variance
            sampler = dps.DDPM(
                use_timesteps=grid, betas=dps.get_named_beta_schedule("linear", 1000),
                model_mean_type="epsilon", model_var_type=variance,
                dynamic_threshold=False, clip_denoised=True, rescale_timesteps=False,
            )
            if sampler.timestep_map != grid:
                raise RuntimeError("Official respacing does not match the archived ZAPS grid")
            calls = []
            expected_times = list(reversed(grid))

            def counted_model(x, t):
                original_t = expected_times[len(calls)]
                if not torch.equal(t, torch.full_like(t, original_t)):
                    raise RuntimeError("UNet received a compact index rather than the expected original time")
                if not calls and not torch.equal(x.detach(), init_noise):
                    raise RuntimeError("DPS arm does not share archived x_T")
                calls.append(original_t)
                return model.model(x, t)

            print(f"\n--- {schedule} | {variance} | 30 DPS steps ---", flush=True)
            torch.set_rng_state(start_rng["cpu"])
            if start_rng["cuda"]:
                torch.cuda.set_rng_state_all(start_rng["cuda"])
            started = time.time()
            with torch.enable_grad():
                reconstruction = sampler.p_sample_loop(
                    model=counted_model, x_start=init_noise.detach().clone(), measurement=y,
                    measurement_cond_fn=condition.conditioning, record=False, save_root=str(output_dir),
                ).detach()
            elapsed = time.time() - started
            end_rng = rng_state(args.device)  # before metrics may initialize LPIPS and consume CPU RNG
            if first_end_rng is None:
                first_end_rng = end_rng
            paired_rng = same_rng(first_end_rng, end_rng)
            if calls != expected_times or not paired_rng or not torch.isfinite(reconstruction).all():
                raise RuntimeError("DPS time-map/noise-consumption/finite-output gate failed")
            if any(parameter.grad is not None for parameter in model.model.parameters()):
                raise RuntimeError("Unexpected accumulated model parameter gradients")
            torch.save(reconstruction.cpu(), output_dir / f"{name}.pt")
            tensor_to_image(reconstruction.cpu().squeeze(0), denormalize=True).save(output_dir / f"{name}.png")
            metrics = compute_all_metrics(reconstruction, gt)
            result = {
                "schedule": schedule, "variance": variance, **metrics,
                "float_psnr": tensor_psnr(gt, reconstruction), "nfe": len(calls), "seconds": elapsed,
                "delta_psnr_vs_DPS1000": metrics["psnr"] - baseline["dps"]["psnr"],
                "residual_norm": float((y - operator.H(reconstruction)).norm()),
                "visited_original_times": calls, "paired_rng_end_equal": paired_rng,
            }
            record["results"].append(result)
            save_record()
            print(f"RESULT PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
                  f"LPIPS={metrics['lpips']:.4f} NFE={len(calls)}", flush=True)
            del reconstruction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        record["status"] = "complete"
        record["new_nfe"] = sum(r["nfe"] for r in record["results"])
        save_record()
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        save_record()
        raise
    print("\n=== DPS 30-step schedule x variance summary ===", flush=True)
    print(f"{'schedule':>20} {'variance':>13} {'PSNR':>9} {'d1000':>9} {'SSIM':>8} {'LPIPS':>8} {'NFE':>5}", flush=True)
    for row in record["results"]:
        print(f"{row['schedule']:>20} {row['variance']:>13} {row['psnr']:9.4f} "
              f"{row['delta_psnr_vs_DPS1000']:+9.4f} {row['ssim']:8.4f} {row['lpips']:8.4f} {row['nfe']:5d}", flush=True)
    print("\n=== Paired variance effect: learned_range minus fixed_small ===", flush=True)
    print(f"{'schedule':>20} {'dPSNR':>9} {'dSSIM':>9} {'dLPIPS':>9}", flush=True)
    for schedule in saved["results"]:
        pair = {r["variance"]: r for r in record["results"] if r["schedule"] == schedule}
        learned, fixed = pair["learned_range"], pair["fixed_small"]
        print(f"{schedule:>20} {learned['psnr']-fixed['psnr']:+9.4f} "
              f"{learned['ssim']-fixed['ssim']:+9.4f} {learned['lpips']-fixed['lpips']:+9.4f}", flush=True)
    print("\n=== DPS30 fixed versus archived ZAPS (same grid/fixed variance) ===", flush=True)
    print(f"{'schedule':>20} {'DPS30':>9} {'ZAPS':>9} {'DPS-ZAPS':>10}", flush=True)
    for row in record["results"]:
        if row["variance"] == "fixed_small":
            source_arm = record["ZAPS_archived"][row["schedule"]]
            zaps_psnr = source_arm["metrics"]["psnr"]
            label = "" if not source_arm["config"]["use_learned_var"] else " (ZAPS variance differs!)"
            print(f"{row['schedule']:>20} {row['psnr']:9.4f} {zaps_psnr:9.4f} "
                  f"{row['psnr']-zaps_psnr:+10.4f}{label}", flush=True)
    print("DPS30 is one 30-step trajectory; ZAPS is 30 steps x 10 optimization epochs=300 NFE.", flush=True)
    print("Do not attribute DPS/ZAPS differences to the approximate Jacobian alone.", flush=True)
    print(f"New NFE: {record['new_nfe']}; no DPS1000/ZAPS reruns. Records: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
