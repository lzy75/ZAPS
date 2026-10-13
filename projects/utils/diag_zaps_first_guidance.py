"""One frozen-path intervention: match only the t=999 guidance norm.

Use a completed archived VJP audit to determine ONE factor, without GT tuning.
No optimizer, core edits, mask, variance/noise changes or factor sweep.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))
ARM = "irregular_15_10_5"
IDENTITIES = ("checkpoint_sha256", "measurement_sha256", "operator_source_sha256",
              "core_sha256", "image_sha256")


def norm_calibration(guidance):
    rows = [r for r in guidance["results"][ARM]["rows"] if r["t"] == 999]
    if len(rows) != 1:
        raise ValueError("Need exactly one audited t=999 VJP; no nearest-time mapping")
    row = rows[0]
    exact, approximate, ratio = row["exact_norm"], row["raw_norm"], row["raw_norm_ratio"]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 1e-12
               for x in (exact, approximate, ratio)):
        raise ValueError("Invalid/zero archived gradient norms")
    if not math.isclose(approximate / exact, ratio, rel_tol=2e-5, abs_tol=1e-8):
        raise ValueError("Archived norm ratio is internally inconsistent")
    return {"factor": exact / approximate, "exact_norm": exact,
            "approximate_norm": approximate, "probe_time": 999,
            "objective": "0.5*sum((y-H(clamp(Tweedie(x))))^2)",
            "source": "completed same-state VJP audit, no GT selection"}


def validate_parent(parent, guidance, guidance_dir):
    if (parent.get("status") != "complete"
            or parent.get("experiment") != "archived_terminal_zeta_direction_audit"
            or parent.get("schedule") != ARM or parent.get("retraining") is not False
            or parent.get("optimizer_steps") != 0 or parent.get("new_nfe") != 600):
        raise ValueError("Need the completed agreed terminal-zeta audit")
    if Path(parent.get("source_guidance", "")).resolve() != Path(guidance_dir).resolve():
        raise ValueError("Parameter/VJP audit source paths differ")
    if parent.get("identity") != {k: guidance[k] for k in IDENTITIES}:
        raise ValueError("Parameter/VJP audit identities differ")
    gates = parent.get("gates", {})
    required = [gates.get("autograd_vs_no_grad", {}), gates.get("archived_loss", {})]
    required.extend(gates.get("archive_fingerprints", {}).values())
    if len(required) != 4 or not all(g.get("passed") is True for g in required):
        raise ValueError("Parent replay gates are missing/failed")


def select_parent(trace_dir, guidance, guidance_dir, explicit=None):
    if explicit:
        path = Path(explicit).resolve() / "run.json"
        parent = json.loads(path.read_text(encoding="utf-8"))
        validate_parent(parent, guidance, guidance_dir)
        return path, parent
    # Latest compatible COMPLETED record only; never use an unfinished run.
    for path in sorted(Path(trace_dir).glob("parameter_gradient_*/run.json"), reverse=True):
        try:
            parent = json.loads(path.read_text(encoding="utf-8"))
            validate_parent(parent, guidance, guidance_dir)
        except (ValueError, KeyError, TypeError):
            continue
        return path, parent
    raise ValueError("No compatible completed parameter audit; provide --parameter-dir")


@contextmanager
def first_guidance_scope(zaps, factor, expected_norm, observation):
    """Scale only the first adjoint vector; linear B then scales its correction.

    No zeta/D mutation, score call, extra RNG draw or posterior intervention.
    A factor-one path returns the original tensor without extra arithmetic.
    """
    import torch
    from utils.diag_zaps_guidance_clipping import approximate_vjp
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Calibration factor must be finite and positive")
    original = zaps.A.transpose
    visited = []
    grid = zaps.tau.flip(0).tolist()
    if not grid or grid[0] != 999:
        raise ValueError("First-step intervention requires t=999")

    def transpose(residual):
        index = len(grid) - 1 - len(visited)
        if index < 0:
            raise RuntimeError("Unexpected extra likelihood transpose")
        t = grid[len(visited)]
        vector = original(residual)
        visited.append(t)
        if len(visited) == 1:
            with torch.no_grad():
                direction = approximate_vjp(zaps, vector, zaps.dm.alphas_cumprod[t], index)
                norm = float(direction.double().norm())
            if not math.isclose(norm, expected_norm, rel_tol=2e-5, abs_tol=1e-6):
                raise RuntimeError("Live first guidance norm differs from audited same-state norm")
            observation.update(t=t, raw_norm=norm, factor=factor,
                               effective_norm=norm * factor)
            return vector if factor == 1 else vector * factor
        return vector

    with patch.object(zaps.A, "transpose", side_effect=transpose):
        yield
    if visited != grid:
        raise RuntimeError("Likelihood traversal changed")
    observation["visited"] = visited
    observation["scaled_times"] = [] if factor == 1 else [999]


def interpret(baseline, calibrated, first_rows):
    """Pre-registered engineering thresholds, not benchmark success criteria."""
    delta = {k: calibrated[k] - baseline[k] for k in ("psnr", "ssim", "lpips")}
    left, right = first_rows
    relief = (right["input_x_rms"] <= .5 * left["input_x_rms"]
              and right["raw_x0_clip_fraction"] <= left["raw_x0_clip_fraction"] + 2e-5)
    quality = delta["psnr"] >= .2 and delta["ssim"] >= .005 and delta["lpips"] <= -.005
    if relief and quality:
        result = "SUPPORTS_FIRST_STEP_AMPLITUDE_SENSITIVITY"
    elif relief and delta["psnr"] <= .05 and delta["ssim"] <= .001 and delta["lpips"] >= -.001:
        result = "EARLY_RELIEF_WITHOUT_FINAL_QUALITY_SUPPORT"
    else:
        result = "MIXED_OR_INCONCLUSIVE"
    return {"classification": result, "early_relief": relief, "metric_deltas": delta,
            "scope": "single-image frozen-path sensitivity; not a paper fix/state gain"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-dir", required=True)
    parser.add_argument("--parameter-dir", default=None,
                        help="Optional explicit completed parent; otherwise latest compatible completed record")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    args = parser.parse_args()
    if not str(args.device).startswith("cuda"):
        parser.error("Full-model audit requires the archived CUDA ImageNet path")
    from utils.diag_zaps_parameter_gradient import (
        validate_protocol, PROBE_TIMES, DELTAS, evaluate_replay, write_csv,
    )
    guidance_dir = Path(args.guidance_dir).resolve()
    guidance = json.loads((guidance_dir / "audit.json").read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"]).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source_dir = Path(saved["source"]).resolve()
    source = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
    validate_protocol(guidance, saved, source, PROBE_TIMES, DELTAS)
    calibration = norm_calibration(guidance)
    parent_path, parent = select_parent(trace_dir, guidance, guidance_dir, args.parameter_dir)
    arm = saved["results"][ARM]
    if parent.get("config") != arm["config"]:
        raise ValueError("Parent parameter audit configuration differs")
    output_dir = trace_dir / ("first_guidance_" + datetime.now(
        ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    card_path = PROJECTS_ROOT / "experiment_cards/R_first_guidance.md"
    card = card_path.read_text(encoding="utf-8")
    record = {"experiment": "first_guidance_norm_frozen_path", "status": "running",
              "arguments": vars(args), "resolved_parameter_run": str(parent_path),
              "source_guidance": str(guidance_dir), "source_trace": str(trace_dir),
              "config": arm["config"], "calibration": calibration, "experiment_card": card,
              "new_nfe": 0, "planned_nfe": 60, "source_method_nfe": 300,
              "optimizer_steps": 0, "retraining": False, "gates": {}, "calls": [], "results": [],
              "limits": "Frozen pre-update epoch-10 path; no author parity or state-aware claim",
              "gt_usage": "evaluation only, not factor/output/epoch selection",
              "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             capture_output=True, text=True, check=True).stdout.strip()}

    def save():
        (output_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False,
            indent=2, allow_nan=False), encoding="utf-8")
        write_csv(output_dir / "summary.csv", record["results"])
        write_csv(output_dir / "trace.csv", [dict(policy=c["policy"], **row)
                                             for c in record["calls"] for row in c["rows"]])

    (output_dir / "experiment_card.md").write_text(card, encoding="utf-8")
    save()
    print("=== First-guidance single-factor frozen replay ===", flush=True)
    print("Goal: can reducing ONLY first guidance to the audited exact norm improve the full path?", flush=True)
    print(f"Parent: {parent_path}\nFactor={calibration['factor']:.9g}; new diagnostic NFE=60", flush=True)
    print("No training/mask/noise/variance/LR/core edits; GT only for reporting.", flush=True)
    print(f"Records: {output_dir}", flush=True)
    try:
        import torch
        from configs.config import IMG_SIZE, METRICS_CONFIG
        from modules.main_single import load_diffusion_model, load_image_as_tensor
        from modules.degradations import get_operator
        from modules.zaps_algorithm import ZAPS
        from utils.diag_dps_same_observation import file_sha256
        from utils.diag_zaps_guidance_clipping import archive_fingerprint_check, saved_cuda_selection
        from utils.diag_zaps_sampler_parity import same_rng
        from utils.metrics import compute_all_metrics

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        measurement = source_dir / "gaussian_deblur/measurement.pt"
        snapshot = trace_dir / f"{ARM}_last_unroll_state.pt"
        paths = {"measurement_sha256": measurement, "core_sha256": PROJECTS_ROOT / "modules/zaps_algorithm.py",
                 "operator_source_sha256": PROJECTS_ROOT / "modules/degradations.py",
                 "image_sha256": Path(source["arguments"]["image"])}
        for key, path in paths.items():
            if file_sha256(path) != guidance[key]:
                raise RuntimeError(f"Archived identity changed: {key}")
        if file_sha256(snapshot) != parent["snapshot_sha256"]:
            raise RuntimeError("Snapshot differs from completed parameter audit")
        print("Loading original model; checking local checkpoint hash (no download)...", flush=True)
        model = load_diffusion_model("imagenet", args.device)
        model.model.eval()
        if file_sha256(model.ckpt_path) != guidance["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint identity changed")
        if model.num_steps != 1000 or any(p.dtype != torch.float32 for p in model.model.parameters()):
            raise RuntimeError("Keep archived fp32/1000-time model")
        state = torch.load(snapshot, map_location="cpu", weights_only=True)
        grid = state["tau"].tolist()
        if grid != sorted(r["t"] for r in arm["rows"]) or len(grid) != 30 or grid[-1] != 999:
            raise RuntimeError("Snapshot grid differs")
        if not torch.equal(state["zeta"], torch.tensor(arm["epoch_used_zeta_ascending"], dtype=state["zeta"].dtype)):
            raise RuntimeError("Snapshot is not the archived PRE-update state")
        for key, shape in (("zeta", (30,)), ("D", (30, 3, 256, 256)), ("init_noise", (1, 3, 256, 256))):
            if tuple(state[key].shape) != shape or state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                raise RuntimeError(f"Snapshot shape/dtype/value mismatch: {key}")
        y = torch.load(measurement, map_location=args.device, weights_only=True)
        if tuple(y.shape) != (1, 3, 256, 256) or y.dtype != torch.float32 or not torch.isfinite(y).all():
            raise RuntimeError("Observation shape/dtype/value mismatch")
        operator = get_operator("gaussian_deblur", device=args.device, **source["task_configs"]["gaussian_deblur"])
        zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **arm["config"])
        initial = state["init_noise"].to(args.device)
        mapping = saved_cuda_selection(state, saved["arguments"].get("device"), args.saved_cuda_index)
        record.update(identity={k: guidance[k] for k in IDENTITIES}, snapshot_sha256=parent["snapshot_sha256"],
                      parent_run_sha256=file_sha256(parent_path), source_vjp_sha256=file_sha256(guidance_dir / "audit.json"),
                      rng_mapping=mapping, runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "visible_cuda_count": torch.cuda.device_count()})
        save()
        print("[1/2] Original frozen path, 30 NFE...", flush=True)
        base_output, base, base_rng = evaluate_replay(zaps, y, state, initial, device=args.device,
            saved_index=mapping["saved_sampling_index"], accounting=record)
        base["policy"] = "raw_baseline"
        record["calls"].append(base)
        fingerprint = archive_fingerprint_check(base["rows"], arm["rows"])
        loss_gate = abs(base["loss32"] - arm["archived_loss_history"][-1]) <= 1e-8 + 2e-5 * abs(base["loss32"])
        first_norm = guidance["results"][ARM]["rows"][0]
        # First residual/input identity was checked above; same snapshot hashes
        # pin the audited raw norm used by the intervention, including D.
        if first_norm["t"] != 999:
            raise RuntimeError("Archived VJP order changed")
        record["gates"].update(archive_fingerprints=fingerprint, archived_loss={"passed": loss_gate})
        save()
        if not fingerprint["passed"] or not loss_gate:
            raise RuntimeError("Baseline archive gate failed; no intervention interpretation")
        print("Baseline gates passed. [2/2] Norm-calibrated FIRST step, remaining 29 unchanged...", flush=True)
        intervention = {}
        with first_guidance_scope(zaps, calibration["factor"], calibration["approximate_norm"], intervention):
            test_output, test, test_rng = evaluate_replay(zaps, y, state, initial, device=args.device,
                saved_index=mapping["saved_sampling_index"], accounting=record)
        test["policy"] = "first_norm_matched"
        record["calls"].append(test)
        rng_equal = same_rng(base_rng, test_rng)
        first_equal = base["rows"][0] == test["rows"][0]
        record["intervention"] = intervention
        record["gates"].update(pairing={"passed": rng_equal and first_equal,
            "rng_end_equal": rng_equal, "pre_first_guidance_fingerprint_equal": first_equal})
        save()
        if not rng_equal or not first_equal or record["new_nfe"] != 60:
            raise RuntimeError("Pairing/budget gate failed")
        gt = load_image_as_tensor(source["arguments"]["image"]).cpu()
        # Two predefined outputs only; GT is loaded AFTER intervention and gates.
        for output, call in ((base_output, base), (test_output, test)):
            metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
            row = dict(policy=call["policy"], **metrics, loss=call["loss64"], residual=call["residual_norm"],
                       next_rms=call["rows"][1]["input_x_rms"], next_clip_fraction=call["rows"][1]["raw_x0_clip_fraction"],
                       diagnostic_nfe=call["nfe"], seconds=call["seconds"])
            record["results"].append(row)
            torch.save(output, output_dir / (call["policy"] + ".pt"))
        record["decision"] = interpret(record["results"][0], record["results"][1],
                                       (base["rows"][1], test["rows"][1]))
        record["status"] = "complete"
        save()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save()
        raise
    print("\n=== Paired first-guidance summary ===", flush=True)
    print(f"{'policy':>20} {'PSNR':>9} {'dPSNR':>9} {'SSIM':>8} {'dSSIM':>8} {'LPIPS':>8} {'dLPIPS':>9} {'nextRMS':>9} {'clip%':>8}", flush=True)
    b = record["results"][0]
    for row in record["results"]:
        print(f"{row['policy']:>20} {row['psnr']:9.4f} {row['psnr']-b['psnr']:+9.4f} "
              f"{row['ssim']:8.4f} {row['ssim']-b['ssim']:+8.4f} {row['lpips']:8.4f} {row['lpips']-b['lpips']:+9.4f} "
              f"{row['next_rms']:9.4f} {100*row['next_clip_fraction']:8.3f}", flush=True)
    print("Decision:", json.dumps(record["decision"], ensure_ascii=False), flush=True)
    print(f"Diagnostic NFE={record['new_nfe']}; records: {output_dir}", flush=True)
    print("Sensitivity only; not retrained ZAPS/paper reproduction/state-aware gain.", flush=True)


if __name__ == "__main__":
    main()
