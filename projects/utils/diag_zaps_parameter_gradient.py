"""Audit three zeta derivatives of an archived ImageNet Gaussian trajectory.

Restore epoch-10 PRE-update parameters and every sampling RNG before each call.
Compare the current stop-gradient unroll derivative with central differences of
the actual terminal measurement MSE. No optimizer, new observation, GT-based
selection, mask change, variance change, or replacement reconstruction.
"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))

PROBE_TIMES = (999, 333, 143)
DELTAS = (1e-4, 3e-4, 1e-3)
LOSS_FLOOR = 1e-10
SIGNAL_MULTIPLIER = 10.0
STABILITY_RTOL = 0.20
AGREEMENT_RTOL = 0.20


def parameter_indices(grid, requested):
    if grid != sorted(set(grid)) or not grid or grid[0] != 0 or grid[-1] != 999:
        raise ValueError("Need a unique ascending original-time grid including 0 and 999")
    if len(requested) != len(set(requested)) or any(t not in grid for t in requested):
        raise ValueError("Probe times must be unique exact grid members; no silent nearest mapping")
    return [grid.index(t) for t in requested]


def validate_protocol(guidance, saved, source, requested, deltas):
    from utils.diag_zaps_clamp_ablation import validate_source
    validate_source(guidance, saved)
    if (guidance.get("task") != "gaussian_deblur"
            or saved.get("arguments", {}).get("task") != "gaussian_deblur"
            or source.get("arguments", {}).get("dataset") != "imagenet"):
        raise ValueError("Only the audited ImageNet Gaussian path is in scope")
    if tuple(requested) != PROBE_TIMES or tuple(deltas) != DELTAS:
        raise ValueError("This card fixes times=999,333,143 and deltas=.0001,.0003,.001")
    arm = saved["results"]["irregular_15_10_5"]
    if not arm.get("gates", {}).get("passed"):
        raise ValueError("Archived last-opt replay gate did not pass")
    config = arm["config"]
    for key, expected in (("wave", "db4"), ("level", 3), ("zeta_init", .2), ("d_init", .2)):
        if config.get(key) != expected:
            raise ValueError(f"Keep archived {key}={expected!r}")
    if not arm.get("rows") or len(arm.get("archived_loss_history", [])) != 10:
        raise ValueError("Need full archived fingerprints and ten-epoch loss history")
    for key in ("checkpoint_sha256", "measurement_sha256", "operator_source_sha256",
                "core_sha256", "image_sha256"):
        value = guidance.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Missing source identity: {key}")


def classify_probe(autograd, pairs, loss_noise_floor):
    """Conservative, pre-registered scalar direction/scale classification.

    A stable disagreement concerns this approximation, not author-code parity.
    All resolved scales must agree: never choose a convenient h or a PSNR winner.
    """
    if not math.isfinite(autograd) or not math.isfinite(loss_noise_floor) or loss_noise_floor < 0:
        raise ValueError("Non-finite gradient or invalid loss-noise floor")
    estimates = []
    for pair in sorted(pairs, key=lambda p: p["requested_delta"]):
        span = pair["plus_value"] - pair["minus_value"]
        difference = pair["plus_loss64"] - pair["minus_loss64"]
        if span <= 0 or not all(math.isfinite(x) for x in (span, difference)):
            raise ValueError("Unrepresentable/non-finite central difference")
        estimates.append({
            "delta": pair["requested_delta"], "effective_span": span,
            "g_fd": difference / span,
            "resolved": abs(difference) > SIGNAL_MULTIPLIER * loss_noise_floor,
        })
    resolved = [p for p in estimates if p["resolved"]]
    result = {
        "g_autograd": autograd, "g_finite_difference": None, "relative_error": None,
        "loss_noise_floor": loss_noise_floor, "stable_scales": len(resolved),
        "fd_relative_spread": None, "estimates": estimates,
        "classification": "INCONCLUSIVE_SMALL_SIGNAL",
    }
    if len(resolved) < 2:
        return result
    slopes = [p["g_fd"] for p in resolved]
    median = statistics.median(slopes)
    spread = (max(slopes) - min(slopes)) / max(abs(x) for x in slopes)
    result.update(g_finite_difference=median, fd_relative_spread=spread)
    if any(x * median <= 0 for x in slopes) or spread > STABILITY_RTOL:
        result["classification"] = "INCONCLUSIVE_SCALE_DEPENDENT"
        return result
    gradient_floor = max(SIGNAL_MULTIPLIER * loss_noise_floor / p["effective_span"]
                         for p in resolved)
    if abs(autograd) <= gradient_floor:
        result["classification"] = "AUTOGRAD_NEAR_ZERO_WITH_FD_SIGNAL"
        return result
    relative = abs(autograd - median) / max(abs(autograd), abs(median))
    result["relative_error"] = relative
    if autograd * median < 0:
        result["classification"] = "DIRECTION_CONFLICT"
    elif relative <= AGREEMENT_RTOL:
        result["classification"] = "LOCAL_AGREEMENT"
    else:
        result["classification"] = "SCALE_MISMATCH"
    return result


def restore_parameters(zaps, state, device):
    import torch
    zaps.tau = state["tau"].to(device)
    with torch.no_grad():
        zaps.zeta.copy_(state["zeta"].to(device))
        zaps.D.copy_(state["D"].to(device))
    zaps.zero_grad(set_to_none=True)


def evaluate_replay(zaps, y, state, initial, *, device, saved_index,
                    gradient=False, probe=None, accounting=None):
    """One real core call; observer-only scope and rollback also on exceptions."""
    import torch
    import torch.nn.functional as F
    from utils.diag_zaps_clamp_ablation import clamp_guidance_scope
    from utils.diag_zaps_guidance_clipping import restore_sampling_rng
    from utils.diag_zaps_sampler_parity import rng_state

    restore_parameters(zaps, state, device)
    zaps._clamp_audit_y = y.detach()
    if probe is not None:
        index, delta = probe
        with torch.no_grad():
            zaps.zeta[index].add_(delta)
        if float(zaps.zeta[index].detach()) <= 1e-6:
            restore_parameters(zaps, state, device)
            raise ValueError("Perturbation crosses the positive-zeta constraint")
    effective_value = float(zaps.zeta[probe[0]].detach()) if probe is not None else None
    if str(device).startswith("cuda"):
        restore_sampling_rng(state, device, saved_index)
    else:  # tiny CPU tests only; the production CLI rejects non-CUDA runs
        torch.set_rng_state(state["cpu_rng"])
    counter, rows, visited = [0], [], []

    def before_model(_module, inputs):
        visited.append(int(inputs[1].detach().flatten()[0]))
        if accounting is not None:
            accounting["new_nfe"] += 1

    handle = zaps.dm.model.register_forward_pre_hook(before_model)
    started = time.perf_counter()
    try:
        with torch.set_grad_enabled(gradient):
            with clamp_guidance_scope(zaps, False, rows):
                output = zaps._reverse_diffusion(
                    y, nfe_counter=counter, init_noise=initial, record_indicators=False,
                )
            prediction = zaps.A.H(output)
            loss32 = F.mse_loss(prediction, y)
            # Same MSE, but square/reduction in double for finite-difference
            # reporting; H/UNet/subtraction remain the archived fp32 operations.
            residual = (prediction - y).detach()
            loss64 = float(residual.double().square().mean())
            residual_norm = float(residual.double().norm())
            if not torch.isfinite(output).all() or not math.isfinite(loss64):
                raise RuntimeError("Non-finite terminal output or measurement loss")
            gradients = None
            if gradient:
                loss32.backward()  # no clipping, projection, Adam or weight update
                if zaps.zeta.grad is None or zaps.D.grad is None:
                    raise RuntimeError("Missing unroll parameter gradient")
                if not torch.isfinite(zaps.zeta.grad).all() or not torch.isfinite(zaps.D.grad).all():
                    raise RuntimeError("Non-finite unroll parameter gradient")
                gradients = {
                    "zeta": zaps.zeta.grad.detach().cpu().tolist(),
                    "zeta_norm": float(zaps.zeta.grad.double().norm()),
                    "D_norm": float(zaps.D.grad.double().norm()),
                }
        if any(p.grad is not None for p in zaps.dm.model.parameters()):
            raise RuntimeError("Score-model parameter gradient unexpectedly accumulated")
        expected = state["tau"].flip(0).tolist()
        if visited != expected or counter[0] != len(expected):
            raise RuntimeError("Model-call count or original-time traversal changed")
        end_rng = rng_state(device)
        data = {
            "nfe": counter[0], "visited": visited, "rows": rows,
            "loss32": float(loss32.detach()), "loss64": loss64,
            "residual_norm": residual_norm, "effective_value": effective_value,
            "gradients": gradients, "seconds": time.perf_counter() - started,
        }
        return output.detach().cpu(), data, end_rng
    finally:
        handle.remove()
        restore_parameters(zaps, state, device)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    parser.add_argument("--probe-times", nargs=3, type=int, default=list(PROBE_TIMES))
    parser.add_argument("--deltas", nargs=3, type=float, default=list(DELTAS))
    args = parser.parse_args()
    if not str(args.device).startswith("cuda"):
        parser.error("Production audit is restricted to the archived CUDA ImageNet path")

    guidance_dir = Path(args.guidance_dir).resolve()
    guidance = json.loads((guidance_dir / "audit.json").read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"]).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source_dir = Path(saved["source"]).resolve()
    source = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
    validate_protocol(guidance, saved, source, args.probe_times, args.deltas)
    name = "irregular_15_10_5"
    arm = saved["results"][name]
    card_path = PROJECTS_ROOT / "experiment_cards/R_parameter_gradient.md"
    card = card_path.read_text(encoding="utf-8")
    planned_nfe = 30 * (2 + 2 * len(args.probe_times) * len(args.deltas))
    output_dir = trace_dir / ("parameter_gradient_" + datetime.now(
        ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "experiment": "archived_terminal_zeta_direction_audit", "status": "running",
        "arguments": vars(args), "source_guidance": str(guidance_dir),
        "source_trace": str(trace_dir), "schedule": name, "config": arm["config"],
        "experiment_card": card, "new_nfe": 0, "planned_nfe": planned_nfe,
        "method_reference_nfe": 300, "retraining": False, "optimizer_steps": 0,
        "objective": "mean((H(raw terminal R(zeta,D;fixed RNG))-y)^2); no final output clamp",
        "numeric_loss": "fp32 H/residual, fp64 square and reduction; AD uses original fp32 MSE",
        "limits": "Three zeta coordinates only; not a D audit, exact author parity, PSNR gain, or root-cause proof",
        "ground_truth_usage": "baseline verification metrics only; never in gradient/probe selection",
        "thresholds": {"loss_floor": LOSS_FLOOR, "signal_multiplier": SIGNAL_MULTIPLIER,
                       "stability_rtol": STABILITY_RTOL, "agreement_rtol": AGREEMENT_RTOL},
        "gates": {}, "calls": [], "pairs": [], "summary": [],
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True, capture_output=True, check=True).stdout.strip(),
    }

    def save():
        (output_dir / "run.json").write_text(json.dumps(
            record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_csv(output_dir / "pairs.csv", record["pairs"])
        write_csv(output_dir / "summary.csv", record["summary"])

    (output_dir / "experiment_card.md").write_text(card, encoding="utf-8")
    save()
    print("=== Archived terminal-parameter gradient audit ===", flush=True)
    print("Goal: does the current unroll derivative predict actual terminal measurement-loss changes?", flush=True)
    print(f"Times={args.probe_times}; deltas={args.deltas}; planned diagnostic NFE={planned_nfe}", flush=True)
    print("No retraining/Adam/late-noise change/GT-selected output. Source baseline unchanged.", flush=True)
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
        from utils.diag_zaps_trace_audit import relative_error
        from utils.metrics import compute_all_metrics

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; full-model numerical audit not run")
        measurement_path = source_dir / "gaussian_deblur/measurement.pt"
        snapshot_path = trace_dir / f"{name}_last_unroll_state.pt"
        print("Checking source/core/H/image identities; loading model (no download)...", flush=True)
        identity_paths = {
            "measurement_sha256": measurement_path,
            "operator_source_sha256": PROJECTS_ROOT / "modules/degradations.py",
            "core_sha256": PROJECTS_ROOT / "modules/zaps_algorithm.py",
            "image_sha256": Path(source["arguments"]["image"]),
        }
        for key, path in identity_paths.items():
            if file_sha256(path) != guidance[key]:
                raise RuntimeError(f"Identity changed since completed audit: {key}")
        model = load_diffusion_model("imagenet", args.device)
        model.model.eval()
        print("Checking checkpoint SHA256 (local file read)...", flush=True)
        if file_sha256(model.ckpt_path) != guidance["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from the completed audit")
        if model.num_steps != 1000 or any(p.dtype != torch.float32 for p in model.model.parameters()):
            raise RuntimeError("Keep the audited fp32/1000-training-time ImageNet model")
        record["identity"] = {key: guidance[key] for key in (*identity_paths, "checkpoint_sha256")}
        record["snapshot_sha256"] = file_sha256(snapshot_path)
        record["runtime"] = {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_cuda_count": torch.cuda.device_count(),
        }
        state = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        grid = state["tau"].tolist()
        indices = parameter_indices(grid, args.probe_times)
        expected_grid = sorted(r["t"] for r in arm["rows"])
        if grid != expected_grid or len(grid) != 30:
            raise RuntimeError("Snapshot grid differs from original archived traversal")
        if not torch.equal(state["zeta"], torch.tensor(
                arm["epoch_used_zeta_ascending"], dtype=state["zeta"].dtype)):
            raise RuntimeError("Snapshot is not the archived pre-update zeta state")
        y = torch.load(measurement_path, map_location=args.device, weights_only=True)
        expected_image_shape = (1, 3, 256, 256)
        for key, shape in (("zeta", (30,)), ("D", (30, 3, 256, 256)),
                           ("init_noise", expected_image_shape)):
            if tuple(state[key].shape) != shape or state[key].dtype != torch.float32:
                raise RuntimeError(f"Snapshot shape/dtype mismatch: {key}")
            if not torch.isfinite(state[key]).all():
                raise RuntimeError(f"Non-finite snapshot: {key}")
        if tuple(y.shape) != expected_image_shape or y.dtype != torch.float32 or not torch.isfinite(y).all():
            raise RuntimeError("Unexpected Gaussian observation shape/dtype/value")
        operator = get_operator("gaussian_deblur", device=args.device,
                                **source["task_configs"]["gaussian_deblur"])
        zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **arm["config"])
        initial = state["init_noise"].to(args.device)
        mapping = saved_cuda_selection(state, saved["arguments"].get("device"), args.saved_cuda_index)
        record["rng_mapping"] = mapping
        record["parameter_indices"] = dict(zip(map(str, args.probe_times), indices))
        save()

        def call(label, gradient=False, probe=None):
            print(f"[{len(record['calls'])+1:02d}/20] {label} ...", flush=True)
            output, data, rng = evaluate_replay(
                zaps, y, state, initial, device=args.device,
                saved_index=mapping["saved_sampling_index"], gradient=gradient,
                probe=probe, accounting=record,
            )
            data["label"] = label
            record["calls"].append(data)
            save()
            print(f"  loss64={data['loss64']:.12g}; NFE={record['new_nfe']}/{planned_nfe}; "
                  f"seconds={data['seconds']:.1f}", flush=True)
            return output, data, rng

        first, base, rng_a = call("baseline_autograd", gradient=True)
        repeat, check, rng_b = call("baseline_no_grad")
        replay_error = relative_error(first, repeat)
        # AD-vs-no-grad is a cross-mode comparison, NOT an independent repeat
        # floor. Do not use this discrepancy to loosen its own acceptance limit.
        metadata_equal = same_rng(rng_a, rng_b) and base["visited"] == check["visited"]
        null_gate = {"passed": math.isfinite(replay_error) and replay_error <= 1e-7 and metadata_equal,
                     "output_relative_error": replay_error, "relative_limit": 1e-7,
                     "metadata_equal": metadata_equal,
                     "scope": "strict AD/no-grad value and RNG parity; not a measured repeat floor"}
        fingerprints = {
            "autograd": archive_fingerprint_check(base["rows"], arm["rows"]),
            "no_grad": archive_fingerprint_check(check["rows"], arm["rows"]),
        }
        archived_loss = arm["archived_loss_history"][-1]
        loss_gate = abs(base["loss32"] - archived_loss) <= 1e-8 + 2e-5 * abs(archived_loss)
        record["gates"] = {"autograd_vs_no_grad": null_gate, "archive_fingerprints": fingerprints,
                           "archived_loss": {"passed": loss_gate, "current": base["loss32"],
                                             "archived": archived_loss}}
        save()
        if not null_gate["passed"] or not loss_gate or not all(g["passed"] for g in fingerprints.values()):
            raise RuntimeError("Baseline/RNG/archive parity failed; no derivative interpretation")
        floor = max(LOSS_FLOOR, abs(base["loss64"] - check["loss64"]))
        record["loss_noise_floor"] = floor
        gt = load_image_as_tensor(source["arguments"]["image"]).cpu()
        record["baseline_metrics"] = compute_all_metrics(
            repeat, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        record["archived_metrics"] = arm["metrics"]
        save()
        print(f"Replay gates passed; baseline PSNR={record['baseline_metrics']['psnr']:.4f}; "
              f"loss noise floor={floor:.3g}", flush=True)
        del first, repeat, gt

        for t, index in zip(args.probe_times, indices):
            pairs = []
            for delta in args.deltas:
                plus, upper, end_plus = call(f"t={t} zeta[{index}] +{delta:g}", probe=(index, delta))
                minus, lower, end_minus = call(f"t={t} zeta[{index}] -{delta:g}", probe=(index, -delta))
                if not same_rng(rng_b, end_plus) or not same_rng(rng_b, end_minus):
                    raise RuntimeError("Perturbation changed sampling RNG consumption")
                pair = {
                    "probe_time": t, "parameter_index": index, "requested_delta": delta,
                    "zeta_value": float(state["zeta"][index]),
                    "plus_value": upper["effective_value"], "minus_value": lower["effective_value"],
                    "plus_loss64": upper["loss64"], "minus_loss64": lower["loss64"],
                    "plus_loss32": upper["loss32"], "minus_loss32": lower["loss32"],
                    "plus_residual": upper["residual_norm"], "minus_residual": lower["residual_norm"],
                }
                pairs.append(pair)
                record["pairs"].append(pair)
                save()
                del plus, minus
            summary = classify_probe(base["gradients"]["zeta"][index], pairs, floor)
            estimates = summary.pop("estimates")
            for pair, estimate in zip(pairs, estimates):
                pair.update(g_finite_difference=estimate["g_fd"], resolved=estimate["resolved"],
                            effective_span=estimate["effective_span"])
            record["summary"].append({"probe_time": t, "parameter_index": index, **summary})
            save()
            print(f"t={t}: {summary['classification']}; AD={summary['g_autograd']:.6g}; "
                  f"FD={summary['g_finite_difference']}", flush=True)
        if record["new_nfe"] != planned_nfe:
            raise RuntimeError("Unexpected diagnostic model-call budget")
        record["status"] = "complete"
        record["interpretation"] = "Read per-coordinate classification; completion is not gradient agreement"
        save()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save()
        raise

    print("\n=== Terminal zeta derivative summary ===", flush=True)
    print(f"{'t':>5} {'index':>6} {'AD':>13} {'FD':>13} {'rel.err':>10} {'classification':>36}", flush=True)
    for row in record["summary"]:
        fd = "n/a" if row["g_finite_difference"] is None else f"{row['g_finite_difference']:.6g}"
        error = "n/a" if row["relative_error"] is None else f"{row['relative_error']:.4f}"
        print(f"{row['probe_time']:5d} {row['parameter_index']:6d} {row['g_autograd']:13.6g} "
              f"{fd:>13} {error:>10} {row['classification']:>36}", flush=True)
    print("Stable conflicts question this unroll approximation; they do NOT prove a paper bug or a PSNR fix.", flush=True)
    print("Only three zeta directions checked; D and the full parameter gradient remain unaudited.", flush=True)
    print(f"New diagnostic NFE={record['new_nfe']}; records: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
