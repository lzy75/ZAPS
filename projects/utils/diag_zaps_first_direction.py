"""Replace only the first guidance direction, retaining its calibrated norm.

Reuse a completed first-guidance run; one input VJP plus one 30-step replay.
No optimizer, GT selection, direction sweep, variance/noise change or core edit.
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
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))


def validate_first_run(first, guidance, trace_dir):
    from utils.diag_zaps_first_guidance import norm_calibration, IDENTITIES
    if (first.get("status") != "complete" or first.get("experiment") != "first_guidance_norm_frozen_path"
            or first.get("retraining") is not False or first.get("optimizer_steps") != 0
            or first.get("new_nfe") != 60):
        raise ValueError("Need a completed first-guidance norm intervention, not training or a failed run")
    if Path(first.get("source_trace", "")).resolve() != Path(trace_dir).resolve():
        raise ValueError("First-guidance/source trace paths differ")
    if first.get("identity") != {k: guidance[k] for k in IDENTITIES}:
        raise ValueError("First-guidance/VJP identities differ")
    gates = first.get("gates", {})
    if not all(gates.get(k, {}).get("passed") is True
               for k in ("archive_fingerprints", "archived_loss", "pairing")):
        raise ValueError("First-guidance baseline or pairing gate failed/missing")
    factor = first.get("calibration", {}).get("factor")
    expected = norm_calibration(guidance)
    if (not isinstance(factor, (int, float)) or not math.isfinite(factor)
            or not math.isclose(factor, expected["factor"], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Parent did not use the agreed norm-derived factor")
    if [r.get("policy") for r in first.get("results", [])] != ["raw_baseline", "first_norm_matched"]:
        raise ValueError("Need the two predefined parent outputs")
    if [c.get("policy") for c in first.get("calls", [])] != ["raw_baseline", "first_norm_matched"]:
        raise ValueError("Need both complete parent trajectories")
    if (first.get("intervention", {}).get("scaled_times") != [999]
            or any(c.get("nfe") != 30 or len(c.get("rows", [])) != 30 for c in first["calls"])):
        raise ValueError("Parent changed more than the single agreed first step")


def interpret_direction(baseline, candidate):
    delta = {k: candidate[k] - baseline[k] for k in ("psnr", "ssim", "lpips")}
    if delta["psnr"] >= .2 and delta["ssim"] >= .005 and delta["lpips"] <= -.005:
        classification = "SUPPORTS_FIRST_STEP_DIRECTION_SENSITIVITY"
    elif delta["psnr"] <= .05 and delta["ssim"] <= .001 and delta["lpips"] >= -.001:
        classification = "NO_FINAL_QUALITY_SUPPORT_FOR_EXACT_DIRECTION"
    else:
        classification = "MIXED_OR_INCONCLUSIVE"
    return {"classification": classification, "metric_deltas_vs_norm_matched": delta,
            "scope": "single-image frozen-path sensitivity only; not a trained method/paper repair"}


@contextmanager
def first_direction_scope(zaps, target, expected_raw_norm, observation, enabled=True):
    """Intercept only first Hv so the ACTUAL unchanged core applies target.

    At t=999 set Hv=(sqrt(alpha)*target-v)/(1-alpha). Then the core computes
    (v+(1-alpha)*Hv)/sqrt(alpha)=target, without changing zeta, D or posterior.
    Both precision and norm are checked on that actual fp32 algebra. Subsequent
    adjoint/synthesis calls return original tensors unchanged. Disabled mode is
    a bitwise-null observer for unit tests, not an extra production trajectory.
    """
    import torch
    from utils.diag_zaps_sampler_parity import error_metrics
    original_transpose, original_synthesis = zaps.A.transpose, zaps.dwt.synthesis
    grid = zaps.tau.flip(0).tolist()
    if not grid or grid[0] != 999 or zaps.surrogate_score_jacobian:
        raise ValueError("Need original t=999 fixed-grid, surrogate-disabled core")
    holder = {"vector": None, "pending": False, "visited": [], "syntheses": []}

    def transpose(residual):
        if holder["pending"] or len(holder["visited"]) >= len(grid):
            raise RuntimeError("Unexpected likelihood transpose ordering")
        vector = original_transpose(residual)
        holder["vector"], holder["pending"] = vector, True
        holder["visited"].append(grid[len(holder["visited"])])
        return vector

    def synthesis(coefficients):
        if not holder["pending"]:
            raise RuntimeError("Unexpected synthesis outside the likelihood update")
        original = original_synthesis(coefficients)
        t = holder["visited"][-1]
        holder["pending"] = False
        holder["syntheses"].append(t)
        if len(holder["syntheses"]) != 1:
            return original
        ab = zaps.dm.alphas_cumprod[t]
        sqrt_ab, q = ab.sqrt().clamp(min=1e-8), 1-ab
        vector = holder["vector"]
        raw = (vector + q * original) / sqrt_ab
        norm = float(raw.double().norm())
        if not math.isclose(norm, expected_raw_norm, rel_tol=2e-5, abs_tol=1e-6):
            raise RuntimeError("First unscaled direction differs from the completed audit")
        if not enabled:
            observation.update(t=t, changed_times=[], raw_norm=norm)
            return original
        if (target.shape != raw.shape or target.dtype != raw.dtype or target.device != raw.device
                or not torch.isfinite(target).all() or float(q) <= 1e-8):
            raise RuntimeError("Invalid first direction target/coefficient")
        replacement = (sqrt_ab * target - vector) / q
        actual = (vector + q * replacement) / sqrt_ab
        algebra = error_metrics(actual, target, atol=2e-5, rtol=2e-5)
        index = len(grid)-1
        desired_correction = zaps.zeta[index] * target
        actual_correction = zaps.zeta[index] * actual
        desired_norm = float(desired_correction.double().norm())
        actual_norm = float(actual_correction.double().norm())
        norm_equal = desired_norm > 1e-12 and math.isclose(actual_norm, desired_norm, rel_tol=2e-5, abs_tol=1e-6)
        observation.update(t=t, changed_times=[999], raw_norm=norm, algebra_gate=algebra,
                           target_guided_norm=float(target.double().norm()),
                           applied_guided_norm=float(actual.double().norm()),
                           desired_correction_norm=desired_norm, applied_correction_norm=actual_norm,
                           correction_norm_gate={"passed": norm_equal, "rtol": 2e-5, "atol": 1e-6})
        if not algebra["passed"] or not norm_equal:
            raise RuntimeError("Actual first correction failed direction/norm algebra gate")
        return replacement

    with patch.object(zaps.A, "transpose", side_effect=transpose), patch.object(
            zaps.dwt, "synthesis", side_effect=synthesis):
        yield
    if holder["pending"] or holder["visited"] != grid or holder["syntheses"] != grid:
        raise RuntimeError("Likelihood traversal or synthesis count changed")
    observation["visited"] = holder["visited"]


@contextmanager
def model_rng_guard(model, device):
    """The fixed-noise pairing assumes eval UNet consumes no sampling RNG."""
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    pending, visited = [], []

    def before(module, inputs):
        pending.append(rng_state(device))

    def after(module, inputs, output):
        if len(pending) != 1 or not same_rng(pending.pop(), rng_state(device)):
            raise RuntimeError("Score evaluation consumed RNG; expected-noise replay is invalid")
        visited.append(True)

    first = model.register_forward_pre_hook(before)
    second = model.register_forward_hook(after)
    try:
        yield visited
    finally:
        first.remove()
        second.remove()


def first_vjp(zaps, y, state, captured, calibration, device, mapping, accounting):
    """ONE actual UNet call plus its input backward; never populate weight grads."""
    import torch
    from utils.diag_zaps_parameter_gradient import restore_parameters
    from utils.diag_zaps_guidance_clipping import restore_sampling_rng, approximate_vjp
    from utils.diag_zaps_sampler_parity import error_metrics, rng_state, same_rng
    restore_parameters(zaps, state, device)
    x = state["init_noise"].to(device).detach().clone().requires_grad_(True)
    if captured["t"] != 999 or captured["index"] != 29 or not torch.equal(x.detach().cpu(), captured["x"]):
        raise RuntimeError("Saved first VJP input differs from the frozen x_T")
    restore_sampling_rng(state, device, mapping["saved_sampling_index"])
    before_rng = rng_state(device)
    visited = []

    def count(module, inputs):
        accounting["new_nfe"] += 1
        visited.append(int(inputs[1].flatten()[0]))

    handle = zaps.dm.model.register_forward_pre_hook(count)
    started = time.perf_counter()
    try:
        with torch.enable_grad():
            epsilon = zaps.dm._predict_eps(x, torch.full((x.shape[0],), 999, device=device, dtype=torch.long))
            epsilon_gate = error_metrics(epsilon, captured["epsilon"].to(device), 2e-5, 2e-5)
            if not epsilon_gate["passed"]:
                raise RuntimeError("First VJP epsilon differs from archived same input")
            ab = zaps.dm.alphas_cumprod[999]
            raw = (x - (1-ab).sqrt() * epsilon) / ab.sqrt().clamp(min=1e-8)
            clipped = raw.clamp(-1, 1)
            residual = y-zaps.A.H(clipped)
            objective = .5 * residual.square().sum()
            exact = -torch.autograd.grad(objective, x)[0].detach()
        with torch.no_grad():
            # The parent core evaluated epsilon under no_grad. Reconstruct its
            # reference value with the saved no-grad output, already gated
            # against this single differentiated forward. This avoids treating
            # fp32 grad-mode rounding as a change of the cached reference path.
            reference_raw = (x.detach()-(1-ab).sqrt()*captured["epsilon"].to(device)) / ab.sqrt().clamp(min=1e-8)
            reference_clipped = reference_raw.clamp(-1, 1)
            reference_residual = y-zaps.A.H(reference_clipped)
            vector = zaps.A.transpose(reference_residual)
            approximate = approximate_vjp(zaps, vector, ab, 29)
            # Reproduce previous scaling BEFORE B, exactly as the parent core.
            matched = approximate_vjp(zaps, vector * calibration["factor"], ab, 29)
            exact_norm, raw_norm = float(exact.double().norm()), float(approximate.double().norm())
            matched_norm = float(matched.double().norm())
            if not all(math.isfinite(v) and v > 1e-12 for v in (exact_norm, raw_norm, matched_norm)):
                raise RuntimeError("Non-finite/zero first direction")
            if (not math.isclose(exact_norm, calibration["exact_norm"], rel_tol=2e-5, abs_tol=1e-6)
                    or not math.isclose(raw_norm, calibration["approximate_norm"], rel_tol=2e-5, abs_tol=1e-6)):
                raise RuntimeError("Live exact/approximate norms differ from completed VJP")
            target = exact * (matched_norm / exact_norm)
            cosine = float(torch.dot(matched.double().flatten(), exact.double().flatten()) /
                           (matched.double().norm()*exact.double().norm()))
            correction_norm = float((zaps.zeta[29] * matched).double().norm())
            data = {"nfe": 1, "epsilon_gate": epsilon_gate, "exact_norm": exact_norm,
                    "raw_norm": raw_norm, "matched_guided_norm": matched_norm,
                    "reference_correction_norm": correction_norm, "cosine_approx_exact": cosine,
                    "seconds": time.perf_counter()-started,
                    "reference_epsilon": "archived no-grad core value, gated against live differentiated forward",
                    "differentiated_objective": float(objective.detach()),
                    "reference_objective": float(.5*reference_residual.square().sum()),
                    "first_fingerprint": {"t": 999, "input_x_rms": float(x.detach().square().mean().sqrt()),
                        "residual_norm": float(reference_residual.norm()),
                        "raw_x0_clip_fraction": float((reference_raw.abs() > 1).float().mean())}}
        if visited != [999] or not same_rng(before_rng, rng_state(device)):
            raise RuntimeError("First input VJP changed time mapping or RNG")
        if any(p.grad is not None for p in zaps.dm.model.parameters()):
            raise RuntimeError("Unexpected score-weight gradients")
        return target, matched, reference_clipped, data
    finally:
        handle.remove()
        restore_parameters(zaps, state, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    args = parser.parse_args()
    if not str(args.device).startswith("cuda"):
        parser.error("Production path requires archived CUDA ImageNet data")
    from utils.diag_zaps_first_guidance import norm_calibration, IDENTITIES, validate_parent
    from utils.diag_zaps_parameter_gradient import validate_protocol, PROBE_TIMES, DELTAS, evaluate_replay, write_csv
    first_dir = Path(args.first_run_dir).resolve()
    first_path = first_dir / "run.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    guidance_dir = Path(first["source_guidance"]).resolve()
    guidance = json.loads((guidance_dir / "audit.json").read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"]).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source_dir = Path(saved["source"]).resolve()
    source = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
    validate_protocol(guidance, saved, source, PROBE_TIMES, DELTAS)
    validate_first_run(first, guidance, trace_dir)
    parameter_path = Path(first["resolved_parameter_run"])
    parameter = json.loads(parameter_path.read_text(encoding="utf-8"))
    validate_parent(parameter, guidance, guidance_dir)
    arm = saved["results"]["irregular_15_10_5"]
    if first["config"] != arm["config"] or parameter["config"] != arm["config"]:
        raise ValueError("Keep identical archived configurations")
    calibration = norm_calibration(guidance)
    output_dir = trace_dir / ("first_direction_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS_ROOT / "experiment_cards/R_first_direction.md").read_text(encoding="utf-8")
    record = {"experiment": "first_direction_fixed_norm_frozen_path", "status": "running",
              "arguments": vars(args), "source_first_run": str(first_path), "source_trace": str(trace_dir),
              "source_guidance": str(guidance_dir), "config": arm["config"], "calibration": calibration,
              "new_nfe": 0, "planned_nfe": 31, "source_method_nfe": 300, "optimizer_steps": 0,
              "retraining": False, "experiment_card": card, "gates": {}, "results": [],
              "limits": "frozen-path direction sensitivity, not author parity/state gain/trained baseline",
              "gt_usage": "report two predefined outputs only, no optimization or selection",
              "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                  text=True, capture_output=True, check=True).stdout.strip()}

    def save():
        (output_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                        allow_nan=False), encoding="utf-8")
        write_csv(output_dir / "summary.csv", record["results"])
        write_csv(output_dir / "trace.csv", record.get("replay", {}).get("rows", []))

    (output_dir / "experiment_card.md").write_text(card, encoding="utf-8")
    save()
    print("=== First direction: exact input VJP at the SAME calibrated norm ===", flush=True)
    print("Reuse completed approximate-direction output; 1 VJP + 30-step replay = 31 diagnostic NFE.", flush=True)
    print("No retraining/noise/LR/mask/default changes; exact squared-residual objective, not DPS L2.", flush=True)
    print(f"Records: {output_dir}", flush=True)
    started = time.perf_counter()
    try:
        import torch
        from configs.config import IMG_SIZE, METRICS_CONFIG
        from modules.main_single import load_diffusion_model, load_image_as_tensor
        from modules.degradations import get_operator
        from modules.zaps_algorithm import ZAPS, ddpm_posterior_step
        from utils.diag_dps_same_observation import file_sha256
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection, restore_sampling_rng, archive_fingerprint_check
        from utils.diag_zaps_parameter_gradient import restore_parameters
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.metrics import compute_all_metrics

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if (file_sha256(parameter_path) != first["parent_run_sha256"]
                or file_sha256(guidance_dir / "audit.json") != first["source_vjp_sha256"]):
            raise RuntimeError("Parent parameter/VJP record changed since first intervention")
        measurement = source_dir / "gaussian_deblur/measurement.pt"
        snapshot = trace_dir / "irregular_15_10_5_last_unroll_state.pt"
        paths = {"measurement_sha256": measurement, "core_sha256": PROJECTS_ROOT / "modules/zaps_algorithm.py",
                 "operator_source_sha256": PROJECTS_ROOT / "modules/degradations.py",
                 "image_sha256": Path(source["arguments"]["image"])}
        for key, path in paths.items():
            if file_sha256(path) != guidance[key]:
                raise RuntimeError(f"Identity changed: {key}")
        if file_sha256(snapshot) != first["snapshot_sha256"] or first["snapshot_sha256"] != parameter["snapshot_sha256"]:
            raise RuntimeError("Snapshot chain differs")
        print("Loading original model; checking checkpoint hash (local read, no download)...", flush=True)
        model = load_diffusion_model("imagenet", args.device)
        model.model.eval()
        if (file_sha256(model.ckpt_path) != guidance["checkpoint_sha256"] or model.num_steps != 1000
                or any(p.dtype != torch.float32 or not p.requires_grad for p in model.model.parameters())):
            raise RuntimeError("Keep archived fp32 weights and custom-checkpoint requires_grad; no optimizer")
        state = torch.load(snapshot, map_location="cpu", weights_only=True)
        grid = state["tau"].tolist()
        if grid != sorted(r["t"] for r in arm["rows"]) or len(grid) != 30 or grid[-1] != 999:
            raise RuntimeError("Snapshot grid differs")
        if not torch.equal(state["zeta"], torch.tensor(arm["epoch_used_zeta_ascending"], dtype=state["zeta"].dtype)):
            raise RuntimeError("Keep original epoch-10 PRE-update parameters")
        y = torch.load(measurement, map_location=args.device, weights_only=True)
        cache_path = first_dir / "first_norm_matched.pt"
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        for name, tensor in (("y", y), ("cached_output", cached), ("x_T", state["init_noise"])):
            if tuple(tensor.shape) != (1, 3, 256, 256) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise RuntimeError(f"Invalid archived tensor: {name}")
        operator = get_operator("gaussian_deblur", device=args.device, **source["task_configs"]["gaussian_deblur"])
        zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **arm["config"])
        mapping = saved_cuda_selection(state, saved["arguments"].get("device"), args.saved_cuda_index)
        restore_parameters(zaps, state, args.device)
        with torch.no_grad():
            cache_residual = y-operator.H(cached.to(args.device))
            cache_loss = float(cache_residual.double().square().mean())
        reference = first["results"][1]
        cache_gate = abs(cache_loss-reference["loss"]) <= 1e-8+2e-5*abs(reference["loss"])
        record["gates"]["cached_terminal_loss"] = {"passed": cache_gate, "current": cache_loss, "archived": reference["loss"]}
        record.update(identity={k: guidance[k] for k in IDENTITIES}, snapshot_sha256=first["snapshot_sha256"],
            parent_run_sha256=file_sha256(first_path), cached_output_sha256=file_sha256(cache_path),
            source_vjp_sha256=file_sha256(guidance_dir / "audit.json"), rng_mapping=mapping,
            runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "visible_cuda_count": torch.cuda.device_count()})
        record["cached_output_provenance_limit"] = (
            "Parent did not store output tensor hash; current read hash plus terminal-loss/metric compatibility, "
            "not historical tensorwise certification")
        save()
        if not cache_gate:
            raise RuntimeError("Cached calibrated output differs from completed parent")
        probe_path = guidance_dir / "irregular_15_10_5_probe_inputs.pt"
        captures = torch.load(probe_path, map_location="cpu", weights_only=True)
        captures = [c for c in captures if c["t"] == 999]
        if len(captures) != 1:
            raise RuntimeError("Need exactly one archived first-input capture")
        record["probe_inputs_sha256"] = file_sha256(probe_path)
        print("[1/2] One exact input-gradient evaluation at t=999...", flush=True)
        target, matched, clipped, probe = first_vjp(zaps, y, state, captures[0], calibration,
                                                  args.device, mapping, record)
        record["probe"] = probe
        first_gate = archive_fingerprint_check([probe["first_fingerprint"]], [first["calls"][1]["rows"][0]])
        record["gates"]["first_input_fingerprint"] = first_gate
        if not first_gate["passed"]:
            raise RuntimeError("First-input fingerprint differs")
        # Reconstruct the reference FIRST posterior and generator tail, with no
        # UNet calls. The core has 29 DDPM draws, none at final t=0. Guard each
        # score call below to verify it consumes no RNG. No third GPU trajectory.
        restore_sampling_rng(state, args.device, mapping["saved_sampling_index"])
        initial = state["init_noise"].to(args.device)
        with torch.no_grad():
            uncond = ddpm_posterior_step(initial, clipped, 999, grid[-2], model.alphas_cumprod,
                                       eta=1., mode="ddpm")
            matched_next = uncond+zaps.zeta[29]*matched
            next_rms = float(matched_next.square().mean().sqrt())
            for _ in range(28):
                torch.randn_like(initial)
        expected_rng = rng_state(args.device)
        old_next = first["calls"][1]["rows"][1]["input_x_rms"]
        first_next_gate = math.isclose(next_rms, old_next, rel_tol=2e-5, abs_tol=1e-6)
        record["gates"]["calibrated_first_next_rms"] = {"passed": first_next_gate, "current": next_rms, "archived": old_next}
        record["rng_reconstruction"] = {"diagnostic_draws": 29, "sampling_draws_expected": 29,
            "policy": "same original generator; eval UNet RNG guarded; fixed-small DDPM with terminal no-draw"}
        save()
        if not first_next_gate:
            raise RuntimeError("Live calibrated first step differs from the completed parent")
        del matched, clipped, captures, uncond, matched_next
        observation = {}
        print(f"Same-amplitude reference verified; approximate/exact cosine={probe['cosine_approx_exact']:.6f}", flush=True)
        print("[2/2] Exact FIRST direction at equal norm; remaining 29 original steps...", flush=True)
        with model_rng_guard(model.model, args.device) as guarded:
            with first_direction_scope(zaps, target, calibration["approximate_norm"], observation):
                output, replay, end_rng = evaluate_replay(zaps, y, state, initial, device=args.device,
                    saved_index=mapping["saved_sampling_index"], accounting=record)
        record["replay"], record["intervention"] = replay, observation
        record["gates"]["pairing"] = {"passed": same_rng(expected_rng, end_rng) and len(guarded) == 30,
            "expected_rng_tail_equal": same_rng(expected_rng, end_rng), "score_rng_guarded_calls": len(guarded)}
        norm_gate = math.isclose(observation["applied_correction_norm"], probe["reference_correction_norm"],
                                 rel_tol=2e-5, abs_tol=1e-6)
        record["gates"]["equal_correction_norm"] = {"passed": norm_gate,
            "reference": probe["reference_correction_norm"], "actual": observation["applied_correction_norm"]}
        replay_first_gate = archive_fingerprint_check([replay["rows"][0]], [first["calls"][1]["rows"][0]])
        record["gates"]["replay_first_input"] = replay_first_gate
        save()
        if (not norm_gate or not replay_first_gate["passed"] or not record["gates"]["pairing"]["passed"]
                or record["new_nfe"] != 31):
            raise RuntimeError("Direction pairing/norm/budget gate failed")
        gt = load_image_as_tensor(source["arguments"]["image"]).cpu()
        cached_metrics = compute_all_metrics(cached, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        metrics_gate = all(math.isclose(cached_metrics[k], reference[k], rel_tol=2e-5, abs_tol=1e-6)
                           for k in ("psnr", "ssim", "lpips"))
        record["gates"]["cached_metrics"] = {"passed": metrics_gate, "current": cached_metrics,
                                             "archived": {k: reference[k] for k in cached_metrics}}
        if not metrics_gate:
            raise RuntimeError("Cached parent evaluation differs; no quality interpretation")
        metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        record["results"] = [dict(reference, policy="cached_norm_matched", new_nfe=0),
            dict(policy="first_exact_equal_norm", **metrics, loss=replay["loss64"], residual=replay["residual_norm"],
                 next_rms=replay["rows"][1]["input_x_rms"], next_clip_fraction=replay["rows"][1]["raw_x0_clip_fraction"],
                 diagnostic_nfe=30, seconds=replay["seconds"], new_nfe=31)]
        # Uniform fields for CSV; cached costs explicitly retained from parent.
        record["results"][0]["source_diagnostic_nfe"] = reference["diagnostic_nfe"]
        record["results"][1]["source_diagnostic_nfe"] = 0
        record["decision"] = interpret_direction(cached_metrics, metrics)
        torch.save(output, output_dir / "first_exact_equal_norm.pt")
        record["total_seconds"] = time.perf_counter()-started
        record["status"] = "complete"
        save()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save()
        raise
    print("\n=== Paired first-direction summary (delta vs calibrated approximate direction) ===", flush=True)
    print(f"{'policy':>24} {'PSNR':>9} {'dPSNR':>9} {'SSIM':>8} {'dSSIM':>8} {'LPIPS':>8} {'dLPIPS':>9} {'nextRMS':>9} {'clip%':>8}", flush=True)
    b = record["results"][0]
    for row in record["results"]:
        print(f"{row['policy']:>24} {row['psnr']:9.4f} {row['psnr']-b['psnr']:+9.4f} "
              f"{row['ssim']:8.4f} {row['ssim']-b['ssim']:+8.4f} {row['lpips']:8.4f} "
              f"{row['lpips']-b['lpips']:+9.4f} {row['next_rms']:9.4f} {100*row['next_clip_fraction']:8.3f}", flush=True)
    print("Decision:", json.dumps(record["decision"], ensure_ascii=False), flush=True)
    print(f"New diagnostic NFE={record['new_nfe']}; records: {output_dir}", flush=True)
    print("Single-image frozen-path intervention, not retrained ZAPS/paper parity/state gain.", flush=True)


if __name__ == "__main__":
    main()
