"""Observe archived ImageNet blur trajectories and audit their likelihood VJPs.

No training, replacement trajectory, GT-based selection, or sampler changes.
All directions use the SAME clipped residual and L=0.5*||y-H(clamp(x0_raw))||^2.
The masked approximation applies the clamp VJP BEFORE the approximate Jacobian.
Improved gradient agreement alone does not prove a reconstruction improvement.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))

from utils.diag_dps_same_observation import file_sha256
from utils.diag_dps_lowstep_variance import find_baseline
from utils.diag_zaps_sampler_parity import error_metrics, restore_rng, rng_state, same_rng


def probe_indices(grid, requested_times):
    if not grid or grid != sorted(set(grid)):
        raise ValueError("Need a sorted, unique original-time grid")
    if not requested_times or any(t < 0 or t > 999 for t in requested_times):
        raise ValueError("Probe times must be in [0,999]")
    return sorted({min(range(len(grid)), key=lambda i: abs(grid[i] - t))
                   for t in requested_times}, reverse=True)


def approximate_vjp(zaps, vector, alpha_bar, index, mask=None):
    """B^T(M v), NOT M(B^T v); B is the symmetric wavelet approximation."""
    source = vector if mask is None else mask * vector
    hessian = zaps.dwt.synthesis(zaps.D[index] * zaps.dwt.analysis(source))
    return (source + (1.0 - alpha_bar) * hessian) / alpha_bar.sqrt().clamp(min=1e-8)


def direction_metrics(candidate, exact):
    import torch
    left, right = candidate.detach().double().flatten(), exact.detach().double().flatten()
    ln, rn = float(left.norm()), float(right.norm())
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Non-finite likelihood direction")
    return {
        "norm": ln,
        "norm_ratio": ln / rn if rn > 1e-12 else None,
        "relative_error": float((left - right).norm()) / rn if rn > 1e-12 else None,
        "cosine": float(torch.dot(left, right) / (left.norm() * right.norm()))
                  if ln > 1e-12 and rn > 1e-12 else None,
    }


def probe(zaps, y, captured):
    """Independent denoiser graphs for the two VJPs; custom checkpoint is single-use."""
    import torch
    index, t = captured["index"], captured["t"]
    ab = zaps.dm.alphas_cumprod[t]
    x = captured["x"].to(zaps.device).requires_grad_(True)
    tb = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    with torch.enable_grad():
        epsilon = zaps.dm._predict_eps(x, tb)
        eps_gate = error_metrics(epsilon, captured["epsilon"].to(zaps.device), 2e-5, 2e-5)
        if not eps_gate["passed"]:
            raise RuntimeError(f"Probe model output differs from the captured trajectory at t={t}: {eps_gate}")
        raw = (x - (1.0 - ab).sqrt() * epsilon) / ab.sqrt().clamp(min=1e-8)
        clipped = raw.clamp(-1.0, 1.0)
        residual = y - zaps.A.H(clipped)
        objective = 0.5 * residual.square().sum()
        exact = -torch.autograd.grad(objective, x)[0].detach()
        # Use the SAME clipped residual, rather than a new un-clipped loss.
        v = zaps.A.transpose(residual.detach()).detach()
        # Guided-diffusion's custom checkpoint deletes ctx input tensors in
        # backward even with retain_graph=True. Do not backward through it twice.
        raw_x = x.detach().requires_grad_(True)
        raw_epsilon = zaps.dm._predict_eps(raw_x, tb)
        second_eps_gate = error_metrics(raw_epsilon, epsilon.detach(), 2e-5, 2e-5)
        if not second_eps_gate["passed"]:
            raise RuntimeError(f"Independent VJP graphs have different predictions at t={t}: {second_eps_gate}")
        raw_graph = (raw_x - (1.0 - ab).sqrt() * raw_epsilon) / ab.sqrt().clamp(min=1e-8)
        exact_raw_same_v = torch.autograd.grad(raw_graph, raw_x, grad_outputs=v)[0].detach()
        # torch.clamp has derivative 1 at its endpoints, 0 strictly outside.
        mask = (raw.detach().abs() <= 1.0).to(raw.dtype)

        frozen_x = x.detach().requires_grad_(True)
        frozen_raw = (frozen_x - (1.0 - ab).sqrt() * epsilon.detach()) / ab.sqrt().clamp(min=1e-8)
        frozen_loss = 0.5 * (y - zaps.A.H(frozen_raw.clamp(-1.0, 1.0))).square().sum()
        frozen_exact = -torch.autograd.grad(frozen_loss, frozen_x)[0].detach()
    with torch.no_grad():
        identity_masked = mask * v / ab.sqrt().clamp(min=1e-8)
        chain_gate = error_metrics(identity_masked, frozen_exact, 2e-5, 2e-5)
        if not chain_gate["passed"]:
            raise RuntimeError(f"Frozen-epsilon clamp/adjoint chain-rule check failed at t={t}: {chain_gate}")
        approximate = approximate_vjp(zaps, v, ab, index)
        masked = approximate_vjp(zaps, v, ab, index, mask=mask)
        variants = {
            "raw": approximate, "masked": masked,
            "identity_raw": v / ab.sqrt().clamp(min=1e-8),
            "identity_masked": identity_masked,
            "exact_raw_same_v": exact_raw_same_v,
        }
        row = {
            "step": len(zaps.tau) - 1 - index, "t": t, "parameter_index": index,
            "clip_fraction": float((mask == 0).float().mean()),
            "residual_norm": float(residual.detach().norm()),
            "objective_half_squared_l2": float(objective.detach()),
            "zeta": float(zaps.zeta[index]), "exact_norm": float(exact.double().norm()),
            "raw_correction_norm": float((zaps.zeta[index] * approximate).norm()),
            "masked_correction_norm": float((zaps.zeta[index] * masked).norm()),
            "mask_change_relative_to_raw": float((masked - approximate).double().norm()
                                                  / approximate.double().norm().clamp(min=1e-12)),
        }
        for name, direction in variants.items():
            row.update({name + "_" + key: value for key, value in direction_metrics(direction, exact).items()})
    del x, epsilon, raw, clipped, residual, objective, frozen_x, frozen_raw, frozen_loss
    del raw_x, raw_epsilon, raw_graph
    return row, {"epsilon_vs_captured": eps_gate, "independent_graph_epsilon": second_eps_gate,
                 "frozen_epsilon_clamp_chain": chain_gate}


def region_summary(rows, predicate):
    selected = [r for r in rows if predicate(r["t"]) and r["raw_relative_error"] is not None]
    if not selected:
        return {"count": 0}
    return {
        "count": len(selected), "times": [r["t"] for r in selected],
        "mean_clip_fraction": statistics.mean(r["clip_fraction"] for r in selected),
        "median_raw_relative_error": statistics.median(r["raw_relative_error"] for r in selected),
        "median_masked_relative_error": statistics.median(r["masked_relative_error"] for r in selected),
        "masked_relative_error_improved_count": sum(r["masked_relative_error"] < r["raw_relative_error"]
                                                    for r in selected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-times", type=int, nargs="+", default=[999, 916, 833, 667, 333, 143, 0])
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source = json.loads((Path(saved["source"]) / "run.json").read_text(encoding="utf-8"))
    task = saved["arguments"]["task"]
    if task != "gaussian_deblur" or source["arguments"]["dataset"] != "imagenet":
        parser.error("Only the archived ImageNet Gaussian-deblurring pair is in scope")
    if set(saved["results"]) != {"irregular_15_10_5", "uniform_30"}:
        raise RuntimeError("Need the two original archived schedule arms")
    for arm in saved["results"].values():
        if (not arm["gates"]["passed"] or arm["config"]["eta"] != 1.0
                or arm["config"]["sampler_mode"] != "ddpm"
                or arm["config"].get("surrogate_score_jacobian", False)):
            raise RuntimeError("Use the original eta=1 fixed-grid last-unroll archive")

    import torch
    from configs.config import IMG_SIZE
    from modules.main_single import load_diffusion_model
    from modules.degradations import get_operator
    from modules.zaps_algorithm import ZAPS
    from utils.diag_zaps_trace_audit import classify_parity, relative_error

    model = load_diffusion_model("imagenet", args.device)
    model.model.eval()
    if not all(p.requires_grad for p in model.model.parameters()):
        raise RuntimeError("Keep model requires_grad intact for custom checkpoint input VJP; no optimizer is used")
    operator = get_operator(task, device=args.device, **source["task_configs"][task])
    measurement_path = Path(saved["source"]) / task / "measurement.pt"
    y = torch.load(measurement_path, map_location=args.device, weights_only=True)
    print("\n=== Archived ZAPS guidance/clamp VJP audit ===", flush=True)
    print("Checking checkpoint/observation/H identities against completed DPS baseline...", flush=True)
    expected = {
        "source_trace": str(trace_dir), "task": task, "seed": int(source["arguments"]["seed"]),
        "checkpoint_sha256": file_sha256(model.ckpt_path),
        "measurement_sha256": file_sha256(measurement_path),
        "image_sha256": file_sha256(source["arguments"]["image"]),
        "operator_config": source["task_configs"][task],
        "operator_source_sha256": file_sha256(PROJECTS_ROOT / "modules/degradations.py"),
    }
    baseline_dir, _ = find_baseline(trace_dir, None, expected)
    output_dir = trace_dir / ("guidance_clipping_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "experiment": "passive_clamp_guidance_VJP", "status": "running", **expected,
        "arguments": vars(args), "identity_baseline": str(baseline_dir),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       capture_output=True, text=True, check=True).stdout.strip(),
        "core_sha256": file_sha256(PROJECTS_ROOT / "modules/zaps_algorithm.py"),
        "objective": "L=0.5*sum((y-H(clamp(x0_raw)))^2); directions are -dL/dx_t",
        "mask_order": "B^T(M H^T r), not M(B^T H^T r)",
        "limits": "No GT, output selection, training or replacement trajectory; VJP agreement is not a PSNR gain",
        "results": {}, "new_nfe": 0,
    }

    def save_record():
        (output_dir / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                          allow_nan=False), encoding="utf-8")

    def number(value):
        return f"{value:.4f}" if value is not None else "undefined"

    save_record()
    try:
        for name, arm in saved["results"].items():
            state = torch.load(trace_dir / f"{name}_last_unroll_state.pt", map_location="cpu", weights_only=True)
            zaps = ZAPS(model, operator, img_size=IMG_SIZE[0], **arm["config"])
            zaps.tau = state["tau"].to(args.device)
            with torch.no_grad():
                zaps.zeta.copy_(state["zeta"].to(args.device))
                zaps.D.copy_(state["D"].to(args.device))
            grid = zaps.tau.tolist()
            selected = set(probe_indices(grid, args.probe_times))
            init_noise = state["init_noise"].to(args.device)
            snapshots, visited = [], []
            original_tweedie = zaps._tweedie_estimate

            def observe(x, eps, ab, index):
                t = int(zaps.tau[index])
                visited.append(t)
                if index in selected:
                    snapshots.append({"index": index, "t": t,
                                      "x": x.detach().cpu().clone(),
                                      "epsilon": eps.detach().cpu().clone()})
                return original_tweedie(x, eps, ab, index)

            print(f"\n--- {name}: three 30-step passive replays, then {len(selected)} gradient probes ---", flush=True)
            restore_rng(state, args.device)
            reference, nfe_a, _ = zaps.sample(y, init_noise=init_noise)
            end_a = rng_state(args.device)
            restore_rng(state, args.device)
            repeat, nfe_b, _ = zaps.sample(y, init_noise=init_noise)
            end_b = rng_state(args.device)
            restore_rng(state, args.device)
            with patch.object(zaps, "_tweedie_estimate", side_effect=observe):
                observed, nfe_c, _ = zaps.sample(y, init_noise=init_noise)
            end_c = rng_state(args.device)
            record["new_nfe"] += nfe_a + nfe_b + nfe_c
            parity = classify_parity(
                relative_error(reference, repeat), relative_error(reference, observed),
                nfe_a == nfe_b == nfe_c == len(grid) == 30
                and visited == list(reversed(grid)) and len(snapshots) == len(selected)
                and same_rng(end_a, end_b) and same_rng(end_a, end_c),
            )
            record["results"][name] = {"replay_gate": parity, "probe_gates": [], "rows": [],
                                       "archived_psnr": arm["metrics"]["psnr"]}
            save_record()
            if not parity["passed"]:
                raise RuntimeError(f"Passive replay gate failed: {parity}")
            print("Passive output/RNG/time-map gate passed; original reconstruction unchanged.", flush=True)
            del reference, repeat, observed
            torch.save(snapshots, output_dir / f"{name}_probe_inputs.pt")
            for snapshot in snapshots:
                print(f"Probing original t={snapshot['t']}...", flush=True)
                row, gates = probe(zaps, y, snapshot)
                record["new_nfe"] += 2
                if any(p.grad is not None for p in model.model.parameters()):
                    raise RuntimeError("Unexpected accumulated model parameter gradients")
                record["results"][name]["rows"].append(row)
                record["results"][name]["probe_gates"].append({"t": row["t"], **gates})
                save_record()
                print(f"t={row['t']} clip={100*row['clip_fraction']:.1f}% "
                      f"cos raw/masked={number(row['raw_cosine'])}/{number(row['masked_cosine'])} "
                      f"rel raw/masked={number(row['raw_relative_error'])}/{number(row['masked_relative_error'])}", flush=True)
            rows = record["results"][name]["rows"]
            record["results"][name]["high_noise"] = region_summary(rows, lambda t: t >= 600)
            record["results"][name]["late_noise"] = region_summary(rows, lambda t: t <= 400)
            with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            del zaps, state, snapshots, init_noise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        record["status"] = "complete"
        save_record()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save_record()
        raise
    print("\n=== Guidance/clamp summary (same clipped squared-residual objective) ===", flush=True)
    print(f"{'schedule':>20} {'t':>4} {'clip%':>7} {'cosRaw':>9} {'cosMask':>9} {'relRaw':>10} {'relMask':>10} {'normRaw/exact':>14} {'normMask/exact':>15}", flush=True)
    for name, result in record["results"].items():
        for row in result["rows"]:
            print(f"{name:>20} {row['t']:4d} {100*row['clip_fraction']:7.1f} "
                  f"{number(row['raw_cosine']):>9} {number(row['masked_cosine']):>9} "
                  f"{number(row['raw_relative_error']):>10} {number(row['masked_relative_error']):>10} "
                  f"{number(row['raw_norm_ratio']):>14} {number(row['masked_norm_ratio']):>15}", flush=True)
        print(f"{name} high-noise summary: {json.dumps(result['high_noise'])}", flush=True)
    print("Smaller relative error/better cosine supports a clipping-path discrepancy; not proof of a PSNR gain or paper bug.", flush=True)
    print("Both approximations poor: examine the score-Jacobian approximation before changing sampling noise or learning rate.", flush=True)
    print(f"No training/replacement reconstruction; new NFE={record['new_nfe']}; records: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
