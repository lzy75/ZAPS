"""Check the existing ZAPS DDPM step against the vendored DPS implementation.

Reuses a saved last-unroll snapshot; no optimization, GT metric, noise scaling,
or variance-policy change. Each comparison uses the SAME x_t, epsilon, variance
prediction and random draw. The reference keeps the source arm's variance policy,
not necessarily DPS/configs/diffusion_config.yaml's default learned_range.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def load_dps_reference(root):
    """Import unmodified DPS sources under a separate package, not guided_diffusion.

    Loading under a private name avoids replacing the package used by the UNet.
    DPS's real util/img_utils.py is imported too; no numerical functions are stubbed.
    """
    root = Path(root).resolve()
    source = root / "guided_diffusion" / "gaussian_diffusion.py"
    processors = source.with_name("posterior_mean_variance.py")
    if not source.is_file() or not processors.is_file():
        raise FileNotFoundError(f"DPS reference source missing under {root}")
    existing = sys.modules.get("util.img_utils")
    if existing is not None and Path(existing.__file__).resolve() != root / "util/img_utils.py":
        raise RuntimeError("An unrelated util.img_utils is already imported; refusing a mixed reference")
    package_name = "_zaps_sampler_dps_reference"
    if package_name in sys.modules:
        raise RuntimeError("Reference package already loaded")
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "guided_diffusion")]
    sys.modules[package_name] = package
    sys.path.insert(0, str(root))
    try:
        reference = importlib.import_module(package_name + ".gaussian_diffusion")
    finally:
        sys.path.remove(str(root))
    if Path(reference.__file__).resolve() != source:
        raise RuntimeError("Unexpected reference module origin")
    return reference, {
        "root": str(root),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (source, processors)},
        "import_policy": "unmodified source; private package; real DPS utility imports",
    }


def error_metrics(actual, expected, atol, rtol):
    import torch
    a, b = actual.detach().double(), expected.detach().double()
    difference = a - b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {
        "relative_error": float(difference.norm() / b.norm().clamp(min=1e-30)),
        "max_absolute_error": float(difference.abs().max()),
        "passed": finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def restore_rng(state, device):
    import torch
    torch.set_rng_state(state["cpu_rng"])
    if str(device).startswith("cuda") and state["cuda_rng"] is not None:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise RuntimeError("Saved/current CUDA device counts differ; RNG replay cannot be guaranteed")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def rng_state(device):
    import torch
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}


def same_rng(first, second):
    import torch
    return (torch.equal(first["cpu"], second["cpu"])
            and len(first["cuda"]) == len(second["cuda"])
            and all(torch.equal(a, b) for a, b in zip(first["cuda"], second["cuda"])))


def check_step(zaps, reference, captured, posterior, x, x0, t, s, ab,
               eta, learned_log_var, mode, rows, atol, rtol):
    """Observe an actual core call and return its unmodified result."""
    import torch
    compact = len(zaps.tau) - 1 - len(rows)
    if int(zaps.tau[compact]) != t or mode != "ddpm" or eta != 1.0:
        raise RuntimeError("Unexpected core step/policy in a baseline-only audit")
    expected_s = int(zaps.tau[compact - 1]) if compact else -1
    if s != expected_s or captured["t"] != t:
        raise RuntimeError("Core timestep or captured model output does not match the grid")
    original_draw = torch.randn_like
    draws = []

    def capture_draw(*args, **kwargs):
        noise = original_draw(*args, **kwargs)
        draws.append(noise)
        return noise

    # Run the REAL core posterior function first, consuming exactly its usual RNG.
    with patch.object(torch, "randn_like", side_effect=capture_draw):
        actual_next = posterior(x, x0, t, s, ab, eta=eta,
                                learned_log_var=learned_log_var, mode=mode)
    if len(draws) != (0 if s < 0 else 1):
        raise RuntimeError("Unexpected number of core posterior noise draws")
    noise = draws[0] if draws else torch.zeros_like(x)
    mapped_times = []

    def saved_model(ref_x, mapped_t):
        if not torch.equal(mapped_t, torch.full_like(mapped_t, t)):
            raise RuntimeError(f"Reference compact index {compact} maps to the wrong original t")
        if not torch.equal(ref_x, x):
            raise RuntimeError("Reference received a different x_t")
        mapped_times.append(int(mapped_t[0]))
        eps, var = captured["eps"], captured["var"]
        return torch.cat((eps, var), dim=1) if var is not None else eps

    with torch.no_grad():
        index = torch.full((x.shape[0],), compact, device=x.device, dtype=torch.long)
        ref = reference.p_mean_variance(saved_model, x, index)
        actual_mean = posterior(x, x0, t, s, ab, eta=0.0, mode=mode)
        # Separate mean-coefficient error from any difference in Tweedie estimates.
        shared_mean = reference.q_posterior_mean_variance(x0, x, index)[0]
        # Probe the actual core's effective std with zero inputs and unit noise.
        # This avoids subtraction cancellation and contains no model/RNG calls.
        with patch.object(torch, "randn_like", return_value=torch.ones_like(x)):
            actual_std = posterior(torch.zeros_like(x), torch.zeros_like(x0), t, s,
                                   ab, eta=1.0, learned_log_var=learned_log_var, mode=mode)
        ref_std = (torch.exp(0.5 * ref["log_variance"]) if s >= 0
                   else torch.zeros_like(x))
        # Call DPS's actual p_sample with the identical noise, not a retyped update.
        with patch.object(torch, "randn_like", return_value=noise):
            ref_next = reference.p_sample(saved_model, x, index)["sample"]
        checks = {
            "x0": error_metrics(x0, ref["pred_xstart"], atol, rtol),
            "mean_shared_x0": error_metrics(actual_mean, shared_mean, atol, rtol),
            "mean": error_metrics(actual_mean, ref["mean"], atol, rtol),
            "std": error_metrics(actual_std, ref_std, atol, rtol),
            "next": error_metrics(actual_next, ref_next, atol, rtol),
        }
    rows.append({
        "step": len(rows), "t": t, "s": s, "compact_index": compact,
        "time_mapping_passed": mapped_times == [t, t],
        "noise_draws": len(draws), "effective_std_rms": float(actual_std.square().mean().sqrt()),
        "checks": checks,
        "passed": mapped_times == [t, t] and all(c["passed"] for c in checks.values()),
    })
    print(f"{len(rows)-1:3d} {t:4d} {s:4d} " + " ".join(
        f"{checks[key]['relative_error']:10.3e}" for key in checks
    ) + (" PASS" if rows[-1]["passed"] else " FAIL"), flush=True)
    return actual_next


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dps-root", default=str(REPO_ROOT / "DPS"))
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()
    if not 0 < args.atol <= 1e-4 or not 0 < args.rtol <= 1e-4:
        parser.error("tolerances must be positive and <= 1e-4; do not hide failures by relaxing them")
    import torch
    from modules.degradations import get_operator
    from modules.diffusion_model import load_imagenet_model, load_ffhq_model
    from modules import zaps_algorithm as core
    from utils.diag_ffhq_regression import TransposeMode

    trace_dir = Path(args.trace_dir).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source = Path(saved["source"])
    source_record = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if not saved["results"]:
        raise RuntimeError("No source arms")
    for arm in saved["results"].values():
        if not arm["gates"]["passed"]:
            raise RuntimeError("Source last-opt replay gate failed")
        if arm["config"]["sampler_mode"] != "ddpm" or arm["config"]["eta"] != 1.0:
            raise RuntimeError("Audit requires the unchanged DDPM/eta=1 baseline")
    task = saved["arguments"]["task"]
    run_args = source_record["arguments"]
    loader = {"imagenet": load_imagenet_model, "ffhq": load_ffhq_model}[run_args["dataset"]]
    model = loader(model_dir=str(PROJECTS_ROOT / "modules/models"), device=args.device)
    dps, reference_meta = load_dps_reference(args.dps_root)
    y = torch.load(source / task / "measurement.pt", map_location=args.device, weights_only=True)
    operator = get_operator(task, device=args.device, **source_record["task_configs"][task])
    if task == "super_resolution":
        operator = TransposeMode(operator, run_args["sr_transpose"]).to(args.device)
    output_dir = trace_dir / ("sampler_parity_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    record = {"arguments": vars(args), "git_revision": revision,
              "reference": reference_meta, "results": {},
              "semantics": "same inputs/model output/noise; source variance policy; no GT or training",
              "tolerance": "elementwise |a-b| <= atol + rtol*|b|; relative norms also reported"}

    def save_record():
        (output_dir / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")

    print("\n=== ZAPS core vs original DPS step (no training/no policy change) ===", flush=True)
    print(f"Reference: {dps.__file__}\nRecords: {output_dir}", flush=True)
    posterior = core.ddpm_posterior_step
    for name, arm in saved["results"].items():
        state = torch.load(trace_dir / f"{name}_last_unroll_state.pt", map_location="cpu", weights_only=True)
        zaps = core.ZAPS(model, operator, img_size=state["init_noise"].shape[-1], **arm["config"])
        zaps.tau = state["tau"].to(args.device)
        with torch.no_grad():
            zaps.zeta.copy_(state["zeta"].to(zaps.device))
            zaps.D.copy_(state["D"].to(zaps.device))
        tau = [int(t) for t in zaps.tau.tolist()]
        if tau != sorted(set(tau)) or tau[0] != 0 or tau[-1] != model.num_steps - 1:
            raise RuntimeError("Source grid must be unique, ascending and include both endpoints")
        variance = "learned_range" if zaps.use_learned_var else "fixed_small"
        reference = dps.DDPM(use_timesteps=tau,
                             betas=dps.get_named_beta_schedule("linear", model.num_steps),
                             model_mean_type="epsilon", model_var_type=variance,
                             dynamic_threshold=False, clip_denoised=True, rescale_timesteps=False)
        if reference.timestep_map != tau:
            raise RuntimeError("Reference respacing did not retain the source grid")
        init_noise = state["init_noise"].to(args.device)
        outputs, nfes, end_rngs = [], [], []
        print(f"\n{name}: variance={variance}; checking unchanged core replay...", flush=True)
        for _ in range(2):
            restore_rng(state, args.device)
            with torch.no_grad():
                result, nfe, _ = zaps.sample(y, init_noise=init_noise)
            outputs.append(result)
            nfes.append(nfe)
            end_rngs.append(rng_state(args.device))
        repeat = error_metrics(outputs[0], outputs[1], 1e-7, 1e-7)
        floor = repeat["relative_error"]
        rows, captured = [], {}
        original_predict = model._predict_eps_var

        def predict_capture(x, t):
            eps, var = original_predict(x, t)
            captured.update(eps=eps, var=var, t=int(t[0]))
            return (eps, var) if zaps.use_learned_var else eps

        def audited_posterior(x, x0, t, s, ab, eta=1.0, learned_log_var=None, mode="ddpm"):
            return check_step(zaps, reference, captured, posterior, x, x0, t, s, ab,
                              eta, learned_log_var, mode, rows, args.atol, args.rtol)

        print("  k    t    s     x0_rel sharedMean       mean        std       next", flush=True)
        predict_method = "_predict_eps_var" if zaps.use_learned_var else "_predict_eps"
        restore_rng(state, args.device)
        with torch.no_grad(), patch.object(model, predict_method, side_effect=predict_capture), \
                patch.object(core, "ddpm_posterior_step", side_effect=audited_posterior):
            audited, audited_nfe, _ = zaps.sample(y, init_noise=init_noise)
        rng_passed = same_rng(end_rngs[0], end_rngs[1]) and same_rng(end_rngs[0], rng_state(args.device))
        observation = error_metrics(audited, outputs[0], 1e-7, 1e-7)
        replay_limit = min(1e-5, max(1e-7, 10 * floor))
        replay_passed = (floor <= 1e-5 and observation["relative_error"] <= replay_limit
                         and nfes == [len(tau), len(tau)] and audited_nfe == len(tau)
                         and len(rows) == len(tau) and rng_passed)
        passed = replay_passed and all(row["passed"] for row in rows)
        record["results"][name] = {
            "passed": passed, "variance_policy": variance, "timesteps": tau,
            "core_repeat": repeat, "audit_vs_core": observation,
            "rng_equal": rng_passed,
            "replay_relative_limit": replay_limit, "replay_passed": replay_passed,
            "nfe": sum(nfes) + audited_nfe, "rows": rows,
        }
        save_record()
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            flat_rows = [{**{k: v for k, v in r.items() if k != "checks"},
                          **{f"{key}_{metric}": value for key, check in r["checks"].items()
                             for metric, value in check.items()}} for r in rows]
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"{name}: {'PASS' if passed else 'FAIL'}; repeat={floor:.3e}, "
              f"observer/core={observation['relative_error']:.3e}", flush=True)
        del zaps, outputs, audited
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    record["passed"] = all(arm["passed"] for arm in record["results"].values())
    save_record()
    print("\n=== Sampler parity summary ===", flush=True)
    for name, result in record["results"].items():
        print(f"{name:>20} {result['variance_policy']:>13} "
              f"{'PASS' if result['passed'] else 'FAIL'} NFE={result['nfe']}", flush=True)
    print("PASS excludes a forward DDPM/Tweedie mismatch on these saved paths only.", flush=True)
    print("It does NOT validate ZAPS guidance/backprop, paper settings or dataset-average reproduction.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)
    if not record["passed"]:
        raise RuntimeError("Sampler audit failed; inspect the first failed component, not PSNR")


if __name__ == "__main__":
    main()
