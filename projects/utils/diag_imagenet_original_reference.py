"""Zero-UNet audit of the ORIGINAL ImageNet Gaussian local reference.

Reuse cached original output and denoiser inputs. Do not load a model, optimize,
sample, alter initialization/variance/tolerance, or promote a tuned reference.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
ROOT = PROJECTS.parent
sys.path.insert(0, str(PROJECTS))
ARMS = ("irregular_15_10_5", "uniform_30")


def original_config(config):
    required = dict(num_steps=30, schedule=[15, 10, 5], timestep_spacing="linear",
        schedule_power=2., num_epochs=10, lr=.001, zeta_init=.2, d_init=.2, eta=1.,
        use_learned_var=False, sampler_mode="ddpm", wave="db4", level=3,
        surrogate_score_jacobian=False)
    if config != required:
        raise ValueError("Need the exact archived original config; no first-zeta/state/LR/noise tuning")


def old_sampler_scope(record):
    """Do not silently rewrite the old failed x0 comparison as an overall PASS."""
    if set(record.get("results", {})) != set(ARMS):
        raise ValueError("Need the two original sampler-audit arms")
    results = {}
    for name, arm in record["results"].items():
        rows = arm.get("rows", [])
        if (arm.get("replay_passed") is not True or len(rows) != 30 or arm.get("nfe") != 90
            or arm.get("variance_policy") != "fixed_small"):
            raise ValueError("Original replay/coverage/policy is not certified")
        failures = []
        for row in rows:
            if row.get("time_mapping_passed") is not True:
                raise ValueError("Original time mapping failed")
            for key, check in row["checks"].items():
                if check["passed"] is not True:
                    failures.append(dict(t=row["t"], component=key, **check))
        if any(f["component"] != "x0" for f in failures):
            raise ValueError("Source has failures beyond Tweedie; inspect before closing anything")
        results[name] = dict(old_overall_passed=arm["passed"], replay_passed=True,
            failures_preserved=failures, scope="past complete 30-point audit; not a new replay")
    return results


def cached_sampler_checks(probe, grid, ab, reference, core):
    """Same x/epsilon; actual core helpers; unmodified DPS q posterior.

    Shared-x0 isolates the posterior. FP64 tests algebra separately; it does
    NOT change production FP32 arithmetic or excuse its original failed gate.
    No model call or random draw is needed; unit noise probes standard deviation.
    """
    import torch
    from utils.diag_zaps_sampler_parity import error_metrics
    t, index = int(probe["t"]), int(probe["index"])
    if not 0 <= index < len(grid) or grid[index] != t or reference.timestep_map != grid:
        raise ValueError("Cached probe index/time mapping is wrong")
    x, eps = probe["x"], probe["epsilon"]
    if x.shape != eps.shape or x.dtype != torch.float32 or eps.dtype != torch.float32:
        raise ValueError("Need same-shape FP32 cached model inputs/outputs")
    if not torch.isfinite(x).all() or not torch.isfinite(eps).all():
        raise ValueError("Non-finite cached tensor")
    s = grid[index-1] if index else -1
    tb = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
    mean_processor = reference.mean_processor
    production_x0 = core.ZAPS._tweedie_estimate(SimpleNamespace(surrogate_score_jacobian=False),
        x, eps, ab[t], index)
    dps_x0 = mean_processor.get_mean_and_xstart(x, tb, eps)[1]
    mean = core.ddpm_posterior_step(x, production_x0, t, s, ab, eta=0.)
    shared_mean, shared_variance, _ = reference.q_posterior_mean_variance(production_x0, x, tb)
    with patch.object(torch, "randn_like", return_value=torch.ones_like(x)):
        std = core.ddpm_posterior_step(torch.zeros_like(x), torch.zeros_like(x), t, s, ab)
        unit_next = core.ddpm_posterior_step(x, production_x0, t, s, ab)
    shared_std = shared_variance.sqrt() if s >= 0 else torch.zeros_like(x)
    shared_next = shared_mean+shared_std
    # Pull the exact respaced coefficients from the existing DPS processor.
    coefficient_a = float(mean_processor.sqrt_recip_alphas_cumprod[index])
    coefficient_b = float(mean_processor.sqrt_recipm1_alphas_cumprod[index])
    alpha64 = float(reference.alphas_cumprod[index])
    x64, eps64 = x.double(), eps.double()
    direct64 = core.ZAPS._tweedie_estimate(SimpleNamespace(surrogate_score_jacobian=False),
        x64, eps64, torch.tensor(alpha64,dtype=torch.float64,device=x.device), index)
    expanded64 = (coefficient_a*x64-coefficient_b*eps64).clamp(-1,1)
    checks = {
        "fp32_x0_original_tolerance": error_metrics(production_x0, dps_x0, 2e-5, 2e-5),
        "shared_x0_mean": error_metrics(mean, shared_mean, 2e-5, 2e-5),
        "shared_x0_std": error_metrics(std, shared_std, 2e-5, 2e-5),
        "shared_x0_unit_noise_next": error_metrics(unit_next, shared_next, 2e-5, 2e-5),
        "fp64_tweedie_algebra": error_metrics(direct64, expanded64, 1e-10, 1e-10),
    }
    return dict(t=t, index=index, next_t=s, checks=checks,
        input_x_rms=float(x.square().mean().sqrt()),
        raw_x0_clip_fraction=float((((x-(1-ab[t]).sqrt()*eps)/ab[t].sqrt()).abs()>1).float().mean()),
        shared_posterior_passed=all(checks[k]["passed"] for k in (
            "shared_x0_mean", "shared_x0_std", "shared_x0_unit_noise_next")),
        fp64_algebra_passed=checks["fp64_tweedie_algebra"]["passed"])


def main():
    started=time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run-dir", required=True,
        help="Completed first_guidance run; ONLY its raw_baseline output is reused")
    parser.add_argument("--sampler-dir", default=None)
    parser.add_argument("--dps-root", default=str(ROOT/"DPS"))
    parser.add_argument("--tensor-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    first_dir = Path(args.first_run_dir).resolve()
    first_path = first_dir/"run.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    guidance_dir = Path(first["source_guidance"])
    guidance_path = guidance_dir/"audit.json"
    guidance = json.loads(guidance_path.read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"])
    trace = json.loads((trace_dir/"audit.json").read_text(encoding="utf-8"))
    source_dir = Path(trace["source"])
    source = json.loads((source_dir/"run.json").read_text(encoding="utf-8"))
    from utils.diag_zaps_first_direction import validate_first_run
    from utils.diag_zaps_parameter_gradient import validate_protocol, PROBE_TIMES, DELTAS
    validate_protocol(guidance, trace, source, PROBE_TIMES, DELTAS)
    validate_first_run(first, guidance, trace_dir)
    config = trace["results"][ARMS[0]]["config"]
    original_config(config)
    if first["config"] != config or source["arguments"]["seed"] != 1001:
        raise ValueError("Wrong original parent settings/seed")
    if source["task_configs"]["gaussian_deblur"] != dict(kernel_size=61,sigma=3.,noise_sigma=.05):
        raise ValueError("Original Gaussian operator config changed")
    sampler_path = Path(args.sampler_dir)/"audit.json" if args.sampler_dir else trace_dir/"sampler_parity_20261008_220150/audit.json"
    old_sampler = json.loads(sampler_path.read_text(encoding="utf-8"))
    historical_sampler = old_sampler_scope(old_sampler)
    if Path(old_sampler["arguments"]["trace_dir"]).resolve() != trace_dir.resolve():
        raise ValueError("Old sampler is from another trace")
    output_dir = trace_dir/("original_reference_audit_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/"experiment_cards/R_original_reference.md").read_text(encoding="utf-8")
    (output_dir/"experiment_card.md").write_text(card, encoding="utf-8")
    record = dict(experiment="original_imagenet_reference_cached_audit", status="running",
        arguments=vars(args), source_first_run=str(first_path), source_guidance=str(guidance_path),
        source_sampler=str(sampler_path), config=config, method_nfe=0, diagnostic_nfe=0,
        model_loaded=False, optimizer_steps=0, reconstruction_reruns=0, gates={}, probes={},
        historical_sampler=historical_sampler, experiment_card=card,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        limits="cached-point mechanics and cached-output integrity only; not whole unroll-gradient or author parity")
    def save():
        record["elapsed_seconds"]=time.perf_counter()-started
        (output_dir/"audit.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save()
    print(f"Original settings only; NO model load/training/sampling; new UNet NFE=0.\nRecords: {output_dir}", flush=True)
    try:
        import torch
        from configs.config import METRICS_CONFIG
        from modules import zaps_algorithm as core
        from modules.diffusion_model import _precompute_diffusion_params
        from modules.main_single import load_image_as_tensor
        from modules.degradations import get_operator
        from utils.diag_dps_same_observation import file_sha256
        from utils.diag_zaps_sampler_parity import load_dps_reference, error_metrics
        from utils.diag_zaps_first_guidance import IDENTITIES
        from utils.metrics import compute_all_metrics
        from utils.diag_zaps_first_guidance import validate_parent
        manifest = json.loads((PROJECTS/"IMAGENET_BASELINE_MANIFEST_20261013.json").read_text(encoding="utf-8"))
        if file_sha256(guidance_path) != first["source_vjp_sha256"]:
            raise RuntimeError("Guidance parent JSON changed")
        parameter_path = Path(first["resolved_parameter_run"])
        parameter = json.loads(parameter_path.read_text(encoding="utf-8"))
        validate_parent(parameter, guidance, guidance_dir)
        if file_sha256(parameter_path) != first["parent_run_sha256"]:
            raise RuntimeError("Completed parameter audit parent changed")
        dps_baseline = json.loads((Path(guidance["identity_baseline"])/"run.json").read_text(encoding="utf-8"))
        if dps_baseline.get("status") != "complete":
            raise RuntimeError("Identity DPS parent incomplete")
        paths = {"checkpoint_sha256": Path(dps_baseline["checkpoint_path"]),
                 "measurement_sha256": source_dir/"gaussian_deblur/measurement.pt",
                 "image_sha256": Path(source["arguments"]["image"]),
                 "core_sha256": PROJECTS/"modules/zaps_algorithm.py",
                 "operator_source_sha256": PROJECTS/"modules/degradations.py"}
        print("Checking existing checkpoint/observation/source hashes (local reads, no download)...", flush=True)
        for key,path in paths.items():
            if file_sha256(path) != guidance[key] or guidance[key] != first["identity"][key]:
                raise RuntimeError(f"Original identity mismatch: {key}")
        snapshot_path = trace_dir/(ARMS[0]+"_last_unroll_state.pt")
        if file_sha256(snapshot_path) != first["snapshot_sha256"]:
            raise RuntimeError("Original PRE-update snapshot changed")
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        if (snapshot["tau"].tolist() != manifest["archived_reference"]["actual_timesteps_ascending"]
            or snapshot["zeta"].tolist() != trace["results"][ARMS[0]]["epoch_used_zeta_ascending"]):
            raise RuntimeError("Original parameter/grid snapshot mismatch")
        record.update(identity={k: guidance[k] for k in IDENTITIES},
            current_source_sha256={name:file_sha256(PROJECTS/name) for name in (
                "modules/diffusion_model.py","modules/wavelet.py","modules/main_single.py","utils/metrics.py")},
            current_source_limits="Unrecorded historical helper hashes cannot be certified retroactively",
            source_first_sha256=file_sha256(first_path), source_sampler_sha256=file_sha256(sampler_path))
        record["gates"]["source_identity"] = dict(passed=True)
        dps, dps_meta = load_dps_reference(args.dps_root)
        if set(dps_meta["source_sha256"].values()) != set(old_sampler["reference"]["source_sha256"].values()):
            raise RuntimeError("Selected DPS reference differs from the historical source")
        for path,digest in old_sampler["reference"]["source_sha256"].items():
            if file_sha256(Path(path)) != digest:
                raise RuntimeError("Vendored DPS source differs from the old sampler audit")
        record["dps_reference"] = dps_meta
        ab = _precompute_diffusion_params(1000)["alphas_cumprod"].to(args.tensor_device)
        all_points_ok = True
        for name in ARMS:
            arm = trace["results"][name]
            original_config(arm["config"])
            grid = [r["t"] for r in reversed(arm["rows"])]
            reference = dps.DDPM(use_timesteps=grid, betas=dps.get_named_beta_schedule("linear",1000),
                model_mean_type="epsilon", model_var_type="fixed_small", dynamic_threshold=False,
                clip_denoised=True, rescale_timesteps=False)
            probe_path = guidance_dir/(name+"_probe_inputs.pt")
            probes = torch.load(probe_path, map_location=args.tensor_device, weights_only=True)
            if (len(probes) != len(guidance["results"][name]["rows"])
                or [p["t"] for p in probes] != [r["t"] for r in guidance["results"][name]["rows"]]
                or probes[0]["t"] != 999 or probes[-1]["t"] != 0):
                raise RuntimeError("Incomplete/wrong cached probes")
            rows=[]
            before_rng = torch.get_rng_state().clone()
            for probe in probes:
                if probe["x"].shape != snapshot["init_noise"].shape:
                    raise RuntimeError("Cached probe shape differs from original image")
                row = cached_sampler_checks(probe, grid, ab, reference, core)
                old_row=next(r for r in arm["rows"] if r["t"]==row["t"])
                if not (math.isclose(row["input_x_rms"],old_row["input_x_rms"],rel_tol=2e-5,abs_tol=1e-6)
                    and math.isclose(row["raw_x0_clip_fraction"],old_row["raw_x0_clip_fraction"],rel_tol=2e-5,abs_tol=2e-5)):
                    raise RuntimeError("Cached input/clipping fingerprint differs from archived path")
                rows.append(row)
                all_points_ok &= row["shared_posterior_passed"] and row["fp64_algebra_passed"]
                print(f"{name} t={row['t']:3d}: shared posterior={row['shared_posterior_passed']} "
                      f"FP64 algebra={row['fp64_algebra_passed']} FP32 x0 error="
                      f"{row['checks']['fp32_x0_original_tolerance']['max_absolute_error']:.3e}", flush=True)
            if not torch.equal(before_rng, torch.get_rng_state()):
                raise RuntimeError("Cached mechanics unexpectedly consumed CPU RNG")
            record["probes"][name] = dict(rows=rows, current_probe_sha256=file_sha256(probe_path),
                probe_identity_limits="No save-time hash in old audit; first x_T checked below, other inputs inherit old replay/probe evidence")
            if not torch.equal(probes[0]["x"].cpu(), snapshot["init_noise"]):
                raise RuntimeError("Cached first input is not the paired original x_T")
            save()
        if not all_points_ok:
            raise RuntimeError("Shared posterior/algebra check failed; no baseline closure")
        record["gates"]["cached_mechanics"] = dict(passed=True, scope="14 stored states, shared-x0 posterior + FP64 algebra; NOT old FP32 x0 overall PASS")
        gt = load_image_as_tensor(source["arguments"]["image"]).to(args.tensor_device)
        y = torch.load(paths["measurement_sha256"], map_location=args.tensor_device, weights_only=True)
        output_path = first_dir/"raw_baseline.pt"
        output = torch.load(output_path, map_location=args.tensor_device, weights_only=True)
        raw = next(r for r in first["results"] if r["policy"] == "raw_baseline")
        metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        expected = trace["results"][ARMS[0]]["metrics"]
        if not all(math.isclose(metrics[k], raw[k], rel_tol=2e-5, abs_tol=1e-6)
            and math.isclose(metrics[k], expected[k], rel_tol=2e-5, abs_tol=1e-6) for k in metrics):
            raise RuntimeError("Original cached reconstruction metric mismatch")
        operator = get_operator("gaussian_deblur", device=args.tensor_device, **source["task_configs"]["gaussian_deblur"])
        with torch.no_grad():
            residual = y-operator.H(output)
            loss = float(residual.double().square().mean())
            noise = y-operator.H(gt)
            noise_std = float(noise.double().std(unbiased=False))
        if not math.isclose(loss, raw["loss"], rel_tol=2e-5, abs_tol=1e-8):
            raise RuntimeError("Original terminal measurement loss differs from completed replay")
        record["gates"]["cached_output_metrics_and_loss"] = dict(passed=True)
        record["baseline"] = dict(task="gaussian_deblur", image=source["arguments"]["image"],seed=1001,
            **metrics, final_mse=loss, final_residual=float(residual.double().norm()),
            historical_method_nfe=300, new_method_nfe=0, output_current_sha256=file_sha256(output_path),
            output_hash_limits="Current hash; no original save-time tensor hash, metrics/loss/fingerprints establish compatibility")
        record["measurement_domain"] = dict(model_domain=[-1,1], configured_sigma=float(operator.noise.sigma),
            empirical_sigma_model_domain=noise_std, equivalent_unit_domain_sigma=noise_std/2,
            paper_reported_unit_domain_sigma=.05,
            status="protocol_alignment_open; do not call completely paper-identical")
        record["reused_parameter_audit"] = dict(path=str(parameter_path),
            result=parameter["summary"],
            limits="No new derivatives; previous scale disagreements remain unvalidated")
        record["runtime"]=dict(torch=str(torch.__version__),python=sys.version,
            tensor_device=args.tensor_device, cuda_runtime=torch.version.cuda)
        record["decision"] = dict(classification="ORIGINAL_LOCAL_CACHE_CONFIRMED_WITH_OPEN_ALIGNMENT_ITEMS",
            production_fp32_sampler_overall_pass_not_claimed=True, author_parity=False,
            benchmark_reproduced=False, tuned_baseline_adopted=False,
            next_action="Review domain/gradient/undocumented conventions; no tuning or GPU reruns automatically unlocked")
        record["status"]="complete"
        save()
    except Exception as error:
        record["status"], record["error"]="failed",repr(error)
        save()
        raise
    b=record["baseline"]
    print("\n=== Original ImageNet local-reference confirmation ===",flush=True)
    print(f"Gaussian original: PSNR={b['psnr']:.4f} SSIM={b['ssim']:.4f} LPIPS={b['lpips']:.4f}",flush=True)
    print("Initial zeta=.2 ALL steps; D=.2; lr=.001; 30x10; state OFF; source settings unchanged.",flush=True)
    print("New UNet NFE=0; no model load/optimization/reconstruction rerun.",flush=True)
    print("Measurement domain:",json.dumps(record["measurement_domain"]),flush=True)
    print("Decision:",json.dumps(record["decision"]),flush=True)
    print(f"Records: {output_dir}",flush=True)


if __name__=="__main__":
    main()
