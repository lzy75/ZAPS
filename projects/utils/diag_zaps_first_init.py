"""Last bounded ImageNet check: lower ONLY first zeta initialization.

Original reverse/Adam equations, 30x10 joint zeta/D, no new final sampling.
Reuse completed prior baselines. This is initialization tuning, NOT paper parity.
"""
import argparse
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


def initial_value(original, factor):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (original, factor)):
        raise ValueError("Need positive finite original initialization and source norm factor")
    if factor >= 1 or original*factor <= 1e-6:
        raise ValueError("This final card requires a lower, interior first initialization")
    return original*factor


def validate_direction_parent(record, first_path, first):
    from utils.diag_zaps_first_guidance import IDENTITIES
    if (record.get("status") != "complete" or record.get("new_nfe") != 31
            or record.get("experiment") != "first_direction_fixed_norm_frozen_path"
            or record.get("optimizer_steps") != 0 or record.get("retraining") is not False
            or Path(record.get("source_first_run", "")).resolve() != Path(first_path).resolve()
            or record.get("config") != first["config"]
            or record.get("identity") != {k: first["identity"][k] for k in IDENTITIES}):
        raise ValueError("Need the matching completed direction experiment")
    required = ("cached_terminal_loss", "first_input_fingerprint", "calibrated_first_next_rms",
                "pairing", "equal_correction_norm", "replay_first_input", "cached_metrics")
    if not all(record.get("gates", {}).get(k, {}).get("passed") is True for k in required):
        raise ValueError("Direction-parent gates incomplete/failed")


def direction_parent(trace_dir, first_path, first, explicit=None):
    paths = [Path(explicit).resolve()/"run.json"] if explicit else sorted(
        Path(trace_dir).glob("first_direction_*/run.json"), reverse=True)
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_direction_parent(record, first_path, first)
        except (ValueError, KeyError, TypeError):
            if explicit:
                raise
            continue
        return path, record
    raise ValueError("No matching completed direction run; do not repeat unfinished experiments")


def interpret(metrics, original, frozen):
    d_raw = {k: metrics[k]-original[k] for k in ("psnr", "ssim", "lpips")}
    d_frozen = {k: metrics[k]-frozen[k] for k in ("psnr", "ssim", "lpips")}
    positive = d_raw["psnr"] >= .2 and d_raw["ssim"] >= .005 and d_raw["lpips"] <= -.005
    retained = d_frozen["psnr"] >= -.1 and d_frozen["ssim"] >= -.005 and d_frozen["lpips"] <= .005
    if positive and retained:
        classification = "SUPPORTS_FIRST_INITIALIZATION_SENSITIVITY"
    elif positive:
        classification = "PARTIAL_GAIN_NOT_RETAINING_FROZEN_BENEFIT"
    else:
        classification = "NO_COMPREHENSIVE_INITIALIZATION_GAIN"
    return {"classification": classification, "deltas_vs_original": d_raw,
            "deltas_vs_frozen_norm": d_frozen,
            "next_action": "Stop this single-image diagnostic round; discuss freezing a labelled local reference and state-aware work",
            "limits": "not proof of author parity, benchmark mean reproduction, or impossibility of reproduction"}


def rng_preflight(seed, state, device, saved_index):
    """0 score calls: reconstruct the fixed x_T and first nine epoch draws."""
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state
    set_seed(seed)
    fixed = torch.randn(*state["init_noise"].shape, device=device)
    x_equal = torch.equal(fixed.cpu(), state["init_noise"])
    for _ in range(9*29):
        torch.randn_like(fixed)
    epoch10 = rng_state(device)
    current_index = torch.device(device).index
    current_index = torch.cuda.current_device() if current_index is None else current_index
    gate = {"x_T_equal": x_equal, "epoch10_cpu_rng_equal": torch.equal(epoch10["cpu"], state["cpu_rng"]),
            "epoch10_sampling_rng_equal": torch.equal(epoch10["cuda"][current_index], state["cuda_rng"][saved_index]),
            "diagnostic_draws": 1+10*29, "score_nfe": 0}
    gate["passed"] = all(gate[k] for k in ("x_T_equal", "epoch10_cpu_rng_equal", "epoch10_sampling_rng_equal"))
    for _ in range(29):
        torch.randn_like(fixed)
    return gate, rng_state(device)


def make_observed_training(snapshot_class, record, save):
    """Passive tracing around the original reverse graph; no training override."""
    import torch
    from utils.diag_zaps_clamp_ablation import clamp_guidance_scope

    class ObservedTraining(snapshot_class):
        def _reverse_diffusion(self, measurement, *positional, **kwargs):
            rows = []
            used = float(self.zeta[-1].detach())
            self._clamp_audit_y = measurement.detach()
            with clamp_guidance_scope(self, False, rows):
                output = super()._reverse_diffusion(measurement, *positional, **kwargs)
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite training output")
            record["epochs"].append({"epoch": self.audit_calls, "first_zeta_used": used,
                "next_rms": rows[1]["input_x_rms"], "next_clip_fraction": rows[1]["raw_x0_clip_fraction"]})
            save()
            return output
    return ObservedTraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run-dir", required=True)
    parser.add_argument("--direction-run-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    args = parser.parse_args()
    if not str(args.device).startswith("cuda"):
        parser.error("Only archived CUDA ImageNet Gaussian path is supported")
    from utils.diag_zaps_first_direction import validate_first_run, model_rng_guard
    from utils.diag_zaps_first_guidance import norm_calibration, validate_parent, IDENTITIES
    from utils.diag_zaps_parameter_gradient import validate_protocol, PROBE_TIMES, DELTAS, write_csv
    first_dir = Path(args.first_run_dir).resolve()
    first_path = first_dir/"run.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    guidance_dir = Path(first["source_guidance"]).resolve()
    guidance = json.loads((guidance_dir/"audit.json").read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"]).resolve()
    saved = json.loads((trace_dir/"audit.json").read_text(encoding="utf-8"))
    source_dir = Path(saved["source"]).resolve()
    source = json.loads((source_dir/"run.json").read_text(encoding="utf-8"))
    validate_protocol(guidance, saved, source, PROBE_TIMES, DELTAS)
    validate_first_run(first, guidance, trace_dir)
    parameter_path = Path(first["resolved_parameter_run"])
    parameter = json.loads(parameter_path.read_text(encoding="utf-8"))
    validate_parent(parameter, guidance, guidance_dir)
    direction_path, direction = direction_parent(trace_dir, first_path, first, args.direction_run_dir)
    config = saved["results"]["irregular_15_10_5"]["config"]
    if config != first["config"] or config != parameter["config"]:
        raise ValueError("Keep original configuration")
    calibration = norm_calibration(guidance)
    first_init = initial_value(config["zeta_init"], calibration["factor"])
    output_dir = trace_dir/("first_init_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS_ROOT/"experiment_cards/R_first_init.md").read_text(encoding="utf-8")
    record = {"experiment": "first_zeta_initialization_joint_training", "status": "running", "arguments": vars(args),
              "source_first_run": str(first_path), "source_direction_run": str(direction_path),
              "source_trace": str(trace_dir), "config": config, "first_zeta_init": first_init,
              "calibration": calibration, "new_nfe": 0, "planned_nfe": 300,
              "planned_optimizer_steps": 10, "optimizer_steps": None,
              "final_mode": "last_opt_pre_final_update", "epochs": [], "gradients": [], "results": [], "gates": {},
              "experiment_card": card, "gt_usage": "only evaluation after training; no selection/stop",
              "limits": "initialization tuning, not original-paper setting or certified reproduction",
              "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                  text=True, capture_output=True, check=True).stdout.strip()}

    def save():
        (output_dir/"run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_csv(output_dir/"summary.csv", record["results"])
        write_csv(output_dir/"epochs.csv", record["epochs"])

    (output_dir/"experiment_card.md").write_text(card, encoding="utf-8")
    save()
    print("=== Final bounded ImageNet initialization check ===", flush=True)
    print(f"ONLY zeta(t=999) init: {config['zeta_init']} -> {first_init:.9g}; original formulas; joint zeta+D; lr=.001; 30x10", flush=True)
    print("No scaling scope, exact-direction replacement, noise change, GT selection or extra sample.", flush=True)
    print(f"Records: {output_dir}", flush=True)
    try:
        import torch
        from configs.config import IMG_SIZE, METRICS_CONFIG
        from modules.main_single import load_diffusion_model, load_image_as_tensor
        from modules.degradations import get_operator
        from utils.diag_dps_same_observation import file_sha256
        from utils.diag_learning_rate_ablation import set_seed
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection
        from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.metrics import compute_all_metrics
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if (file_sha256(first_path) != direction["parent_run_sha256"]
                or file_sha256(parameter_path) != first["parent_run_sha256"]
                or file_sha256(guidance_dir/"audit.json") != first["source_vjp_sha256"]):
            raise RuntimeError("Completed parent JSON chain changed")
        measurement = source_dir/"gaussian_deblur/measurement.pt"
        snapshot = trace_dir/"irregular_15_10_5_last_unroll_state.pt"
        for key, path in (("measurement_sha256", measurement), ("core_sha256", PROJECTS_ROOT/"modules/zaps_algorithm.py"),
                          ("operator_source_sha256", PROJECTS_ROOT/"modules/degradations.py"),
                          ("image_sha256", Path(source["arguments"]["image"]))):
            if file_sha256(path) != guidance[key]:
                raise RuntimeError(f"Identity changed: {key}")
        if file_sha256(snapshot) != first["snapshot_sha256"]:
            raise RuntimeError("Archived snapshot changed")
        print("Loading original checkpoint (no download)...", flush=True)
        model = load_diffusion_model("imagenet", args.device)
        model.model.eval()
        if (file_sha256(model.ckpt_path) != guidance["checkpoint_sha256"] or model.num_steps != 1000
                or any(p.dtype != torch.float32 for p in model.model.parameters())):
            raise RuntimeError("Keep original fp32 model")
        old_state = torch.load(snapshot, map_location="cpu", weights_only=True)
        mapping = saved_cuda_selection(old_state, saved["arguments"].get("device"), args.saved_cuda_index)
        operator = get_operator("gaussian_deblur", device=args.device, **source["task_configs"]["gaussian_deblur"])
        y = torch.load(measurement, map_location=args.device, weights_only=True)
        if tuple(y.shape) != (1, 3, 256, 256) or y.dtype != torch.float32 or not torch.isfinite(y).all():
            raise RuntimeError("Observation mismatch")

        ObservedTraining = make_observed_training(LastUnrollSnapshotZAPS, record, save)

        seed = int(source["arguments"]["seed"])
        set_seed(seed)
        zaps = ObservedTraining(model, operator, img_size=IMG_SIZE[0], **config)
        if not torch.equal(zaps.tau.cpu(), old_state["tau"]):
            raise RuntimeError("Constructed original grid differs")
        with torch.no_grad():
            original_zeta = zaps.zeta.detach().clone()
            zaps.zeta[-1] = first_init
        if torch.nonzero(zaps.zeta != original_zeta).flatten().tolist() != [29]:
            raise RuntimeError("Initialization changed another timestep")
        preflight, expected_end = rng_preflight(seed, old_state, args.device, mapping["saved_sampling_index"])
        record["gates"]["zero_nfe_noise_preflight"] = preflight
        record.update(identity={k: guidance[k] for k in IDENTITIES}, rng_mapping=mapping,
            parent_direction_sha256=file_sha256(direction_path), parent_first_sha256=file_sha256(first_path),
            current_supporting_source_sha256={name: file_sha256(PROJECTS_ROOT/name) for name in (
                "modules/diffusion_model.py", "modules/wavelet.py", "modules/main_single.py",
                "modules/dataset_loader.py", "configs/config.py", "utils/metrics.py")},
            supporting_source_limits="Current fingerprints only; historical parents did not certify these files",
            runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
                     "python": sys.version,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "visible_cuda_count": torch.cuda.device_count()})
        save()
        if not preflight["passed"]:
            raise RuntimeError("Original training noise prefix cannot be reconstructed; stop before 300 NFE")
        visited = []
        def count(module, inputs):
            record["new_nfe"] += 1
            visited.append(int(inputs[1].flatten()[0]))
        original_clip = torch.nn.utils.clip_grad_norm_
        def clip(parameters, max_norm, **kwargs):
            parameters = list(parameters)
            if len(parameters) != 2 or parameters[0] is not zaps.zeta or parameters[1] is not zaps.D or max_norm != 1:
                raise RuntimeError("Optimizer clipping contract changed")
            before_first = float(zaps.zeta.grad[-1].detach())
            total = original_clip(parameters, max_norm, **kwargs)
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite training gradients")
            record["gradients"].append({"epoch": len(record["gradients"])+1, "first_grad_before_clip": before_first,
                "joint_grad_norm_before_clip": float(total), "clip_active": float(total) > 1+1e-6})
            return total
        hook = model.model.register_forward_pre_hook(count)
        set_seed(seed)
        started = time.perf_counter()
        try:
            with model_rng_guard(model.model, args.device) as guarded:
                with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=clip):
                    losses = zaps.optimize(y, verbose=True, x0_gt=None)
        finally:
            hook.remove()
        record["optimizer_steps"] = len(losses)
        new_state = zaps.audit_state
        if new_state is None or zaps.audit_calls != 10 or len(record["epochs"]) != 10 or len(losses) != 10:
            raise RuntimeError("Missing last-unroll snapshot or incorrect optimization count")
        current_index = torch.device(args.device).index
        current_index = torch.cuda.current_device() if current_index is None else current_index
        pairing = {"x_T_equal": torch.equal(new_state["init_noise"], old_state["init_noise"]),
            "epoch10_cpu_rng_equal": torch.equal(new_state["cpu_rng"], old_state["cpu_rng"]),
            "epoch10_sampling_rng_equal": torch.equal(new_state["cuda_rng"][current_index], old_state["cuda_rng"][mapping["saved_sampling_index"]]),
            "end_rng_equal": same_rng(expected_end, rng_state(args.device)),
            "times_equal": visited == old_state["tau"].flip(0).tolist()*10,
            "model_rng_guarded_calls": len(guarded)}
        pairing["passed"] = all(v is True for k,v in pairing.items() if k != "model_rng_guarded_calls") and len(guarded) == 300
        record["gates"]["training_pairing"] = pairing
        record["loss_history"], record["seconds"] = losses, time.perf_counter()-started
        save()
        if not pairing["passed"] or record["new_nfe"] != 300 or zaps._last_nfe != 300 or len(record["gradients"]) != 10:
            raise RuntimeError("Training pairing/budget failed; no quality interpretation")
        if any(p.grad is not None for p in model.model.parameters()):
            raise RuntimeError("Unexpected model weight gradients")
        output = zaps._last_opt_x0.detach().cpu()
        torch.save(new_state, output_dir/"last_unroll_state.pt")
        torch.save(output, output_dir/"first_init_last_opt.pt")
        record["output_sha256"] = file_sha256(output_dir/"first_init_last_opt.pt")
        record["last_unroll_state_sha256"] = file_sha256(output_dir/"last_unroll_state.pt")
        gt = load_image_as_tensor(source["arguments"]["image"]).cpu()
        # Confirm the cached reference metrics; no old 30-step trajectory rerun.
        for old in first["results"]:
            cached = torch.load(first_dir/(old["policy"]+".pt"), map_location="cpu", weights_only=True)
            metrics = compute_all_metrics(cached, gt, lpips_net=METRICS_CONFIG["lpips_net"])
            if not all(math.isclose(metrics[k], old[k], rel_tol=2e-5, abs_tol=1e-6) for k in metrics):
                raise RuntimeError("Cached reference metric mismatch")
            record["results"].append({"policy": "cached_"+old["policy"], **metrics, "new_nfe": 0,
                "final_measurement_mse": None, "final_residual_norm": None})
        metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        record["results"].append(dict(policy="first_init_joint_training", **metrics, new_nfe=300,
            final_measurement_mse=losses[-1], final_residual_norm=math.sqrt(losses[-1]*y.numel())))
        record["decision"] = interpret(metrics, first["results"][0], first["results"][1])
        record["used_first_zeta_last_opt"] = float(new_state["zeta"][-1])
        record["after_update_first_zeta"] = float(zaps.zeta[-1].detach())
        record["after_update_zeta_ascending"] = zaps.zeta.detach().cpu().tolist()
        record["d_delta_rms"] = float((zaps.D.detach()-config["d_init"]).double().square().mean().sqrt())
        record["gates"]["cached_metrics"] = {"passed": True, "limits": "compatibility, not historical tensorwise certification"}
        record["status"] = "complete"
        save()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save()
        raise
    print("\n=== Final first-initialization summary ===", flush=True)
    print(f"{'policy':>30} {'PSNR':>9} {'dRaw':>9} {'dFrozen':>9} {'SSIM':>8} {'LPIPS':>8} {'newNFE':>7}", flush=True)
    raw, frozen = record["results"][:2]
    for r in record["results"]:
        print(f"{r['policy']:>30} {r['psnr']:9.4f} {r['psnr']-raw['psnr']:+9.4f} "
              f"{r['psnr']-frozen['psnr']:+9.4f} {r['ssim']:8.4f} {r['lpips']:8.4f} {r['new_nfe']:7d}", flush=True)
    print("Decision:", json.dumps(record["decision"], ensure_ascii=False), flush=True)
    print(f"First zeta: init={first_init:.6g}; last_opt used={record['used_first_zeta_last_opt']:.6g}", flush=True)
    print(f"New NFE={record['new_nfe']}; records: {output_dir}", flush=True)
    print("This closes the bounded single-image diagnostic round, not the possibility of future paper reproduction.", flush=True)


if __name__ == "__main__":
    main()
