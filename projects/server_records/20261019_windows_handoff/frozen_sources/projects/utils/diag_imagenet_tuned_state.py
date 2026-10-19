"""Fixed-30 state test on the completed first-zeta-tuned Gaussian reference.

Independent entry point. Original core/defaults/FFHQ/batch remain unchanged.
90 diagnostic NFE for fixed-repeat/null VJP parity, then 2x300 method NFE.
"""
import argparse
from bisect import bisect_left
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
ROOT = PROJECTS.parent
sys.path.insert(0, str(PROJECTS))
from modules.adaptive_scheduler import BudgetedSchedulerConfig, BudgetedStateAwareScheduler


def interpolate(coordinates, values, u):
    k = bisect_left(coordinates, u)
    if k == 0:
        return values[0]
    if k == len(coordinates):
        return values[-1]
    if abs(coordinates[k]-u) < 1e-12:
        return values[k]
    a, b = values[k-1:k+1]
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    w = (u-coordinates[k-1])/(coordinates[k]-coordinates[k-1])
    return a+(b-a)*w


class NoiseProfileScheduler(BudgetedStateAwareScheduler):
    """Existing physical-veto controller; pilot references live on log-sigma u.

    zeta/D remain attached to the SAME 30 reverse-call positions. No new
    parameter interpolation, weight multiplier, sampler or variable budget.
    """
    def __init__(self, grid, alpha_bar, response):
        if len(alpha_bar) != 1000 or any(not 0 < a < 1 for a in alpha_bar):
            raise ValueError("Need the actual 1000-step alpha_bar strictly inside (0,1)")
        logs = [.5*(math.log1p(-a)-math.log(a)) for a in alpha_bar]
        if any(a >= b for a,b in zip(logs, logs[1:])):
            raise ValueError("Noise coordinate must be strictly ordered")
        self.u_by_t = [(logs[-1]-v)/(logs[-1]-logs[0]) for v in logs]
        self.anchors = [self.u_by_t[t] for t in grid]
        self.pilot_log_r = [float("nan")]*len(grid)
        self.pilot_cos = [float("nan")]*len(grid)
        super().__init__(grid, BudgetedSchedulerConfig(
            response_strength=response, residual_mode="reference_profile",
            profile_warmup_epochs=1, profile_center_decay=.8,
            profile_residual_scale=.15, profile_cosine_scale=.2,
            profile_cosine_gate=.25, profile_gate_mode="veto_only",
            profile_cosine_feature_mode="global", weight_mode="identity",
            mod_min=.8, mod_max=1.2))

    def reset(self):
        super().reset()
        self.current_t = None
        self.previous_u = None
        self.current_u = None

    def set_time(self, t):
        if not 0 <= t < 1000:
            raise ValueError("Invalid model time")
        self.current_t = int(t)

    def _update_reference_profile(self, resid_norm, cos_x0=float("nan")):
        if self.current_t is None or not math.isfinite(resid_norm) or resid_norm < 0:
            raise RuntimeError("Missing actual model time or invalid residual")
        u, k = self.u_by_t[self.current_t], self.used
        c = self.cfg
        if self.previous_u is not None and u <= self.previous_u:
            raise RuntimeError("Noise position must advance strictly")
        log_r = math.log(max(resid_norm, 1e-12))
        warmup = self._completed_unrolls < 1
        if warmup:
            if self.current_t != self.nominal_timesteps[k]:
                raise RuntimeError("Pilot must follow the fixed nominal grid")
            self.pilot_log_r[k], self.pilot_cos[k] = log_r, float(cos_x0)
            reference, ratio, er, ec, q = resid_norm, 0., 0., 0., 1.
        else:
            reference_log = interpolate(self.anchors, self.pilot_log_r, u)
            if not math.isfinite(reference_log):
                raise RuntimeError("Incomplete pilot profile")
            reference, ratio = math.exp(reference_log), log_r-reference_log
            if self._profile_log_ratio_ema is None:
                er = 0.
                self._profile_log_ratio_ema = ratio
            else:
                er = math.tanh((ratio-self._profile_log_ratio_ema)/c.profile_residual_scale)
                # .8 per mean pilot interval, not .8 regardless of actual jump.
                decay = c.profile_center_decay**((u-self.previous_u)*(self.N-1))
                self._profile_log_ratio_ema = decay*self._profile_log_ratio_ema+(1-decay)*ratio
            reference_cos = interpolate(self.anchors, self.pilot_cos, u)
            ec = math.tanh((reference_cos-cos_x0)/c.profile_cosine_scale) if all(
                math.isfinite(v) for v in (reference_cos, cos_x0)) else 0.
            q = 1.-c.profile_cosine_gate*max(0., -(1. if er > 0 else -1. if er < 0 else 0.)*ec)
        drop = (self._prev_r-resid_norm)/self._prev_r if self._prev_r and self._prev_r > 1e-12 else float("nan")
        self._prev_r = resid_norm
        self.previous_u, self.current_u = u, u
        self.residual_norm, self.relative_residual_drop = resid_norm, drop
        self.smoothed_relative_residual_drop = drop
        self.residual_baseline, self.residual_scale, self.residual_zscore = reference, c.profile_residual_scale, er
        self.cosine_x0 = cos_x0
        self.residual_error, self.cosine_error = .5+.5*er, .5+.5*ec
        self.state_score = .5+.5*er*q
        self.profile_warmup, self.profile_log_ratio = warmup, ratio
        self.profile_residual_signal, self.profile_cosine_signal, self.profile_confidence = er, ec, q
        self.current_t = None  # require a new model call before next update

    def snapshot(self):
        return {**super().snapshot(), "noise_u": self.current_u, "reference_alignment": "log_sigma_u"}


def validate_parent(record):
    config = record.get("config", {})
    required = {"num_steps": 30, "num_epochs": 10, "lr": .001, "zeta_init": .2,
                "d_init": .2, "eta": 1., "use_learned_var": False, "sampler_mode": "ddpm",
                "wave": "db4", "level": 3}
    if (record.get("status") != "complete" or record.get("experiment") != "first_zeta_initialization_joint_training"
        or record.get("new_nfe") != 300 or record.get("optimizer_steps") != 10
        or record.get("final_mode") != "last_opt_pre_final_update"
        or any(config.get(k) != v for k,v in required.items())
        or config.get("surrogate_score_jacobian", False)
        or not math.isclose(record.get("first_zeta_init", 0), .04002983177868308, rel_tol=1e-10)):
        raise ValueError("Need the completed agreed first-initialization run, not a tuned variant")
    for gate in ("zero_nfe_noise_preflight", "training_pairing", "cached_metrics"):
        if record.get("gates", {}).get(gate, {}).get("passed") is not True:
            raise ValueError("Parent gate failed/incomplete")
    if record.get("decision", {}).get("classification") != "SUPPORTS_FIRST_INITIALIZATION_SENSITIVITY":
        raise ValueError("This experiment requires the supported initialization parent")
    if sum(r.get("policy") == "first_init_joint_training" for r in record.get("results", [])) != 1:
        raise ValueError("Parent result missing/duplicated")


def find_parent(trace_dir, explicit=None):
    paths = [Path(explicit)/"run.json"] if explicit else sorted(Path(trace_dir).glob("first_init_*/run.json"), reverse=True)
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_parent(record)
            if Path(record["source_trace"]).resolve() != Path(trace_dir).resolve():
                raise ValueError("Different archived trace")
        except (ValueError, KeyError, TypeError):
            if explicit:
                raise
            continue
        return path.resolve(), record
    raise ValueError("No compatible completed first_init parent; do not repeat it")


def parity_decision(repeat, cross, metadata):
    caps = {"output": 1e-6, "zeta_grad": 1e-4, "D_grad": 1e-4}
    limits = {k: min(caps[k], max(1e-7 if k == "output" else 1e-6, 3*repeat[k])) for k in caps}
    good = all(math.isfinite(v) for d in (repeat, cross) for v in d.values())
    passed = good and metadata and all(repeat[k] <= caps[k] and cross[k] <= limits[k] for k in caps)
    return dict(passed=passed, repeat=repeat, adaptive_null=cross, limits=limits, hard_caps=caps,
                metadata_equal=metadata, scope="one initial unroll forward/VJP only; not full optimization parity")


def interpret(null, state, parent):
    delta = {k: state[k]-null[k] for k in ("psnr", "ssim", "lpips")}
    drift = {k: null[k]-parent[k] for k in delta}
    compatible = abs(drift["psnr"]) <= .05 and abs(drift["ssim"]) <= .005 and abs(drift["lpips"]) <= .005
    if not compatible:
        tag = "INCONCLUSIVE_NULL_DRIFT"
    elif delta["psnr"] >= .1 and delta["ssim"] >= -.002 and delta["lpips"] <= .002:
        tag = "SINGLE_IMAGE_STATE_GAIN_CANDIDATE"
    elif delta["psnr"] > 0 and (delta["ssim"] < -.002 or delta["lpips"] > .002):
        tag = "PSNR_ONLY_TRADEOFF"
    else:
        tag = "NO_CLEAR_COMPREHENSIVE_STATE_GAIN"
    return dict(classification=tag, deltas_vs_null=delta, null_drift_vs_parent=drift,
                limits="single image; engineering thresholds, not statistical proof or paper reproduction")


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def save_record(directory, record):
    (directory/"run.json").write_text(json.dumps(finite(record), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    rows = record["results"]
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (directory/"summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--init-run-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("This entry uses the archived CUDA ImageNet Gaussian path")
    parent_path, parent = find_parent(args.trace_dir, args.init_run_dir)
    directory = Path(args.trace_dir)/("tuned_state_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/"experiment_cards/A_imagenet_tuned_state.md").read_text(encoding="utf-8")
    (directory/"experiment_card.md").write_text(card, encoding="utf-8")
    record = dict(experiment="imagenet_first_init_fixed30_state", status="running", arguments=vars(args),
        source_init_run=str(parent_path), config=parent["config"], first_zeta_init=parent["first_zeta_init"],
        planned_method_nfe=600, planned_diagnostic_nfe=90, method_nfe=0, diagnostic_nfe=0,
        results=[], arms={}, gates={}, experiment_card=card,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        gt_usage="post-training evaluation only; no online selection/early stop",
        parameter_mapping="same 30 reverse-call positions; no noise-anchor parameter interpolation",
        random_pairing="same x_T and normals by epoch/call position; not equal physical-time Brownian paths")
    save = lambda: save_record(directory, record)
    save()
    print(f"Parent: {parent_path}\nRecords: {directory}", flush=True)
    print("90 diagnostic NFE then null/state 300+300; first epoch is an included pilot; no extra final sampling.", flush=True)
    try:
        import torch
        import torch.nn.functional as F
        from configs.config import IMG_SIZE, METRICS_CONFIG
        from modules.main_single import load_diffusion_model, load_image_as_tensor
        from modules.degradations import get_operator
        from modules.zaps_algorithm import ZAPS
        from utils.diag_dps_same_observation import file_sha256
        from utils.diag_learning_rate_ablation import set_seed
        from utils.diag_zaps_first_init import rng_preflight
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection, restore_sampling_rng
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.metrics import compute_all_metrics
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        trace = json.loads((Path(args.trace_dir)/"audit.json").read_text(encoding="utf-8"))
        source_dir = Path(trace["source"])
        source = json.loads((source_dir/"run.json").read_text(encoding="utf-8"))
        if (source["arguments"]["dataset"] != "imagenet" or int(source["arguments"]["seed"]) != 1001
            or trace["arguments"]["task"] != "gaussian_deblur"):
            raise RuntimeError("Keep archived ImageNet image/seed")
        paths = {"core_sha256": PROJECTS/"modules/zaps_algorithm.py",
                 "operator_source_sha256": PROJECTS/"modules/degradations.py",
                 "measurement_sha256": source_dir/"gaussian_deblur/measurement.pt",
                 "image_sha256": Path(source["arguments"]["image"])}
        for key,path in paths.items():
            if file_sha256(path) != parent["identity"][key]:
                raise RuntimeError(f"Identity changed: {key}")
        for name,digest in parent["current_supporting_source_sha256"].items():
            if file_sha256(PROJECTS/name) != digest:
                raise RuntimeError(f"Supporting source changed: {name}")
        for key,name in (("output_sha256", "first_init_last_opt.pt"), ("last_unroll_state_sha256", "last_unroll_state.pt")):
            if file_sha256(parent_path.parent/name) != parent[key]:
                raise RuntimeError("Parent tensor changed")
        for key,name in (("parent_first_sha256", "source_first_run"), ("parent_direction_sha256", "source_direction_run")):
            if file_sha256(Path(parent[name])) != parent[key]:
                raise RuntimeError("Parent JSON chain changed")
        model = load_diffusion_model("imagenet", args.device)
        model.model.eval()
        if (file_sha256(model.ckpt_path) != parent["identity"]["checkpoint_sha256"] or model.num_steps != 1000
            or any(p.dtype != torch.float32 for p in model.model.parameters())
            or str(torch.__version__) != parent["runtime"]["torch"]):
            raise RuntimeError("Keep the parent checkpoint/fp32/PyTorch runtime")
        operator = get_operator("gaussian_deblur", device=args.device, **source["task_configs"]["gaussian_deblur"])
        y = torch.load(paths["measurement_sha256"], map_location=args.device, weights_only=True)
        archived = torch.load(parent_path.parent/"last_unroll_state.pt", map_location="cpu", weights_only=True)
        mapping = saved_cuda_selection(archived, args.device, args.saved_cuda_index)
        seed = int(source["arguments"]["seed"])
        preflight, expected_end = rng_preflight(seed, archived, args.device, mapping["saved_sampling_index"])
        record["gates"]["noise_preflight"] = preflight
        record.update(identity=parent["identity"], parent_sha256=file_sha256(parent_path),
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, python=sys.version),
            reference_alignment="log_sigma_u; log-residual interpolation; u-scaled EMA; global clean cosine",
            supporting_source_sha256={**parent["current_supporting_source_sha256"],
                "modules/adaptive_scheduler.py": file_sha256(PROJECTS/"modules/adaptive_scheduler.py")})
        save()
        if not preflight["passed"]:
            raise RuntimeError("Noise preflight failed before UNet calls")
        def construct():
            set_seed(seed)
            z = ZAPS(model, operator, img_size=IMG_SIZE[0], **parent["config"])
            with torch.no_grad():
                z.zeta[-1] = parent["first_zeta_init"]
            if not torch.equal(z.tau.cpu(), archived["tau"]):
                raise RuntimeError("Nominal grid changed")
            return z
        def scheduler(z, response):
            return NoiseProfileScheduler(z.tau.flip(0).tolist(), model.alphas_cumprod.cpu().tolist(), response)
        z = construct()
        set_seed(seed)
        initial = torch.randn(*archived["init_noise"].shape, device=args.device)
        start_rng = rng_state(args.device)
        probe_state = dict(cpu_rng=start_rng["cpu"], cuda_rng=start_rng["cuda"])
        current_index = torch.device(args.device).index
        current_index = torch.cuda.current_device() if current_index is None else current_index
        def probe(adaptive):
            restore_sampling_rng(probe_state, args.device, current_index)
            s = scheduler(z, 0.) if adaptive else None
            visited, indices = [], []
            def before(module, inputs):
                record["diagnostic_nfe"] += 1
                t = int(inputs[1].flatten()[0]); visited.append(t)
                if s is not None:
                    s.set_time(t)
            original = z._tweedie_estimate
            def tweedie(x, eps, ab, index):
                indices.append(index)
                return original(x, eps, ab, index)
            hook = model.model.register_forward_pre_hook(before)
            try:
                with model_rng_guard(model.model, args.device), patch.object(z, "_tweedie_estimate", side_effect=tweedie):
                    counter = [0]
                    if adaptive:
                        out = z._reverse_diffusion_adaptive(y, s, nfe_counter=counter, init_noise=initial)
                    else:
                        out = z._reverse_diffusion(y, nfe_counter=counter, init_noise=initial)
                    loss = F.mse_loss(operator.H(out), y)
                    gradients = torch.autograd.grad(loss, (z.zeta, z.D))
                return dict(output=out.detach().cpu(), zeta_grad=gradients[0].detach().cpu(),
                    D_grad=gradients[1].detach().cpu(), visited=visited, indices=indices,
                    nfe=counter[0], rng=rng_state(args.device))
            finally:
                hook.remove()
        print("Checking fixed-repeat/adaptive-null forward and gradients (90 NFE)...", flush=True)
        a, b, c = probe(False), probe(False), probe(True)
        def errors(left, right):
            return {k: float((left[k].double()-right[k].double()).norm()/left[k].double().norm().clamp_min(1e-12))
                    for k in ("output", "zeta_grad", "D_grad")}
        metadata = all(v["visited"] == z.tau.flip(0).tolist() and v["indices"] == list(range(29,-1,-1)) and v["nfe"] == 30
                       for v in (a,b,c)) and same_rng(a["rng"], b["rng"]) and same_rng(a["rng"], c["rng"])
        record["gates"]["one_unroll_parity"] = parity_decision(errors(a,b), errors(a,c), metadata)
        save()
        if not record["gates"]["one_unroll_parity"]["passed"] or record["diagnostic_nfe"] != 90:
            raise RuntimeError("One-unroll parity failed; do not interpret state quality")
        del a,b,c,z
        for name,response in (("null", 0.), ("state", .2)):
            z, rows = construct(), []
            s = scheduler(z, response)
            record["scheduler_config"] = asdict(s.cfg)
            def before(module, inputs):
                record["method_nfe"] += 1
                t = int(inputs[1].flatten()[0]); s.set_time(t)
                if len(rows) == 270:
                    # PRE epoch-10 forward/Adam state, not post-update parameters.
                    rng = rng_state(args.device)
                    torch.save(dict(zeta=z.zeta.detach().cpu().clone(), D=z.D.detach().cpu().clone(),
                        tau=z.tau.detach().cpu(), init_noise=z._last_opt_noise.detach().cpu(),
                        cpu_rng=rng["cpu"], cuda_rng=rng["cuda"],
                        sampling_device=str(args.device), sampling_cuda_index=current_index,
                        pilot_log_residual=list(s.pilot_log_r), pilot_cosine=list(s.pilot_cos),
                        completed_pilot_epochs=s._completed_unrolls), directory/(name+"_last_unroll_state.pt"))
                rows.append(t)
            hook = model.model.register_forward_pre_hook(before)
            print(f"\n--- {name}: response={response}; joint zeta+D; 30x10 ---", flush=True)
            set_seed(seed)
            started = time.perf_counter()
            try:
                with model_rng_guard(model.model, args.device) as guarded:
                    losses = z.optimize(y, verbose=True, x0_gt=None, scheduler=s)
            finally:
                hook.remove()
            grids = [rows[k:k+30] for k in range(0,len(rows),30)]
            pairing = same_rng(expected_end, rng_state(args.device)) and torch.equal(z._last_opt_noise.cpu(), archived["init_noise"])
            valid = len(grids) == 10 and all(len(g) == 30 and g[0] == 999 and g[-1] == 0
                    and all(a>b for a,b in zip(g,g[1:])) for g in grids)
            pilot_equal = grids[0] == archived["tau"].flip(0).tolist()
            if (not pairing or not valid or not pilot_equal or len(guarded) != 300 or z._last_nfe != 300
                or len(losses) != 10 or (name == "null" and any(g != grids[0] for g in grids))):
                raise RuntimeError("Training grid/pairing/budget failed")
            if any(p.grad is not None for p in model.model.parameters()):
                raise RuntimeError("Unexpected model weight gradients")
            output = z._last_opt_x0.detach().cpu()
            torch.save(output, directory/(name+"_last_opt.pt"))
            torch.save(dict(zeta=z.zeta.detach().cpu(), D=z.D.detach().cpu()), directory/(name+"_post_update_params.pt"))
            record["arms"][name] = dict(method_nfe=300, seconds=time.perf_counter()-started, loss_history=losses,
                grids=grids, first_zeta_after_update=float(z.zeta[-1].detach()),
                pilot_log_residual=s.pilot_log_r, pilot_cosine=s.pilot_cos,
                last_indicators=z._indicator_log, noise_pairing_passed=pairing,
                output_sha256=file_sha256(directory/(name+"_last_opt.pt")),
                last_unroll_state_sha256=file_sha256(directory/(name+"_last_unroll_state.pt")),
                parameter_semantics="post_update_params do NOT generate last_opt; no fresh sample")
            save()
            del z,output
        # All training is complete before GT/LPIPS enters the process.
        gt = load_image_as_tensor(source["arguments"]["image"]).cpu()
        cached = torch.load(parent_path.parent/"first_init_last_opt.pt", map_location="cpu", weights_only=True)
        old = next(r for r in parent["results"] if r["policy"] == "first_init_joint_training")
        cached_metrics = compute_all_metrics(cached, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        if not all(math.isclose(cached_metrics[k], old[k], rel_tol=2e-5, abs_tol=1e-6) for k in cached_metrics):
            raise RuntimeError("Cached metric compatibility failed")
        for name in ("null", "state"):
            output = torch.load(directory/(name+"_last_opt.pt"), map_location="cpu", weights_only=True)
            metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
            loss = record["arms"][name]["loss_history"][-1]
            record["results"].append(dict(policy=name, **metrics, method_nfe=300,
                final_mse=loss, final_residual=math.sqrt(loss*y.numel())))
        record["historical_uniform_reference"] = dict(metrics=trace["results"]["uniform_30"]["metrics"],
            limits="historical same-image comparator, different initial parameter setting; not current paired state-null")
        record["state_minus_historical_uniform"] = {k: record["results"][1][k]-trace["results"]["uniform_30"]["metrics"][k]
            for k in ("psnr", "ssim", "lpips")}
        record["cached_parent_metrics"] = cached_metrics
        record["decision"] = interpret(record["results"][0], record["results"][1], cached_metrics)
        if record["method_nfe"] != 600:
            raise RuntimeError("Unexpected total method cost")
        record["status"] = "complete"
        save()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save()
        raise
    print("\n=== Paired ImageNet tuned-state summary ===", flush=True)
    print(f"{'policy':>12} {'PSNR':>9} {'dNull':>9} {'SSIM':>8} {'LPIPS':>8} {'NFE':>6}", flush=True)
    null = record["results"][0]
    for r in record["results"]:
        print(f"{r['policy']:>12} {r['psnr']:9.4f} {r['psnr']-null['psnr']:+9.4f} {r['ssim']:8.4f} {r['lpips']:8.4f} {r['method_nfe']:6d}", flush=True)
    print("Decision:", json.dumps(record["decision"], ensure_ascii=False), flush=True)
    print("State minus historical uniform (context, NOT paired state gain):",
          json.dumps(record["state_minus_historical_uniform"]), flush=True)
    print(f"Method NFE={record['method_nfe']}; diagnostic NFE={record['diagnostic_nfe']}; records: {directory}", flush=True)


if __name__ == "__main__":
    main()
