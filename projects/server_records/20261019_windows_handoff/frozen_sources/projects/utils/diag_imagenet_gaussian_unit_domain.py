"""One original-setting Gaussian unit-measurement candidate, NOT tuning/state gain.

Keep UNet/model state in [-1,1]; A(x)=H((x+1)/2), J_A^T=.5 H^T.
One new 300-NFE arm; original completed model-domain output is cache only.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
ROOT = PROJECTS.parent
sys.path.insert(0, str(PROJECTS))
ARM = "irregular_15_10_5"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_reference(reference):
    from utils.diag_imagenet_original_reference import original_config
    original_config(reference.get("config"))
    if (reference.get("experiment") != "original_imagenet_reference_cached_audit"
            or reference.get("status") != "complete" or reference.get("method_nfe") != 0
            or reference.get("diagnostic_nfe") != 0 or reference.get("model_loaded") is not False):
        raise ValueError("Need completed ORIGINAL zero-UNet reference, not tuned/state data")
    for key in ("source_identity", "cached_mechanics", "cached_output_metrics_and_loss"):
        if reference.get("gates", {}).get(key, {}).get("passed") is not True:
            raise ValueError(f"Reference gate missing/failed: {key}")
    b, domain = reference["baseline"], reference["measurement_domain"]
    if (b["task"] != "gaussian_deblur" or b["seed"] != 1001
            or b["historical_method_nfe"] != 300 or b["new_method_nfe"] != 0
            or domain["model_domain"] != [-1, 1] or domain["configured_sigma"] != .05
            or reference["decision"].get("author_parity") is not False):
        raise ValueError("Wrong reference task/domain/seed/cost or overclaimed parity")


def interpret(candidate, original):
    delta = {k: candidate[k]-original[k] for k in ("psnr", "ssim", "lpips")}
    if delta["psnr"] >= .2 and delta["ssim"] >= .005 and delta["lpips"] <= -.005:
        label = "SUPPORTS_GAUSSIAN_MEASUREMENT_PROTOCOL_SENSITIVITY"
    elif delta["psnr"] <= .05 and delta["ssim"] <= .001 and delta["lpips"] >= -.001:
        label = "NO_COMPREHENSIVE_QUALITY_SUPPORT_FOR_UNIT_PROTOCOL"
    else:
        label = "MIXED_OR_INCONCLUSIVE"
    return dict(classification=label, deltas_vs_original=delta,
        author_parity=False, benchmark_reproduced=False, state_gain=False,
        original_reference_overwritten=False,
        next_action="Discuss protocol evidence and remaining unverified conventions; no automatic tuning or state run",
        limits="One image; changed physical SNR and guidance/loss units together; quality cannot certify the author's internal conventions")


def unit_operator(base):
    """Pure affine measurement; transpose means Jacobian adjoint, not inverse.

    Gaussian.transpose accepts ONLY r. Do not reuse SR's output_size keyword.
    """
    import torch.nn as nn
    class UnitMeasurement(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base

        def H(self, x):
            return self.base.H((x+1)*.5)

        def forward(self, x):
            return self.H(x)

        def transpose(self, residual):
            return .5*self.base.transpose(residual)
    return UnitMeasurement()


def measurement_and_gates(base, operator, gt, old_y, direction):
    """Only convolution/VJP, zero UNet and zero random draws. GT builds y only."""
    import torch
    def error(a, b):
        return float((a.detach().double()-b.detach().double()).norm()/b.detach().double().norm().clamp(min=1e-12))
    with torch.no_grad():
        old_noise = old_y-base.H(gt)
        standard_noise = old_noise/.05
        y = operator.H(gt)+old_noise
        expected_affine = .5*base.H(gt)+.5*base.H(torch.ones_like(gt))
        affine_error = error(operator.H(gt), expected_affine)
        # Use a well-scaled direction/residual; no stochastic probe or model.
        residual = base.H(direction)
        left = ((operator.H(gt+direction)-operator.H(gt)).double()*residual.double()).sum()
        right = (direction.double()*operator.transpose(residual).double()).sum()
        adjoint_error = float((left-right).abs()/torch.maximum(left.abs(), right.abs()).clamp(min=1e-12))
        noise_error = error(y-operator.H(gt), old_noise)
        boundary_offset = float((base.H(torch.ones_like(gt))-1).abs().max())
    with torch.enable_grad():
        x = gt.detach().clone().requires_grad_(True)
        actual_vjp = torch.autograd.grad((operator.H(x)*residual).sum(), x)[0]
    vjp_error = error(operator.transpose(residual), actual_vjp)
    gate = dict(relative_tolerance=2e-5, affine_relative_error=affine_error,
        increment_adjoint_relative_error=adjoint_error, autograd_vjp_relative_error=vjp_error,
        noise_reconstruction_relative_error=noise_error,
        zero_padding_H1_max_difference_from_one=boundary_offset, score_nfe=0,
        empirical_numeric_sigma=float(old_noise.double().std(unbiased=False)),
        numeric_noise_mean=float(old_noise.double().mean()))
    gate["passed"] = all(math.isfinite(v) and v <= 2e-5 for v in (
        affine_error, adjoint_error, vjp_error, noise_error))
    return y.detach(), old_noise.detach(), standard_noise.detach(), gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, help="Completed original_reference_audit directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("Production run requires the archived CUDA noise path")
    reference_path = Path(args.reference_dir).resolve()/"audit.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    validate_reference(reference)
    first_path = Path(reference["source_first_run"])
    first = json.loads(first_path.read_text(encoding="utf-8"))
    guidance_path = Path(reference["source_guidance"])
    guidance = json.loads(guidance_path.read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"])
    trace = json.loads((trace_dir/"audit.json").read_text(encoding="utf-8"))
    source_dir = Path(trace["source"])
    source = json.loads((source_dir/"run.json").read_text(encoding="utf-8"))
    from utils.diag_zaps_first_direction import validate_first_run, model_rng_guard
    from utils.diag_zaps_parameter_gradient import validate_protocol, PROBE_TIMES, DELTAS
    validate_first_run(first, guidance, trace_dir)
    validate_protocol(guidance, trace, source, PROBE_TIMES, DELTAS)
    config = reference["config"]
    if (trace["results"][ARM]["config"] != config or first["config"] != config
            or source["arguments"]["seed"] != 1001
            or source["arguments"]["image"] != reference["baseline"]["image"]
            or source["task_configs"]["gaussian_deblur"] != dict(kernel_size=61,sigma=3.,noise_sigma=.05)):
        raise ValueError("Original config/image/Gaussian protocol chain differs")
    output_dir = trace_dir/("gaussian_unit_domain_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/"experiment_cards/R_gaussian_unit_domain.md").read_text(encoding="utf-8")
    (output_dir/"experiment_card.md").write_text(card, encoding="utf-8")
    record = dict(experiment="imagenet_gaussian_unit_measurement_original_settings", status="running",
        arguments=vars(args), source_reference=str(reference_path), source_reference_sha256=digest(reference_path),
        config=config, task_config=source["task_configs"]["gaussian_deblur"],
        planned_method_nfe=300, new_nfe=0, diagnostic_nfe=0, optimizer_steps=0,
        gates={}, results=[], gradients=[], experiment_card=card,
        resolved_protocol=dict(model_state_domain=[-1,1], measurement_domain=[0,1], configured_sigma=.05,
            operator="H((x+1)/2)", jacobian_adjoint=".5 H^T(r)", normalized_noise="from archived y-H(gt), no redraw",
            observation_clipped=False, state=False, final_mode="last_opt_pre_final_adam_update",
            physical_noise_changed=True, new_final_sample=False),
        gt_usage="Synthetic measurement construction and final evaluation only; optimize x0_gt=None; no selection",
        inherited_open_items=dict(production_fp32_sampler_overall_pass_not_claimed=True,
            parameter_backward="Past scale disagreements remain unverified",author_parity=False),
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    started = time.perf_counter()
    def save():
        record["elapsed_seconds"] = time.perf_counter()-started
        (output_dir/"run.json").write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    save()
    print("=== Original-setting Gaussian unit-measurement candidate ===", flush=True)
    print("Keep ALL zeta/D=.2, lr=.001, paper30x10; one new arm=300 NFE; original output CACHE ONLY.", flush=True)
    print("New physical sigma=.05 in [0,1], not the same y/SNR; no state/noise attenuation/init tuning.", flush=True)
    print(f"Records: {output_dir}", flush=True)
    try:
        import torch
        from configs.config import IMG_SIZE, METRICS_CONFIG
        from modules.main_single import load_diffusion_model, load_image_as_tensor
        from modules.degradations import get_operator
        from utils.diag_learning_rate_ablation import set_seed
        from utils.diag_zaps_first_init import rng_preflight
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection
        from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.metrics import compute_all_metrics
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        dps = json.loads((Path(guidance["identity_baseline"])/"run.json").read_text(encoding="utf-8"))
        paths = dict(checkpoint_sha256=Path(dps["checkpoint_path"]),
            measurement_sha256=source_dir/"gaussian_deblur/measurement.pt",
            image_sha256=Path(source["arguments"]["image"]), core_sha256=PROJECTS/"modules/zaps_algorithm.py",
            operator_source_sha256=PROJECTS/"modules/degradations.py")
        print("Checking parent/current identities (local file reads, no download)...", flush=True)
        for key,path in paths.items():
            if digest(path) != reference["identity"][key] or guidance[key] != reference["identity"][key]:
                raise RuntimeError(f"Identity changed: {key}")
        for name,sha in reference["current_source_sha256"].items():
            if digest(PROJECTS/name) != sha:
                raise RuntimeError(f"Reference supporting source changed: {name}")
        if digest(first_path) != reference["source_first_sha256"] or digest(guidance_path) != first["source_vjp_sha256"]:
            raise RuntimeError("Parent JSON chain changed")
        raw_path = first_path.parent/"raw_baseline.pt"
        state_path = trace_dir/(ARM+"_last_unroll_state.pt")
        if digest(raw_path) != reference["baseline"]["output_current_sha256"] or digest(state_path) != first["snapshot_sha256"]:
            raise RuntimeError("Original cached output/snapshot changed")
        record.update(identity=reference["identity"],
            supporting_source_sha256={name:digest(PROJECTS/name) for name in (
                "modules/diffusion_model.py","modules/wavelet.py","utils/metrics.py","configs/config.py",
                "utils/diag_imagenet_gaussian_unit_domain.py","utils/diag_zaps_optimized_trace.py")},
            archived_snapshot_sha256=digest(state_path), original_output_sha256=digest(raw_path),
            runtime=dict(torch=str(torch.__version__),python=sys.version,cuda=torch.version.cuda,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),visible_cuda_count=torch.cuda.device_count(),
                sampling_device=args.device,gpu_name=torch.cuda.get_device_name(torch.device(args.device)),
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        record["gates"]["identity"] = dict(passed=True)
        state = torch.load(state_path,map_location="cpu",weights_only=True)
        mapping = saved_cuda_selection(state,trace["arguments"].get("device"),args.saved_cuda_index)
        base = get_operator("gaussian_deblur",device=args.device,**source["task_configs"]["gaussian_deblur"])
        operator = unit_operator(base).to(args.device)
        gt = load_image_as_tensor(source["arguments"]["image"]).to(args.device)
        old_y = torch.load(paths["measurement_sha256"],map_location=args.device,weights_only=True)
        raw = torch.load(raw_path,map_location="cpu",weights_only=True)
        for name,tensor in (("gt",gt),("old_y",old_y),("raw",raw),("x_T",state["init_noise"])):
            if tensor.shape != (1,3,256,256) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise RuntimeError(f"Invalid tensor: {name}")
        before = rng_state(args.device)
        y,noise,standard_noise,gates = measurement_and_gates(base,operator,gt,old_y,state["init_noise"].to(args.device))
        gates["rng_unchanged"] = same_rng(before,rng_state(args.device))
        gates["passed"] &= gates["rng_unchanged"]
        record["gates"]["affine_measurement"] = gates
        save()
        print("0-UNet affine/VJP/adjoint/noise gate:",json.dumps(gates),flush=True)
        if not gates["passed"]:
            raise RuntimeError("Affine measurement gate failed; no training")
        with torch.no_grad():
            old_residual = old_y-base.H(raw.to(args.device))
        if not math.isclose(float(old_residual.double().square().mean()),reference["baseline"]["final_mse"],rel_tol=2e-5,abs_tol=1e-8):
            raise RuntimeError("Cached original loss differs")
        baseline_metrics = compute_all_metrics(raw,gt.cpu(),lpips_net=METRICS_CONFIG["lpips_net"])
        if not all(math.isclose(baseline_metrics[k],reference["baseline"][k],rel_tol=2e-5,abs_tol=1e-6) for k in baseline_metrics):
            raise RuntimeError("Cached original metrics differ")
        record["gates"]["cached_metrics_loss"] = dict(passed=True)
        record["results"].append(dict(policy="cached_original_model_domain",**baseline_metrics,new_nfe=0,
            loss_domain="model",final_mse=reference["baseline"]["final_mse"],final_residual=reference["baseline"]["final_residual"]))
        for name,tensor in (("measurement_unit.pt",y),("archived_numeric_noise.pt",noise),("standard_noise.pt",standard_noise)):
            torch.save(tensor.cpu(),output_dir/name)
        record["new_measurement_sha256"] = digest(output_dir/"measurement_unit.pt")
        print("Loading original ImageNet checkpoint; next: zero-UNet noise pairing, then 10 epochs...",flush=True)
        model = load_diffusion_model("imagenet",args.device)
        model.model.eval()
        if (Path(model.ckpt_path).resolve() != paths["checkpoint_sha256"].resolve() or model.num_steps != 1000
                or any(p.dtype != torch.float32 for p in model.model.parameters())):
            raise RuntimeError("Model/checkpoint/fp32 mismatch")
        zaps = LastUnrollSnapshotZAPS(model,operator,img_size=IMG_SIZE[0],**config)
        if (not torch.equal(zaps.tau.cpu(),state["tau"])
                or not torch.equal(zaps.zeta,torch.full_like(zaps.zeta,.2))
                or not torch.equal(zaps.D,torch.full_like(zaps.D,.2))):
            raise RuntimeError("Original initial values or grid changed")
        seed = 1001
        preflight,expected_end = rng_preflight(seed,state,args.device,mapping["saved_sampling_index"])
        record["gates"]["noise_preflight"],record["rng_mapping"] = preflight,mapping
        save()
        if not preflight["passed"]:
            raise RuntimeError("Original noise path cannot be paired; stop before training")
        visited=[]
        def count(module,inputs):
            record["new_nfe"] += 1
            if record["new_nfe"] > 300:
                raise RuntimeError("300-NFE budget exceeded")
            visited.append(int(inputs[1].flatten()[0]))
        original_clip = torch.nn.utils.clip_grad_norm_
        def clip(parameters,max_norm,**kwargs):
            params=list(parameters)
            if len(params)!=2 or params[0] is not zaps.zeta or params[1] is not zaps.D or max_norm!=1:
                raise RuntimeError("Original optimizer/clip contract changed")
            total=original_clip(params,max_norm,**kwargs)
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite gradient")
            record["gradients"].append(dict(epoch=len(record["gradients"])+1,joint_norm=float(total)))
            return total
        hook=model.model.register_forward_pre_hook(count)
        set_seed(seed)
        train_started=time.perf_counter()
        try:
            with model_rng_guard(model.model,args.device) as guarded:
                with patch.object(torch.nn.utils,"clip_grad_norm_",side_effect=clip):
                    losses=zaps.optimize(y,verbose=True,x0_gt=None)
        finally:
            hook.remove()
        record["training_seconds"] = time.perf_counter()-train_started
        record["optimizer_steps"],record["loss_history"] = len(losses),losses
        last=zaps.audit_state
        current_index=torch.device(args.device).index
        current_index=torch.cuda.current_device() if current_index is None else current_index
        pairing=dict(x_T_equal=last is not None and torch.equal(last["init_noise"],state["init_noise"]),
            epoch10_cpu_rng_equal=last is not None and torch.equal(last["cpu_rng"],state["cpu_rng"]),
            epoch10_sampling_rng_equal=last is not None and torch.equal(last["cuda_rng"][current_index],state["cuda_rng"][mapping["saved_sampling_index"]]),
            end_rng_equal=same_rng(expected_end,rng_state(args.device)),times_equal=visited==state["tau"].flip(0).tolist()*10,
            budget_equal=record["new_nfe"]==zaps._last_nfe==len(guarded)==300,
            epochs_equal=zaps.audit_calls==len(losses)==len(record["gradients"])==10)
        pairing["passed"]=all(pairing.values())
        record["gates"]["training_pairing"]=pairing
        save()
        if not pairing["passed"] or not all(math.isfinite(v) for v in losses):
            raise RuntimeError("Training pairing/budget/finite loss gate failed")
        if any(p.grad is not None for p in model.model.parameters()):
            raise RuntimeError("Model weights received gradients")
        output=zaps._last_opt_x0.detach().cpu()
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite last_opt")
        torch.save(output,output_dir/"unit_last_opt.pt")
        torch.save(last,output_dir/"last_unroll_state.pt")
        record["output_sha256"],record["snapshot_sha256"] = digest(output_dir/"unit_last_opt.pt"),digest(output_dir/"last_unroll_state.pt")
        record["last_opt_used_zeta_ascending"]=last["zeta"].tolist()
        record["post_update_zeta_ascending"]=zaps.zeta.detach().cpu().tolist()
        record["post_update_note"]="Not the parameters that produced last_opt; use saved PRE snapshot for replay"
        with torch.no_grad():
            residual=y-operator.H(output.to(args.device))
        metrics=compute_all_metrics(output,gt.cpu(),lpips_net=METRICS_CONFIG["lpips_net"])
        record["results"].append(dict(policy="unit_protocol_original_initialization",**metrics,new_nfe=record["new_nfe"],
            loss_domain="unit",final_mse=float(residual.double().square().mean()),final_residual=float(residual.double().norm())))
        record["decision"]=interpret(metrics,baseline_metrics)
        record["status"]="complete"
        with (output_dir/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(record["results"][0]))
            writer.writeheader()
            writer.writerows(record["results"])
        save()
    except Exception as error:
        record["status"],record["error"]="failed",repr(error)
        save()
        raise
    print("\n=== Original-setting Gaussian measurement-domain summary ===",flush=True)
    for r in record["results"]:
        print(f"{r['policy']:>36} PSNR={r['psnr']:.4f} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f} newNFE={r['new_nfe']}",flush=True)
    print("MSE/residual are native-domain units; do not compare their magnitudes directly across rows.",flush=True)
    print("Decision:",json.dumps(record["decision"],ensure_ascii=False),flush=True)
    print(f"New UNet NFE={record['new_nfe']}; records: {output_dir}",flush=True)


if __name__=="__main__":
    main()
