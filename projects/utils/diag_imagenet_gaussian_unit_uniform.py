"""Add ONLY uniform30 under the completed Gaussian unit-measurement protocol.

Reuse cached paper30 output; no retraining/replay of that arm. One new 300-NFE
trajectory optimization, original joint initialization/LR, no state/tuning.
"""
import argparse
import csv
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

PROJECTS = Path(__file__).resolve().parents[1]
ROOT = PROJECTS.parent
sys.path.insert(0, str(PROJECTS))
from utils.diag_imagenet_gaussian_unit_domain import digest, unit_operator, validate_reference

RUNTIME_KEYS = ("torch", "cuda", "gpu_name", "cudnn_benchmark", "cudnn_deterministic",
    "cudnn_allow_tf32", "matmul_allow_tf32", "deterministic_algorithms")


def uniform_grid():
    # denominator=29 has no half-integer ties; equivalent to round(999*i/29).
    return [(999*i+14)//29 for i in range(30)]


def validate_parent(parent):
    from utils.diag_imagenet_original_reference import original_config
    original_config(parent.get("config"))
    if (parent.get("experiment") != "imagenet_gaussian_unit_measurement_original_settings"
            or parent.get("status") != "complete" or parent.get("new_nfe") != 300
            or parent.get("diagnostic_nfe") != 0 or parent.get("optimizer_steps") != 10
            or parent.get("task_config") != dict(kernel_size=61,sigma=3.,noise_sigma=.05)):
        raise ValueError("Need the completed original-setting Gaussian unit-domain run")
    expected = dict(model_state_domain=[-1,1],measurement_domain=[0,1],configured_sigma=.05,
        operator="H((x+1)/2)",jacobian_adjoint=".5 H^T(r)",
        normalized_noise="from archived y-H(gt), no redraw",observation_clipped=False,state=False,
        final_mode="last_opt_pre_final_adam_update",physical_noise_changed=True,new_final_sample=False)
    if parent.get("resolved_protocol") != expected:
        raise ValueError("Wrong unit measurement, initialization/state or output protocol")
    for key in ("identity","affine_measurement","cached_metrics_loss","noise_preflight","training_pairing"):
        if parent.get("gates",{}).get(key,{}).get("passed") is not True:
            raise ValueError(f"Parent gate failed/missing: {key}")
    if (len(parent.get("loss_history",[])) != 10
            or not all(math.isfinite(v) for v in parent["loss_history"])
            or len(parent.get("gradients",[])) != 10):
        raise ValueError("Incomplete/non-finite parent training")
    rows=parent.get("results",[])
    if ([r.get("policy") for r in rows] != ["cached_original_model_domain","unit_protocol_original_initialization"]
            or rows[1].get("new_nfe") != 300 or rows[1].get("loss_domain") != "unit"):
        raise ValueError("Need the completed unit paper result, not old model-domain uniform")
    if not all(math.isfinite(rows[1][k]) for k in ("psnr","ssim","lpips","final_mse","final_residual")):
        raise ValueError("Non-finite parent metrics")
    return rows[1]


def check_runtime(expected, actual):
    differences={k:dict(parent=expected.get(k),current=actual.get(k)) for k in RUNTIME_KEYS
        if k not in expected or k not in actual or expected[k] != actual[k]}
    return dict(passed=not differences,checked_keys=list(RUNTIME_KEYS),differences=differences,
        limits="Same runtime flags/GPU model, not a guarantee of deterministic CUDA backward")


def interpret(paper, uniform):
    delta={k:paper[k]-uniform[k] for k in ("psnr","ssim","lpips")}
    if delta["psnr"] >= .1 and delta["ssim"] >= -.002 and delta["lpips"] <= .005:
        label="IRREGULAR_PSNR_ADVANTAGE_WITHOUT_MATERIAL_AUXILIARY_REGRESSION"
    elif delta["psnr"] <= -.1:
        label="UNIFORM_PSNR_ADVANTAGE_PERSISTS"
    elif abs(delta["psnr"]) < .1:
        label="PSNR_CLOSE_ON_THIS_IMAGE"
    else:
        label="MIXED_PSNR_AND_PERCEPTUAL_TRADEOFF"
    return dict(classification=label,irregular_minus_uniform=delta,
        author_parity=False,benchmark_reproduced=False,state_gain=False,
        next_action="Discuss this matched-protocol result before another image or state experiment",
        limits="Single image/seed, cached paper arm; engineering thresholds, not significance or a universal schedule ranking")


def operator_gate(operator, x, direction):
    """Check actual affine increment/adjoint/VJP; no GT or random draws."""
    import torch
    with torch.no_grad():
        residual=operator.base.H(direction)
        actual=operator.H(x)
        expected=.5*operator.base.H(x)+.5*operator.base.H(torch.ones_like(x))
        affine=float((actual.double()-expected.double()).norm()/expected.double().norm().clamp(min=1e-12))
        left=((operator.H(x+direction)-actual).double()*residual.double()).sum()
        right=(direction.double()*operator.transpose(residual).double()).sum()
        adjoint=float((left-right).abs()/torch.maximum(left.abs(),right.abs()).clamp(min=1e-12))
    with torch.enable_grad():
        probe=x.detach().clone().requires_grad_(True)
        exact=torch.autograd.grad((operator.H(probe)*residual).sum(),probe)[0]
    approximate=operator.transpose(residual)
    vjp=float((approximate.double()-exact.double()).norm()/exact.double().norm().clamp(min=1e-12))
    return dict(passed=all(math.isfinite(v) and v<=2e-5 for v in (affine,adjoint,vjp)),
        affine_relative_error=affine,increment_adjoint_relative_error=adjoint,
        autograd_vjp_relative_error=vjp,relative_tolerance=2e-5,score_nfe=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-run-dir",required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--saved-cuda-index",type=int,default=None)
    args=parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("Production run needs the archived CUDA random draws")
    parent_dir=Path(args.unit_run_dir).resolve()
    parent_path=parent_dir/"run.json"
    parent=json.loads(parent_path.read_text(encoding="utf-8"))
    paper=validate_parent(parent)
    reference_path=Path(parent["source_reference"])
    reference=json.loads(reference_path.read_text(encoding="utf-8"))
    validate_reference(reference)
    if digest(reference_path)!=parent["source_reference_sha256"] or parent["config"]!=reference["config"]:
        raise ValueError("Original reference chain/config changed")
    first_path=Path(reference["source_first_run"])
    first=json.loads(first_path.read_text(encoding="utf-8"))
    guidance=json.loads(Path(reference["source_guidance"]).read_text(encoding="utf-8"))
    trace_dir=Path(guidance["source_trace"])
    trace=json.loads((trace_dir/"audit.json").read_text(encoding="utf-8"))
    source_dir=Path(trace["source"])
    dps=json.loads((Path(guidance["identity_baseline"])/"run.json").read_text(encoding="utf-8"))
    output_dir=parent_dir/("uniform_comparison_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True,exist_ok=False)
    card=(PROJECTS/"experiment_cards/R_gaussian_unit_uniform.md").read_text(encoding="utf-8")
    (output_dir/"experiment_card.md").write_text(card,encoding="utf-8")
    resolved={k:v for k,v in parent["config"].items() if k not in ("schedule","timestep_spacing","schedule_power")}
    resolved.update(schedule_label="uniform_30",grid_builder="nearest integer 999*i/29",
        actual_timesteps_ascending=uniform_grid(),parameter_indexing="ascending grid ordinal; all fresh .2; joint training",
        dataset="imagenet",task="gaussian_deblur",seed=1001,image=reference["baseline"]["image"])
    record=dict(experiment="gaussian_unit_uniform_vs_cached_paper",status="running",arguments=vars(args),
        parent_run=str(parent_path),parent_run_sha256=digest(parent_path),constructor_only_config=parent["config"],
        resolved_config=resolved,resolved_protocol=parent["resolved_protocol"],source_paper_method_nfe=300,
        planned_method_nfe=300,new_nfe=0,diagnostic_nfe=0,optimizer_steps=0,gates={},gradients=[],results=[],
        gt_usage="Final predefined metrics only; no measurement generation, online control or selection",
        random_pairing="Same x_T and epoch/ordinal standard normals, NOT the same physical Brownian path",
        experiment_card=card,git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    started=time.perf_counter()
    def save():
        record["elapsed_seconds"]=time.perf_counter()-started
        (output_dir/"run.json").write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    save()
    print("=== Gaussian UNIT protocol: cached paper vs new uniform30 ===",flush=True)
    print("ONLY grid changes; all zeta/D init=.2, lr=.001,30x10. No paper rerun or new observation.",flush=True)
    print(f"New method NFE=300 maximum; diagnostic UNet NFE=0. Records: {output_dir}",flush=True)
    try:
        import torch
        from configs.config import IMG_SIZE,METRICS_CONFIG
        from modules.main_single import load_diffusion_model,load_image_as_tensor
        from modules.degradations import get_operator
        from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.diag_zaps_first_init import rng_preflight
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection
        from utils.diag_learning_rate_ablation import set_seed
        from utils.diag_zaps_sampler_parity import rng_state,same_rng
        from utils.metrics import compute_all_metrics
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=sys.version,
            gpu_name=torch.cuda.get_device_name(torch.device(args.device)),sampling_device=args.device,
            visible_cuda_count=torch.cuda.device_count(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        record["runtime"],record["gates"]["runtime"]=runtime,check_runtime(parent["runtime"],runtime)
        save()
        if not record["gates"]["runtime"]["passed"]:
            raise RuntimeError("Numerical environment changed; inspect differences before consuming 300 NFE")
        print("Checking cached parent identities (local file reads, no download)...",flush=True)
        paths=dict(checkpoint_sha256=Path(dps["checkpoint_path"]),image_sha256=Path(reference["baseline"]["image"]),
            measurement_sha256=source_dir/"gaussian_deblur/measurement.pt",core_sha256=PROJECTS/"modules/zaps_algorithm.py",
            operator_source_sha256=PROJECTS/"modules/degradations.py")
        if parent["identity"]!=reference["identity"]:
            raise RuntimeError("Parent/reference identities differ")
        for key,path in paths.items():
            if digest(path)!=parent["identity"][key]:
                raise RuntimeError(f"Source identity changed: {key}")
        for name,sha in parent["supporting_source_sha256"].items():
            if digest(PROJECTS/name)!=sha:
                raise RuntimeError(f"Supporting source changed: {name}")
        if digest(first_path)!=reference["source_first_sha256"]:
            raise RuntimeError("Original source record changed")
        archived_paths={"new_measurement_sha256":parent_dir/"measurement_unit.pt",
            "output_sha256":parent_dir/"unit_last_opt.pt","snapshot_sha256":parent_dir/"last_unroll_state.pt"}
        for key,path in archived_paths.items():
            if digest(path)!=parent[key]:
                raise RuntimeError(f"Unit parent tensor changed: {key}")
        record["identity"],record["unit_parent_tensor_sha256"]=parent["identity"],{key:parent[key] for key in archived_paths}
        record["actual_measurement"]=dict(path=str(archived_paths["new_measurement_sha256"]),
            sha256=parent["new_measurement_sha256"],domain="unit",sigma=.05,
            note="identity.measurement_sha256 retains the OLD model-domain provenance, not this run's y")
        record["supporting_source_sha256"]={name:digest(PROJECTS/name) for name in (
            "utils/diag_imagenet_gaussian_unit_uniform.py","utils/diag_imagenet_gaussian_unit_domain.py",
            "utils/diag_zaps_first_init.py","utils/diag_zaps_first_direction.py","utils/diag_zaps_sampler_parity.py",
            "utils/diag_zaps_optimized_trace.py","modules/diffusion_model.py","modules/wavelet.py","utils/metrics.py","configs/config.py")}
        record["gates"]["identity"]=dict(passed=True)
        y=torch.load(archived_paths["new_measurement_sha256"],map_location=args.device,weights_only=True)
        raw=torch.load(archived_paths["output_sha256"],map_location="cpu",weights_only=True)
        state=torch.load(archived_paths["snapshot_sha256"],map_location="cpu",weights_only=True)
        for name,tensor in (("y",y),("cached paper",raw),("x_T",state["init_noise"])):
            if tensor.shape!=(1,3,256,256) or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
                raise RuntimeError(f"Invalid parent tensor: {name}")
        manifest=json.loads((PROJECTS/"IMAGENET_BASELINE_MANIFEST_20261013.json").read_text(encoding="utf-8"))
        if state["tau"].tolist()!=manifest["archived_reference"]["actual_timesteps_ascending"]:
            raise RuntimeError("Parent is not the completed paper grid")
        if state["zeta"].tolist()!=parent["last_opt_used_zeta_ascending"]:
            raise RuntimeError("Parent PRE-update parameter snapshot differs")
        mapping=saved_cuda_selection(state,parent["runtime"]["sampling_device"],args.saved_cuda_index)
        base=get_operator("gaussian_deblur",device=args.device,**parent["task_config"])
        operator=unit_operator(base).to(args.device)
        before=rng_state(args.device)
        gate=operator_gate(operator,raw.to(args.device),state["init_noise"].to(args.device))
        gate["rng_unchanged"]=same_rng(before,rng_state(args.device))
        gate["passed"] &= gate["rng_unchanged"]
        record["gates"]["operator"]=gate
        save()
        if not gate["passed"]:
            raise RuntimeError("Affine operator gate failed; stop before UNet")
        # GT is not used to construct/change y, nor given to optimize.
        gt=load_image_as_tensor(reference["baseline"]["image"]).cpu()
        metrics=compute_all_metrics(raw,gt,lpips_net=METRICS_CONFIG["lpips_net"])
        with torch.no_grad():
            residual=y-operator.H(raw.to(args.device))
        if (not all(math.isclose(metrics[k],paper[k],rel_tol=2e-5,abs_tol=1e-6) for k in metrics)
                or not math.isclose(float(residual.double().square().mean()),paper["final_mse"],rel_tol=2e-5,abs_tol=1e-8)):
            raise RuntimeError("Cached paper metrics/loss mismatch; no paper rerun")
        record["gates"]["cached_metrics_loss"]=dict(passed=True)
        record["results"].append(dict(schedule="irregular_15_10_5",**metrics,final_mse=paper["final_mse"],
            final_residual=paper["final_residual"],historical_method_nfe=300,new_nfe=0))
        print("Parent/cache/operator gates passed; loading checkpoint, then noise preflight...",flush=True)
        model=load_diffusion_model("imagenet",args.device)
        model.model.eval()
        if (Path(model.ckpt_path).resolve()!=paths["checkpoint_sha256"].resolve() or model.num_steps!=1000
                or any(p.dtype!=torch.float32 for p in model.model.parameters())):
            raise RuntimeError("Need original fp32 model/checkpoint")
        zaps=LastUnrollSnapshotZAPS(model,operator,img_size=IMG_SIZE[0],**parent["config"])
        # Constructor remains original; replace ONLY the actual grid before any call.
        zaps.tau=torch.tensor(uniform_grid(),device=args.device,dtype=torch.long)
        if (not torch.equal(zaps.zeta,torch.full_like(zaps.zeta,.2))
                or not torch.equal(zaps.D,torch.full_like(zaps.D,.2))):
            raise RuntimeError("No trained parameter transfer or first-init adjustment allowed")
        preflight,expected_end=rng_preflight(1001,state,args.device,mapping["saved_sampling_index"])
        record["gates"]["noise_preflight"],record["rng_mapping"]=preflight,mapping
        save()
        if not preflight["passed"]:
            raise RuntimeError("Cannot pair original x_T/ordinal noise draws")
        print("Uniform actual ascending grid:",uniform_grid(),flush=True)
        visited=[]
        def count(module,inputs):
            record["new_nfe"]+=1
            if record["new_nfe"]>300:
                raise RuntimeError("300-NFE budget exceeded")
            visited.append(int(inputs[1].flatten()[0]))
        original_clip=torch.nn.utils.clip_grad_norm_
        def clip(parameters,max_norm,**kwargs):
            params=list(parameters)
            if len(params)!=2 or params[0] is not zaps.zeta or params[1] is not zaps.D or max_norm!=1:
                raise RuntimeError("Original joint optimizer/clip contract changed")
            total=original_clip(params,max_norm,**kwargs)
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite gradient")
            record["gradients"].append(dict(epoch=len(record["gradients"])+1,joint_norm=float(total)))
            return total
        hook=model.model.register_forward_pre_hook(count)
        set_seed(1001)
        train_started=time.perf_counter()
        try:
            with model_rng_guard(model.model,args.device) as guarded:
                with patch.object(torch.nn.utils,"clip_grad_norm_",side_effect=clip):
                    losses=zaps.optimize(y,verbose=True,x0_gt=None)
        finally:
            hook.remove()
        record["training_seconds"]=time.perf_counter()-train_started
        record["optimizer_steps"],record["loss_history"]=len(losses),losses
        last=zaps.audit_state
        index=torch.device(args.device).index
        index=torch.cuda.current_device() if index is None else index
        pairing=dict(x_T_equal=last is not None and torch.equal(last["init_noise"],state["init_noise"]),
            epoch10_cpu_rng_equal=last is not None and torch.equal(last["cpu_rng"],state["cpu_rng"]),
            epoch10_sampling_rng_equal=last is not None and torch.equal(last["cuda_rng"][index],state["cuda_rng"][mapping["saved_sampling_index"]]),
            end_rng_equal=same_rng(expected_end,rng_state(args.device)),times_equal=visited==uniform_grid()[::-1]*10,
            budget_equal=record["new_nfe"]==zaps._last_nfe==len(guarded)==300,
            epochs_equal=zaps.audit_calls==len(losses)==len(record["gradients"])==10)
        pairing["passed"]=all(pairing.values())
        record["gates"]["training_pairing"]=pairing
        save()
        if not pairing["passed"] or not all(math.isfinite(v) for v in losses):
            raise RuntimeError("Training pairing/finite loss gate failed")
        if any(p.grad is not None for p in model.model.parameters()):
            raise RuntimeError("Unexpected model-weight gradients")
        output=zaps._last_opt_x0.detach().cpu()
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite last_opt")
        torch.save(output,output_dir/"uniform_last_opt.pt")
        torch.save(last,output_dir/"last_unroll_state.pt")
        record["output_sha256"],record["snapshot_sha256"]=digest(output_dir/"uniform_last_opt.pt"),digest(output_dir/"last_unroll_state.pt")
        record["last_opt_used_zeta_ascending"],record["post_update_zeta_ascending"]=last["zeta"].tolist(),zaps.zeta.detach().cpu().tolist()
        record["post_update_note"]="Post-update parameters did not generate last_opt; replay the saved PRE snapshot"
        with torch.no_grad():
            residual=y-operator.H(output.to(args.device))
        metrics=compute_all_metrics(output,gt,lpips_net=METRICS_CONFIG["lpips_net"])
        record["results"].append(dict(schedule="uniform_30",**metrics,final_mse=float(residual.double().square().mean()),
            final_residual=float(residual.double().norm()),historical_method_nfe=0,new_nfe=record["new_nfe"]))
        record["decision"]=interpret(record["results"][0],metrics)
        with (output_dir/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(record["results"][0]));writer.writeheader();writer.writerows(record["results"])
        record["status"]="complete"
        save()
    except Exception as error:
        record["status"],record["error"]="failed",repr(error)
        save()
        raise
    print("\n=== Same-unit-protocol paper vs uniform summary ===",flush=True)
    for r in record["results"]:
        print(f"{r['schedule']:>22} PSNR={r['psnr']:.4f} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f} newNFE={r['new_nfe']}",flush=True)
    print("Decision (irregular minus uniform):",json.dumps(record["decision"],ensure_ascii=False),flush=True)
    print(f"New UNet NFE={record['new_nfe']}; records: {output_dir}",flush=True)


if __name__=="__main__":
    main()
