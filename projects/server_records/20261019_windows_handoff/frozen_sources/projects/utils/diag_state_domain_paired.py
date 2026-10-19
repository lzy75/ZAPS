"""Fixed30 state tests: ImageNet UNIT Gaussian, FFHQ domain x state factorial.

One frozen candidate per dataset. No core/default edits or LR/init/noise scans.
ImageNet: 90 diagnostic +600 method NFE. FFHQ:180 diagnostic +1200 method NFE.
"""
import argparse
import csv
from dataclasses import asdict
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

PROJECTS=Path(__file__).resolve().parents[1]
ROOT=PROJECTS.parent
sys.path.insert(0,str(PROJECTS))
from utils.diag_imagenet_gaussian_unit_domain import digest,unit_operator,validate_reference
from utils.diag_imagenet_gaussian_unit_uniform import validate_parent,check_runtime,operator_gate
from utils.diag_imagenet_tuned_state import NoiseProfileScheduler,parity_decision,finite


def base_config(dataset):
    if dataset not in ("imagenet","ffhq"):
        raise ValueError("Need an explicit supported dataset")
    return dict(num_steps=30,schedule=[15,10,5],timestep_spacing="linear",schedule_power=2.,
        num_epochs=10,lr=.001 if dataset=="imagenet" else .01,
        zeta_init=.2 if dataset=="imagenet" else .1,d_init=.2,eta=1.,
        use_learned_var=False,sampler_mode="ddpm",wave="db4",level=3,surrogate_score_jacobian=False)


def arms_for(dataset):
    if dataset=="imagenet":return [("unit","null",0.),("unit","state",.2)]
    if dataset=="ffhq":return [("model","null",0.),("model","state",.2),("unit","null",0.),("unit","state",.2)]
    raise ValueError("Unknown dataset")


def saved_rng_view(rng):
    """Convert the live rng_state schema to the archived restore-helper schema."""
    return dict(cpu_rng=rng["cpu"],cuda_rng=rng["cuda"])


def scheduler_for(dataset,grid,alpha,response):
    if dataset not in ("imagenet","ffhq"):raise ValueError("Unknown dataset")
    if response not in (0.,.2):raise ValueError("This card has no response scan")
    if dataset=="imagenet":return NoiseProfileScheduler(grid,alpha,response)
    from modules.adaptive_scheduler import BudgetedStateAwareScheduler,BudgetedSchedulerConfig
    return BudgetedStateAwareScheduler(grid,BudgetedSchedulerConfig(
        residual_weight=.8,cosine_weight=.2,response_strength=response,residual_mode="reference_profile",
        profile_warmup_epochs=1,profile_center_decay=.8,profile_residual_scale=.15,
        profile_cosine_scale=.2,profile_cosine_gate=.35,profile_gate_mode="veto_only",
        profile_cosine_feature_mode="global",profile_cosine_detail_weight=.7,
        weight_mode="identity",mod_min=.8,mod_max=1.2))


def assess_state(null,state,changed_points,parent=None):
    delta={k:state[k]-null[k] for k in ("psnr","ssim","lpips")}
    drift={k:null[k]-parent[k] for k in delta} if parent is not None else None
    compatible=drift is None or (abs(drift["psnr"])<=.05 and abs(drift["ssim"])<=.005 and abs(drift["lpips"])<=.005)
    if not compatible:tag="INCONCLUSIVE_NULL_DRIFT"
    elif changed_points==0:tag="STATE_NOT_ACTIVATED"
    elif delta["psnr"]>=.1 and delta["ssim"]>=-.002 and delta["lpips"]<=.002:
        tag="SINGLE_IMAGE_STATE_GAIN_CANDIDATE"
    elif delta["psnr"]>0 and (delta["ssim"]<-.002 or delta["lpips"]>.002):tag="PSNR_ONLY_TRADEOFF"
    else:tag="NO_CLEAR_COMPREHENSIVE_STATE_GAIN"
    return dict(classification=tag,state_minus_null=delta,null_drift_vs_cache=drift,
        changed_points_after_pilot=changed_points,author_parity=False,benchmark_reproduced=False,
        limits="Single image/seed; engineering thresholds, not statistical proof")


def summarize(rows,arms,parent=None):
    grouped={domain:{r["policy"]:r for r in rows if r["domain"]==domain} for domain in {r["domain"] for r in rows}}
    decisions={domain:assess_state(pair["null"],pair["state"],arms[domain+"_state"]["changed_points_after_pilot"],parent)
               for domain,pair in grouped.items()}
    result=dict(state_by_domain=decisions,next_action="Discuss outputs/traces; no automatic scan or sample expansion")
    if set(grouped)=={"model","unit"}:
        result["unit_minus_model_null"]={k:grouped["unit"]["null"][k]-grouped["model"]["null"][k] for k in ("psnr","ssim","lpips")}
        d=result["unit_minus_model_null"]
        if d["psnr"]>=.1 and d["ssim"]>=-.002 and d["lpips"]<=.002:
            domain_tag="UNIT_PROTOCOL_QUALITY_ADVANTAGE_ON_THIS_IMAGE"
        elif d["psnr"]<=-.1 and d["ssim"]<=.002 and d["lpips"]>=-.002:
            domain_tag="MODEL_PROTOCOL_QUALITY_ADVANTAGE_ON_THIS_IMAGE"
        elif abs(d["psnr"])<.1 and abs(d["ssim"])<=.002 and abs(d["lpips"])<=.002:
            domain_tag="NO_MATERIAL_NULL_QUALITY_DIFFERENCE"
        else:domain_tag="MEASUREMENT_DOMAIN_QUALITY_TRADEOFF"
        result["domain_null_quality_classification"]=domain_tag
        result["state_domain_interaction"]={k:decisions["unit"]["state_minus_null"][k]-decisions["model"]["state_minus_null"][k]
                                            for k in ("psnr","ssim","lpips")}
        result["domain_effect_limits"]="Physical SNR, loss/guidance units change together; legacy T remains empirical"
    return result


def observed_class():
    """Passive per-epoch tracing; inherit the ACTUAL optimizer and adaptive graph."""
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.diag_zaps_sampler_parity import rng_state
    class ObservedZAPS(ZAPS):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.audit_calls=0;self.audit_state=None;self.audit_trace=[]

        def _reverse_diffusion_adaptive(self,y,scheduler,*args,**kwargs):
            self.audit_calls+=1
            if self.audit_calls==self.num_epochs:
                init=kwargs.get("init_noise")
                if init is None:raise RuntimeError("Need fixed optimize x_T")
                rng=rng_state(self.device)
                self.audit_state=dict(zeta=self.zeta.detach().cpu().clone(),D=self.D.detach().cpu().clone(),
                    tau=self.tau.detach().cpu().clone(),init_noise=init.detach().cpu().clone(),
                    cpu_rng=rng["cpu"],cuda_rng=rng["cuda"],sampling_device=str(self.device),
                    sampling_cuda_index=(torch.device(self.device).index if torch.device(self.device).index is not None
                        else torch.cuda.current_device()) if str(self.device).startswith("cuda") else None,
                    scheduler_pilot_log_residual=list(getattr(scheduler,"pilot_log_r",[])),
                    scheduler_pilot_cosine=list(getattr(scheduler,"pilot_cos",[])),
                    scheduler_reference_residual=list(scheduler._profile_reference_residual),
                    scheduler_reference_cosine=list(scheduler._profile_reference_cosine),
                    scheduler_completed_unrolls=scheduler._completed_unrolls,scheduler_used=scheduler.used,
                    scheduler_config=asdict(scheduler.cfg))
            kwargs["record_indicators"]=True
            output=super()._reverse_diffusion_adaptive(y,scheduler,*args,**kwargs)
            self.audit_trace.extend(dict(row,epoch=self.audit_calls) for row in self._indicator_log)
            return output
    return ObservedZAPS


def gate_unroll(construct,scheduler_factory,y,initial,start_rng,device,record):
    """Three original-config 30-step forward/VJP probes, never train weights."""
    import torch
    import torch.nn.functional as F
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import restore_sampling_rng
    z=construct();current=torch.device(device).index
    current=torch.cuda.current_device() if current is None and str(device).startswith("cuda") else current
    def restore():
        if str(device).startswith("cuda"):restore_sampling_rng(saved_rng_view(start_rng),device,current)
        else:torch.set_rng_state(start_rng["cpu"])
    def probe(adaptive):
        restore();s=scheduler_factory(z,0.) if adaptive else None
        visited=[];indices=[]
        def before(module,inputs):
            record["diagnostic_nfe"]+=1
            if record["diagnostic_nfe"]>record.get("planned_diagnostic_nfe",90):raise RuntimeError("Diagnostic budget exceeded")
            t=int(inputs[1].flatten()[0]);visited.append(t)
            if s is not None and hasattr(s,"set_time"):s.set_time(t)
        original=z._tweedie_estimate
        def tweedie(x,eps,ab,index):indices.append(index);return original(x,eps,ab,index)
        hook=z.dm.model.register_forward_pre_hook(before)
        try:
            with model_rng_guard(z.dm.model,device),patch.object(z,"_tweedie_estimate",side_effect=tweedie):
                counter=[0]
                if adaptive:out=z._reverse_diffusion_adaptive(y,s,nfe_counter=counter,init_noise=initial)
                else:out=z._reverse_diffusion(y,nfe_counter=counter,init_noise=initial)
                loss=F.mse_loss(z.A.H(out),y)
                grads=torch.autograd.grad(loss,(z.zeta,z.D))
            if any(p.grad is not None for p in z.dm.model.parameters()):raise RuntimeError("Gate populated score weight gradients")
            return dict(output=out.detach().cpu(),zeta_grad=grads[0].detach().cpu(),D_grad=grads[1].detach().cpu(),
                visited=visited,indices=indices,nfe=counter[0],rng=rng_state(device),loss=float(loss.detach()))
        finally:hook.remove()
    a,b,c=probe(False),probe(False),probe(True)
    def error(left,right):return {k:float((left[k].double()-right[k].double()).norm()/left[k].double().norm().clamp(min=1e-12))
                                 for k in ("output","zeta_grad","D_grad")}
    metadata=all(v["visited"]==z.tau.flip(0).tolist() and v["indices"]==list(range(29,-1,-1)) and v["nfe"]==30
                 and math.isfinite(v["loss"]) for v in (a,b,c)) and same_rng(a["rng"],b["rng"]) and same_rng(a["rng"],c["rng"])
    return parity_decision(error(a,b),error(a,c),metadata)


def write_record(directory,record):
    (directory/"run.json").write_text(json.dumps(finite(record),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for filename,rows in (("summary.csv",record["results"]),("trace.csv",record["trace"])):
        if rows:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            with (directory/filename).open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(finite(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("imagenet","ffhq"),required=True)
    parser.add_argument("--unit-run-dir",default=None)
    parser.add_argument("--image",default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--saved-cuda-index",type=int,default=None)
    args=parser.parse_args()
    if not args.device.startswith("cuda"):parser.error("Production experiments require CUDA")
    if args.dataset=="imagenet" and (not args.unit_run_dir or args.image or args.seed is not None):
        parser.error("ImageNet uses exact parent image/seed: specify only --unit-run-dir")
    if args.dataset=="ffhq" and (not args.image or args.seed!=1000 or args.unit_run_dir or args.saved_cuda_index is not None):
        parser.error("FFHQ card requires --image and --seed 1000; no ImageNet parent")
    parent=None;reference=None
    if args.dataset=="imagenet":
        parent_dir=Path(args.unit_run_dir).resolve();parent_path=parent_dir/"run.json"
        parent=json.loads(parent_path.read_text(encoding="utf-8"));validate_parent(parent)
        reference_path=Path(parent["source_reference"])
        reference=json.loads(reference_path.read_text(encoding="utf-8"));validate_reference(reference)
        if digest(reference_path)!=parent["source_reference_sha256"] or parent["config"]!=base_config("imagenet"):
            raise RuntimeError("Need the unmodified original unit parent")
        image=reference["baseline"]["image"];seed=1001
        root=parent_dir
    else:image=str(Path(args.image).resolve());seed=1000;root=PROJECTS/"results/diag_state_domain_paired/ffhq"
    directory=root/("state_domain_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True,exist_ok=False)
    card=(PROJECTS/"experiment_cards/A_state_domain_paired.md").read_text(encoding="utf-8")
    (directory/"experiment_card.md").write_text(card,encoding="utf-8")
    domains=["unit"] if args.dataset=="imagenet" else ["model","unit"]
    config=base_config(args.dataset)
    record=dict(experiment="fixed30_state_measurement_domain_paired",status="running",arguments=vars(args),
        dataset=args.dataset,image=image,seed=seed,config=config,arms={},gates={},results=[],trace=[],
        planned_method_nfe=600 if args.dataset=="imagenet" else 1200,
        planned_diagnostic_nfe=90*len(domains),method_nfe=0,diagnostic_nfe=0,
        gt_usage="FFHQ synthetic measurement and final evaluation only; ImageNet final evaluation only; optimize x0_gt=None",
        random_pairing="Same x_T/epoch/call normals; not equal physical-time Brownian paths",experiment_card=card,
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    started=time.perf_counter()
    def save():record["elapsed_seconds"]=time.perf_counter()-started;write_record(directory,record)
    save()
    print(f"=== {args.dataset}: fixed30 state/domain experiment ===\nRecords: {directory}",flush=True)
    print(f"Method budget={record['planned_method_nfe']}; diagnostic budget={record['planned_diagnostic_nfe']}; no extra final sample.",flush=True)
    try:
        import torch
        from configs.config import IMG_SIZE,METRICS_CONFIG
        from modules.main_single import load_diffusion_model,load_image_as_tensor
        from modules.degradations import get_operator
        from utils.diag_learning_rate_ablation import set_seed
        from utils.diag_zaps_first_init import rng_preflight
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.diag_zaps_sampler_parity import rng_state,same_rng
        from utils.metrics import compute_all_metrics
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=sys.version,
            gpu_name=torch.cuda.get_device_name(torch.device(args.device)),sampling_device=args.device,
            visible_cuda_count=torch.cuda.device_count(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        record["runtime"]=runtime
        if parent is not None:
            record["parent_run"],record["parent_run_sha256"]=str(parent_path),digest(parent_path)
            record["gates"]["runtime"]=check_runtime(parent["runtime"],runtime)
            save()
            if not record["gates"]["runtime"]["passed"]:raise RuntimeError("Parent/current numerical environment differs")
            for name,sha in parent["supporting_source_sha256"].items():
                if digest(PROJECTS/name)!=sha:raise RuntimeError(f"Parent source changed: {name}")
            for name,sha in reference["current_source_sha256"].items():
                if digest(PROJECTS/name)!=sha:raise RuntimeError(f"Reference helper changed: {name}")
            for key,name in (("new_measurement_sha256","measurement_unit.pt"),("output_sha256","unit_last_opt.pt"),("snapshot_sha256","last_unroll_state.pt")):
                if digest(parent_dir/name)!=parent[key]:raise RuntimeError(f"Parent tensor changed: {name}")
            for key,path in (("image_sha256",Path(image)),("core_sha256",PROJECTS/"modules/zaps_algorithm.py"),
                             ("operator_source_sha256",PROJECTS/"modules/degradations.py")):
                if digest(path)!=parent["identity"][key]:raise RuntimeError(f"Identity changed: {key}")
        print("Loading checkpoint and hashing actual inputs/sources (no weight download)...",flush=True)
        model=load_diffusion_model(args.dataset,args.device);model.model.eval()
        checkpoint_sha=digest(model.ckpt_path)
        if model.num_steps!=1000 or any(p.dtype!=torch.float32 for p in model.model.parameters()):raise RuntimeError("Need original fp32 1000-step model")
        if parent is not None and checkpoint_sha!=parent["identity"]["checkpoint_sha256"]:raise RuntimeError("Checkpoint changed")
        record["identity"]=dict(checkpoint_sha256=checkpoint_sha,image_sha256=digest(image),
            source_sha256={name:digest(PROJECTS/name) for name in (
                "modules/zaps_algorithm.py","modules/degradations.py","modules/diffusion_model.py","modules/wavelet.py",
                "modules/main_single.py","modules/dataset_loader.py",
                "modules/adaptive_scheduler.py","utils/diag_state_domain_paired.py","utils/diag_imagenet_tuned_state.py",
                "utils/diag_ffhq_regression.py","utils/diag_ffhq_timestep_ablation.py","utils/metrics.py",
                "utils/diag_imagenet_gaussian_unit_domain.py","utils/diag_imagenet_gaussian_unit_uniform.py",
                "utils/diag_zaps_first_init.py","utils/diag_zaps_first_direction.py","utils/diag_zaps_sampler_parity.py")})
        if args.dataset=="imagenet":
            task="gaussian_deblur";base=get_operator(task,device=args.device,**parent["task_config"])
            operators={"unit":unit_operator(base).to(args.device)}
            measurements={"unit":torch.load(parent_dir/"measurement_unit.pt",map_location=args.device,weights_only=True)}
            archived=torch.load(parent_dir/"last_unroll_state.pt",map_location="cpu",weights_only=True)
            if archived["zeta"].tolist()!=parent["last_opt_used_zeta_ascending"]:raise RuntimeError("Parent PRE parameters mismatch")
            grid=archived["tau"].to(args.device);mapping=saved_cuda_selection(archived,parent["runtime"]["sampling_device"],args.saved_cuda_index)
            manifest=json.loads((PROJECTS/"IMAGENET_BASELINE_MANIFEST_20261013.json").read_text(encoding="utf-8"))
            if grid.cpu().tolist()!=manifest["archived_reference"]["actual_timesteps_ascending"]:raise RuntimeError("Need original paper grid")
            cache=torch.load(parent_dir/"unit_last_opt.pt",map_location=args.device,weights_only=True)
            record["gates"]["operator"]=operator_gate(operators["unit"],cache,archived["init_noise"].to(args.device))
            if not record["gates"]["operator"]["passed"]:raise RuntimeError("Unit Gaussian operator gate failed")
            preflight,expected_end=rng_preflight(seed,archived,args.device,mapping["saved_sampling_index"])
            record["gates"]["noise_preflight"],record["rng_mapping"]=preflight,mapping
            if not preflight["passed"]:raise RuntimeError("Archived x_T/noise prefix mismatch")
            record["guidance_mapping"]="exact .5H^T; unchanged Gaussian"
            record["actual_observation_source"]=dict(path=str(parent_dir/"measurement_unit.pt"),sha256=parent["new_measurement_sha256"])
            gt=None
        else:
            from utils.diag_ffhq_regression import TransposeMode
            from utils.diag_ffhq_timestep_ablation import karras_timesteps
            task="super_resolution";base=get_operator(task,device=args.device,scale_factor=4,noise_sigma=0.)
            legacy=TransposeMode(base,"legacy_bicubic").to(args.device)
            operators={"model":legacy,"unit":unit_operator(legacy).to(args.device)}
            gt=load_image_as_tensor(image).to(args.device)
            set_seed(seed)
            with torch.no_grad():
                clean=base.H(gt);xi=torch.randn_like(clean);noise=.05*xi
                measurements={"model":clean+noise,"unit":operators["unit"].H(gt)+noise}
                affine=operators["unit"].H(gt)-(.5*base.H(gt)+.5*base.H(torch.ones_like(gt)))
                scaling=operators["unit"].transpose(noise)-.5*legacy.transpose(noise)
                ok=float(affine.abs().max())<=2e-5 and float(scaling.abs().max())<=2e-5
            record["gates"]["operator"]=dict(passed=ok,affine_max_error=float(affine.abs().max()),
                empirical_mapping_scaling_max_error=float(scaling.abs().max()),exact_adjoint_claim=False)
            if not ok:raise RuntimeError("FFHQ affine/empirical mapping contract failed")
            grid=karras_timesteps(model.alphas_cumprod,30,3.).to(args.device)
            torch.save(xi.cpu(),directory/"standard_measurement_noise.pt")
            record["guidance_mapping"]="legacy_bicubic empirical T in model domain; .5T in unit domain; NOT exact H^T"
            set_seed(seed);fixed=torch.randn(1,3,IMG_SIZE[0],IMG_SIZE[0],device=args.device)
            for _ in range(290):torch.randn_like(fixed)
            expected_end=rng_state(args.device)
        if len(grid)!=30 or grid[0].item()!=0 or grid[-1].item()!=999 or not bool((grid[1:]>grid[:-1]).all()):raise RuntimeError("Invalid nominal grid")
        record["nominal_timesteps_ascending"]=grid.cpu().tolist()
        record["task"],record["measurement_protocol"]=task,dict(model_state=[-1,1],numeric_sigma=.05,
            domains=domains,unit_sigma=.05,old_model_equivalent_unit_sigma=.025,
            physical_SNR_changed_between_domains=args.dataset=="ffhq",observation_clipped=False)
        record["task_config"]=parent["task_config"] if parent is not None else dict(scale_factor=4,numeric_noise_sigma=.05)
        record["reference_alignment"]="log_sigma_u + u-scaled EMA" if args.dataset=="imagenet" else "historical reverse-call-position reference"
        for domain,y in measurements.items():
            if not torch.isfinite(y).all() or y.dtype!=torch.float32:raise RuntimeError("Invalid measurement")
            torch.save(y.cpu(),directory/(domain+"_measurement.pt"))
        record["measurement_sha256"]={d:digest(directory/(d+"_measurement.pt")) for d in domains}
        cls=observed_class()
        def construct(domain):
            set_seed(seed);z=cls(model,operators[domain],img_size=IMG_SIZE[0],**config);z.tau=grid.clone()
            if (not torch.equal(z.zeta,torch.full_like(z.zeta,config["zeta_init"])) or
                not torch.equal(z.D,torch.full_like(z.D,.2))):raise RuntimeError("Original all-step init changed")
            return z
        def scheduler(z,response):return scheduler_for(args.dataset,z.tau.flip(0).tolist(),model.alphas_cumprod.cpu().tolist(),response)
        set_seed(seed);initial=torch.randn(1,3,IMG_SIZE[0],IMG_SIZE[0],device=args.device);start_rng=rng_state(args.device)
        torch.save(initial.cpu(),directory/"initial_noise.pt")
        record["initial_noise_sha256"]=digest(directory/"initial_noise.pt")
        if parent is not None and not torch.equal(initial.cpu(),archived["init_noise"]):raise RuntimeError("Parent x_T mismatch")
        record["gates"]["unroll_parity"]={}
        for domain in domains:
            print(f"\nChecking {domain} fixed-repeat/null forward+VJP parity: 90 diagnostic NFE...",flush=True)
            gate=gate_unroll(lambda:construct(domain),scheduler,measurements[domain],initial,start_rng,args.device,record)
            record["gates"]["unroll_parity"][domain]=gate;save()
            if not gate["passed"]:raise RuntimeError(f"{domain} null parity failed; stop before methods")
        if record["diagnostic_nfe"]!=record["planned_diagnostic_nfe"]:raise RuntimeError("Diagnostic accounting mismatch")
        for domain,policy,response in arms_for(args.dataset):
            key=domain+"_"+policy;z=construct(domain);s=scheduler(z,response);visited=[];gradients=[]
            record.setdefault("scheduler_config",{})[key]=asdict(s.cfg)
            def before(module,inputs):
                record["method_nfe"]+=1
                if record["method_nfe"]>record["planned_method_nfe"] or len(visited)>=300:raise RuntimeError("Method budget exceeded")
                t=int(inputs[1].flatten()[0]);visited.append(t)
                if hasattr(s,"set_time"):s.set_time(t)
            clip_original=torch.nn.utils.clip_grad_norm_
            def clip(parameters,max_norm,**kwargs):
                params=list(parameters)
                if len(params)!=2 or params[0] is not z.zeta or params[1] is not z.D or max_norm!=1:raise RuntimeError("Joint optimizer/clip contract changed")
                total=clip_original(params,max_norm,**kwargs)
                if not torch.isfinite(total):raise RuntimeError("Non-finite gradient")
                gradients.append(dict(epoch=len(gradients)+1,joint_norm=float(total)));return total
            hook=model.model.register_forward_pre_hook(before);set_seed(seed);arm_started=time.perf_counter()
            print(f"\n--- {key}: response={response}; all initial zeta={config['zeta_init']}; lr={config['lr']} ---",flush=True)
            try:
                with model_rng_guard(model.model,args.device) as guarded,patch.object(torch.nn.utils,"clip_grad_norm_",side_effect=clip):
                    losses=z.optimize(measurements[domain],verbose=True,x0_gt=None,scheduler=s)
            finally:hook.remove()
            grids=[visited[k:k+30] for k in range(0,len(visited),30)];nominal=grid.flip(0).tolist()
            valid=(len(grids)==10 and all(len(g)==30 and g[0]==999 and g[-1]==0 and all(a>b for a,b in zip(g,g[1:])) for g in grids))
            pairing=same_rng(expected_end,rng_state(args.device)) and torch.equal(z._last_opt_noise.cpu(),initial.cpu())
            if (not valid or not pairing or grids[0]!=nominal or len(guarded)!=300 or z._last_nfe!=300 or
                len(losses)!=10 or len(gradients)!=10 or len(z.audit_trace)!=300 or z.audit_state is None or
                [r["parameter_index"] for r in z.audit_trace]!=list(range(29,-1,-1))*10 or
                any(not math.isfinite(v) for v in losses) or (policy=="null" and any(g!=nominal for g in grids))):
                raise RuntimeError(f"{key} training metadata/noise/finite gate failed")
            if any(p.grad is not None for p in model.model.parameters()):raise RuntimeError("Score weights received gradients")
            output=z._last_opt_x0.detach().cpu()
            if not torch.isfinite(output).all():raise RuntimeError("Non-finite output")
            torch.save(output,directory/(key+"_last_opt.pt"));torch.save(z.audit_state,directory/(key+"_pre_last_state.pt"))
            torch.save(dict(zeta=z.zeta.detach().cpu(),D=z.D.detach().cpu()),directory/(key+"_post_params.pt"))
            changed=sum(a!=b for g in grids[1:] for a,b in zip(g,nominal))
            record["arms"][key]=dict(method_nfe=300,seconds=time.perf_counter()-arm_started,loss_history=losses,
                gradients=gradients,grids=grids,changed_points_after_pilot=changed,noise_pairing_passed=pairing,
                output_sha256=digest(directory/(key+"_last_opt.pt")),pre_snapshot_sha256=digest(directory/(key+"_pre_last_state.pt")),
                parameter_semantics="Use PRE snapshot for last_opt; post_params do NOT generate this output")
            record["trace"].extend(dict(row,domain=domain,policy=policy) for row in z.audit_trace)
            save();del z,output
        # Final evaluation only, after all predefined arms; no best output selection.
        gt=load_image_as_tensor(image).cpu()
        cached_metrics=None
        if parent is not None:
            cached_metrics=compute_all_metrics(torch.load(parent_dir/"unit_last_opt.pt",map_location="cpu",weights_only=True),gt,lpips_net=METRICS_CONFIG["lpips_net"])
            expected=parent["results"][1]
            if not all(math.isclose(cached_metrics[k],expected[k],rel_tol=2e-5,abs_tol=1e-6) for k in cached_metrics):raise RuntimeError("Parent cached metrics mismatch")
            record["cached_parent_metrics"]=cached_metrics
        for domain,policy,response in arms_for(args.dataset):
            key=domain+"_"+policy;output=torch.load(directory/(key+"_last_opt.pt"),map_location="cpu",weights_only=True)
            metrics=compute_all_metrics(output,gt,lpips_net=METRICS_CONFIG["lpips_net"])
            loss=record["arms"][key]["loss_history"][-1]
            record["results"].append(dict(domain=domain,policy=policy,**metrics,method_nfe=300,
                native_domain_mse=loss,native_domain_residual=math.sqrt(loss*measurements[domain].numel())))
        if record["method_nfe"]!=record["planned_method_nfe"]:raise RuntimeError("Total method accounting mismatch")
        record["decision"]=summarize(record["results"],record["arms"],cached_metrics)
        record["status"]="complete";save()
    except Exception as error:
        record["status"],record["error"]="failed",repr(error);save();raise
    print("\n=== Paired state/domain summary ===",flush=True)
    for r in record["results"]:
        null=next(a for a in record["results"] if a["domain"]==r["domain"] and a["policy"]=="null")
        print(f"{r['domain']:>7} {r['policy']:>6} PSNR={r['psnr']:.4f} dNull={r['psnr']-null['psnr']:+.4f} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f} NFE=300",flush=True)
    print("Decision:",json.dumps(finite(record["decision"]),ensure_ascii=False),flush=True)
    print("Native-domain MSE is NOT directly comparable between model/unit rows.",flush=True)
    print(f"Method NFE={record['method_nfe']}; diagnostic NFE={record['diagnostic_nfe']}; records: {directory}",flush=True)


if __name__=="__main__":main()
