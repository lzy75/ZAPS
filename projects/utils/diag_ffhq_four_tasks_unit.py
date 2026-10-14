"""FFHQ four tasks, paper major settings and UNIT physical measurements.

No state/LR/initial-grid tuning. EXACT adjoints, original motion Kernel generator.
Single image=1200 method NFE; explicit N-image subset=1200*N. No extra UNet.
"""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS=Path(__file__).resolve().parents[1]
ROOT=PROJECTS.parent
sys.path.insert(0,str(PROJECTS))
from utils.diag_imagenet_gaussian_unit_domain import digest,unit_operator
from utils.diag_imagenet_gaussian_unit_uniform import operator_gate
from utils.diag_state_domain_paired import base_config
from utils.diag_imagenet_tuned_state import finite

TASKS=("gaussian_deblur","inpainting","motion_deblur","super_resolution")
PAPER={"gaussian_deblur":dict(psnr=26.06,ssim=.757,lpips=.121),
       "inpainting":dict(psnr=27.79,ssim=.813,lpips=.078),
       "motion_deblur":dict(psnr=24.16,ssim=.709,lpips=.141),
       "super_resolution":dict(psnr=26.63,ssim=.768,lpips=.104)}


def config_for(task):
    if task not in TASKS:raise ValueError("Unknown paper task")
    config=base_config("imagenet")
    config["zeta_init"]=.2 if task in ("gaussian_deblur","motion_deblur") else .1
    return config


def select_images(image,data_dir,max_images,start_index):
    if image:
        if data_dir or max_images is not None or start_index!=0:raise ValueError("Single image cannot also select a subset")
        paths=[Path(image).resolve()]
    else:
        if not data_dir or type(max_images) is not int or not 1<=max_images<=20 or start_index<0:
            raise ValueError("Explicit data-dir and max-images 1..20 required; no automatic expansion")
        available=sorted(p.resolve() for p in Path(data_dir).rglob("*") if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg",".webp"))
        paths=available[start_index:start_index+max_images]
        if len(paths)!=max_images:raise ValueError("Not enough images for the requested fixed manifest")
    if not all(p.is_file() for p in paths):raise ValueError("Selected image missing")
    return paths


def summarize(rows,count):
    summaries=[]
    for task in TASKS:
        selected=[r for r in rows if r["task"]==task]
        if len(selected)!=count:raise ValueError("Incomplete task coverage: cannot summarize partial run")
        item=dict(task=task,images=count,paper_images=1000)
        for metric in ("psnr","ssim","lpips","float_psnr"):
            values=[r[metric] for r in selected]
            item[metric+"_mean"]=statistics.mean(values)
            item[metric+"_std"]=statistics.stdev(values) if count>1 else None
            if metric in PAPER[task]:
                item["paper_"+metric]=PAPER[task][metric]
                item["delta_"+metric+"_vs_paper_mean"]=item[metric+"_mean"]-PAPER[task][metric]
        summaries.append(item)
    return summaries


def original_motion_kernel(seed=0):
    """Vendored original generator; writable bitmap copy is a NumPy/PIL fix only."""
    import numpy as np
    path=ROOT/"DPS/motionblur/motionblur.py"
    state=np.random.get_state()
    try:
        spec=importlib.util.spec_from_file_location("_ffhq_paper_motion_kernel",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        np.random.seed(seed)
        generated=module.Kernel(size=(61,61),intensity=.5)
        generated._createKernel()
        array=np.array(generated.kernel_image,dtype=np.float32,copy=True)
        if array.shape!=(61,61) or not np.isfinite(array).all() or float(array.sum())<=0:
            raise RuntimeError("Original motion generator produced an invalid kernel; NO fallback")
        array/=array.sum()
        return array
    finally:np.random.set_state(state)


def motion_operator(kernel,device):
    import torch
    import torch.nn.functional as F
    class OriginalKernelConv(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer("kernel",torch.as_tensor(kernel).reshape(1,1,61,61).to(device))
        def H(self,x):return F.conv2d(x,self.kernel.expand(x.shape[1],1,61,61),padding=30,groups=x.shape[1])
        def transpose(self,r):return F.conv2d(r,self.kernel.flip((-2,-1)).expand(r.shape[1],1,61,61),padding=30,groups=r.shape[1])
        def forward(self,x):return self.H(x)
    return OriginalKernelConv()


def make_base(task,device,gt,mask_seed,kernel):
    import torch
    from modules.degradations import get_operator
    if task=="gaussian_deblur":return get_operator(task,device=device,kernel_size=61,sigma=3.,noise_sigma=0.)
    if task=="super_resolution":return get_operator(task,device=device,scale_factor=4,noise_sigma=0.)
    if task=="motion_deblur":return motion_operator(kernel,device)
    base=get_operator("inpainting",device=device,random_ratio=.7,mode="random",box_size=0,noise_sigma=0.,seed=-1)
    generator=torch.Generator(device=device);generator.manual_seed(mask_seed)
    base._mask_cache=(torch.rand(gt.shape[0],1,*gt.shape[-2:],device=device,generator=generator)>=.7).to(gt.dtype)
    return base


def save_record(directory,record):
    (directory/"run.json").write_text(json.dumps(finite(record),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for name,rows in (("per_image.csv",record["results"]),("summary.csv",record.get("summary",[]))):
        if rows:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            with (directory/name).open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(finite(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--image");selection.add_argument("--data-dir")
    parser.add_argument("--max-images",type=int,default=None)
    parser.add_argument("--start-index",type=int,default=0)
    parser.add_argument("--seed-base",type=int,default=1000)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if not args.device.startswith("cuda"):parser.error("Production run needs CUDA; use CPU unit tests separately")
    try:images=select_images(args.image,args.data_dir,args.max_images,args.start_index)
    except ValueError as error:parser.error(str(error))
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    directory=PROJECTS/"results/diag_ffhq_four_tasks_unit"/stamp;directory.mkdir(parents=True,exist_ok=False)
    card=(PROJECTS/"experiment_cards/R_ffhq_four_tasks_unit.md").read_text(encoding="utf-8")
    (directory/"experiment_card.md").write_text(card,encoding="utf-8")
    record=dict(experiment="ffhq_four_task_public_settings_unit_measurements",status="running",arguments=vars(args),
        configs={task:config_for(task) for task in TASKS},paper_reference=PAPER,
        paper_source="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114.pdf",
        supplement_source="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114-supp.pdf",
        input_manifest=[dict(path=str(p),sha256=digest(p),seed=args.seed_base+args.start_index+i,
            mask_seed=args.seed_base+args.start_index+i+2000000,
            measurement_seed=args.seed_base+args.start_index+i+1000000) for i,p in enumerate(images)],
        planned_method_nfe=1200*len(images),method_nfe=0,diagnostic_nfe=0,gates={},arms={},results=[],
        resolved_protocol=dict(dataset="ffhq",measurement_domain="unit",model_state_domain=[-1,1],measurement_sigma=.05,
            A="H((x+1)/2)",jacobian_adjoint=".5 H^T",sr_transpose="exact",schedule="paper15/10/5",
            state=False,extra_final_sample=False,final_mode="last_opt_pre_final_adam_update",
            motion_kernel="vendored original Kernel(61,.5), NumPy seed0",padding="zero for convolutions",
            inpainting="Bernoulli .7 drop; shared RGB mask; noise on full output, frozen per image"),
        gt_usage="Synthetic y and final metrics only; optimize x0_gt=None; no tuning or selection",
        known_limits=["not author implementation parity","author image subset/metric conventions unmatched",
            "rounding/padding/db4 level3/clamp/mean-loss/clip/projection/last_opt conventions not all documented",
            "combined public-setting reset, NOT a one-factor value-domain ablation","no legacy T, no tuned lr/Karras"],
        experiment_card=card,git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    started=time.perf_counter()
    def save():record["elapsed_seconds"]=time.perf_counter()-started;save_record(directory,record)
    save()
    print(f"=== FFHQ four tasks / public settings / unit measurements ===\nRecords: {directory}",flush=True)
    print(f"Images={len(images)}; NEW method NFE={record['planned_method_nfe']}; zero extra UNet diagnostics/state.",flush=True)
    print("Paper columns are 1000-image MEANS, not same-image baselines; no automatic reproduction claim.",flush=True)
    try:
        import torch
        from modules.main_single import load_diffusion_model,load_image_as_tensor
        from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS
        from utils.diag_zaps_sampler_parity import rng_state,same_rng
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.diag_learning_rate_ablation import set_seed
        from utils.metrics import compute_all_metrics
        from modules.dataset_loader import tensor_to_image
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        kernel=original_motion_kernel(0);torch.save(torch.from_numpy(kernel),directory/"motion_kernel.pt")
        record["motion_generator_sha256"]=digest(ROOT/"DPS/motionblur/motionblur.py")
        record["motion_kernel_sha256"]=digest(directory/"motion_kernel.pt")
        record["source_sha256"]={name:digest(PROJECTS/name) for name in (
            "modules/zaps_algorithm.py","modules/degradations.py","modules/diffusion_model.py","modules/wavelet.py",
            "modules/main_single.py","modules/dataset_loader.py","utils/metrics.py","configs/config.py",
            "utils/diag_state_domain_paired.py","utils/diag_imagenet_gaussian_unit_domain.py",
            "utils/diag_imagenet_gaussian_unit_uniform.py","utils/diag_imagenet_tuned_state.py",
            "utils/diag_zaps_optimized_trace.py","utils/diag_zaps_sampler_parity.py",
            "utils/diag_zaps_first_direction.py","utils/diag_learning_rate_ablation.py",
            "utils/diag_ffhq_four_tasks_unit.py")}
        record["runtime"]=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=sys.version,
            device=args.device,gpu=torch.cuda.get_device_name(torch.device(args.device)),
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        model=None
        for image_index,item in enumerate(record["input_manifest"]):
            image_dir=directory/f"image_{image_index:03d}";image_dir.mkdir()
            gt=load_image_as_tensor(item["path"]).to(args.device);seed=item["seed"]
            if gt.shape!=(1,3,256,256) or not torch.isfinite(gt).all():raise RuntimeError("Invalid input image shape/value")
            tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(image_dir/"gt.png")
            set_seed(seed);initial=torch.randn_like(gt)
            for _ in range(290):torch.randn_like(initial)
            expected_end=rng_state(args.device)
            torch.save(initial.cpu(),image_dir/"initial_noise.pt")
            prepared={}
            # All four operator gates before the first score call for this image.
            for task in TASKS:
                task_dir=image_dir/task;task_dir.mkdir()
                base=make_base(task,args.device,gt,item["mask_seed"],kernel);operator=unit_operator(base).to(args.device)
                before=rng_state(args.device);gate=operator_gate(operator,gt,initial)
                gate["rng_unchanged"]=same_rng(before,rng_state(args.device));gate["passed"] &= gate["rng_unchanged"]
                key=f"{image_index:03d}_{task}";record["gates"][key]=gate;save()
                if not gate["passed"]:raise RuntimeError(f"{task} affine/adjoint gate failed BEFORE score calls")
                with torch.no_grad():
                    clean=operator.H(gt)
                    generator=torch.Generator(device=args.device);generator.manual_seed(item["measurement_seed"])
                    noise=.05*torch.randn(clean.shape,device=args.device,dtype=clean.dtype,generator=generator)
                    y=clean+noise
                torch.save(y.cpu(),task_dir/"measurement.pt");torch.save(noise.cpu(),task_dir/"measurement_noise.pt")
                if task=="inpainting":torch.save(base._mask_cache.cpu(),task_dir/"mask.pt")
                prepared[task]=(operator,y,task_dir,key)
                record.setdefault("observations",{})[key]=dict(measurement_sha256=digest(task_dir/"measurement.pt"),
                    noise_sha256=digest(task_dir/"measurement_noise.pt"),numeric_noise_std=float(noise.double().std(unbiased=False)),
                    mask_drop_fraction=float(1-base._mask_cache.mean()) if task=="inpainting" else None,
                    mask_sha256=digest(task_dir/"mask.pt") if task=="inpainting" else None)
            if model is None:
                print("All operator gates passed; loading/hash-checking FFHQ weights...",flush=True)
                model=load_diffusion_model("ffhq",args.device);model.model.eval()
                if model.num_steps!=1000 or any(p.dtype!=torch.float32 for p in model.model.parameters()):raise RuntimeError("Need fp32 FFHQ 1000-step model")
                record["checkpoint"]=dict(path=str(model.ckpt_path),sha256=digest(model.ckpt_path))
            for task in TASKS:
                operator,y,task_dir,key=prepared[task]
                z=LastUnrollSnapshotZAPS(model,operator,img_size=256,**config_for(task))
                manifest=json.loads((PROJECTS/"IMAGENET_BASELINE_MANIFEST_20261013.json").read_text(encoding="utf-8"))
                if z.tau.cpu().tolist()!=manifest["archived_reference"]["actual_timesteps_ascending"]:raise RuntimeError("Original paper grid changed")
                if not torch.equal(z.zeta,torch.full_like(z.zeta,config_for(task)["zeta_init"])) or not torch.equal(z.D,torch.full_like(z.D,.2)):
                    raise RuntimeError("Original uniform initialization changed")
                visited=[];gradients=[]
                def before(module,inputs):
                    record["method_nfe"]+=1
                    if record["method_nfe"]>record["planned_method_nfe"] or len(visited)>=300:raise RuntimeError("NFE budget exceeded")
                    visited.append(int(inputs[1].flatten()[0]))
                original_clip=torch.nn.utils.clip_grad_norm_
                def clip(parameters,max_norm,**kwargs):
                    parameters=list(parameters)
                    if len(parameters)!=2 or parameters[0] is not z.zeta or parameters[1] is not z.D or max_norm!=1:
                        raise RuntimeError("Original joint optimizer/clip contract changed")
                    norm=original_clip(parameters,max_norm,**kwargs)
                    if not torch.isfinite(norm):raise RuntimeError("Non-finite gradient")
                    gradients.append(float(norm));return norm
                hook=model.model.register_forward_pre_hook(before);set_seed(seed);arm_started=time.perf_counter()
                print(f"\n--- image {image_index+1}/{len(images)} / {task}: paper30x10, lr=.001, zeta={config_for(task)['zeta_init']} ---",flush=True)
                try:
                    with model_rng_guard(model.model,args.device) as guarded,patch.object(torch.nn.utils,"clip_grad_norm_",side_effect=clip):
                        losses=z.optimize(y,verbose=True,x0_gt=None)
                finally:hook.remove()
                if (z._last_nfe!=300 or len(visited)!=300 or len(guarded)!=300 or len(losses)!=10 or len(gradients)!=10
                        or visited!=z.tau.flip(0).tolist()*10 or not torch.equal(z._last_opt_noise.cpu(),initial.cpu())
                        or not same_rng(expected_end,rng_state(args.device)) or any(not math.isfinite(v) for v in losses)):
                    raise RuntimeError("Training NFE/grid/RNG/finite gate failed")
                if any(p.grad is not None for p in model.model.parameters()):raise RuntimeError("Unexpected score weight gradients")
                output=z._last_opt_x0.detach().cpu()
                if not torch.isfinite(output).all():raise RuntimeError("Non-finite output")
                torch.save(output,task_dir/"last_opt.pt");torch.save(z.audit_state,task_dir/"pre_last_state.pt")
                torch.save(dict(zeta=z.zeta.detach().cpu(),D=z.D.detach().cpu()),task_dir/"post_params.pt")
                tensor_to_image(output.squeeze(0),denormalize=True).save(task_dir/"reconstruction.png")
                metrics=compute_all_metrics(output,gt.cpu(),lpips_net="alex")
                mse=float(((output.clamp(-1,1)-gt.cpu()).double()/2).square().mean())
                float_psnr=-10*math.log10(max(mse,1e-30))
                with torch.no_grad():
                    final_residual=float((operator.H(output.to(args.device))-y).double().norm())
                row=dict(image=item["path"],seed=seed,task=task,**metrics,float_psnr=float_psnr,method_nfe=300,
                    unit_measurement_mse=losses[-1],unit_residual_norm=final_residual,
                    output_rms=float(output.double().square().mean().sqrt()),
                    output_out_of_range_fraction=float((output.abs()>1).double().mean()),
                    seconds=time.perf_counter()-arm_started)
                record["results"].append(row)
                record["arms"][key]=dict(config=config_for(task),grid=z.tau.cpu().tolist(),loss_history=losses,
                    gradients_before_clip=gradients,training_gate_passed=True,
                    output_sha256=digest(task_dir/"last_opt.pt"),pre_snapshot_sha256=digest(task_dir/"pre_last_state.pt"),
                    post_params_note="POST parameters did not produce last_opt; use PRE for replay")
                save();print(f"Finished {task}: PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}",flush=True)
                del z,output
        if record["method_nfe"]!=record["planned_method_nfe"]:raise RuntimeError("Total budget mismatch")
        record["summary"]=summarize(record["results"],len(images))
        record["decision"]=dict(classification="SINGLE_IMAGE_CONTEXT_ONLY" if len(images)==1 else "SMALL_SUBSET_NOT_BENCHMARK",
            author_parity=False,benchmark_reproduced=False,state_gain=False,
            limits="Different image subset from paper; contextual differences, NOT paired benchmark improvements",
            next_action="Review all four tasks together; discuss any expansion or state experiments, no automatic tuning")
        record["status"]="complete";save()
    except Exception as error:record["status"],record["error"]="failed",repr(error);save();raise
    print("\n=== FFHQ public settings / UNIT / four-task comparison ===",flush=True)
    print(f"images={len(images)}; PAPER means use 1000 images; deltas are contextual, not matched-sample gains.",flush=True)
    for r in record["summary"]:
        print(f"{r['task']:>20} PSNR={r['psnr_mean']:.4f} paper={r['paper_psnr']:.2f} d={r['delta_psnr_vs_paper_mean']:+.4f} "
              f"SSIM={r['ssim_mean']:.4f}/{r['paper_ssim']:.3f} LPIPS={r['lpips_mean']:.4f}/{r['paper_lpips']:.3f}",flush=True)
    print("Decision:",json.dumps(record["decision"],ensure_ascii=False),flush=True)
    print(f"New method NFE={record['method_nfe']}; diagnostic UNet NFE=0; records: {directory}",flush=True)


if __name__=="__main__":main()
