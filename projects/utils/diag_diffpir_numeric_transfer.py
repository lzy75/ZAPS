"""A79R1 DiffPIR native-loop scalar-module transfer with independently paired observations."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files
from utils import diffpir_numeric_transfer as rules
from utils import diffpir_numeric_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import gpu_inventory
ROOT='/data0/lzy/ZAPS-results/diag_diffpir_multiplier_transfer/A79R1_20261103'
PARENT=parent.ROOT
BRANCH='results/diffpir-numeric-transfer-r1-20261103'
CARD=PROJECTS/'experiment_cards/A79R1_diffpir_numeric_transfer_20261103.md'
MEASURED_PEAK=0


def resource(uuid,own=0):
    row=next(v for v in gpu_inventory() if v['uuid']==uuid)
    required=max(rules.MIN_FREE,MEASURED_PEAK+files.GIB)
    return dict(**row,own_reserved_bytes=own,required_bytes=required,admitted=row['free_bytes']+own>=required)


def check(root):
    root=Path(root);p=files.read(root/'plan.json')
    files.require(p['experiment']=='A79R1_diffpir_numeric_transfer','Wrong A79R1 plan')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json')==p['parent_plan_sha256'],'A71 plan changed')
    rules.verify_upstream();return p


def prepare(args):
    import numpy as np
    root=Path(args.root).resolve();source=Path(args.a71_root).resolve()
    if (root/'plan.json').exists():check(root);print('A79R1 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=parent.check(source);rules.verify_upstream()
    from utils import diag_diffpir_multiplier_transfer as failed_driver
    failed_source=Path(args.failed_a79_root).resolve();old_failed=failed_driver.check(failed_source)
    previous_summary=files.read(failed_source/'summary.json')
    files.require(not any((failed_source/'cases').glob('*/*/run.json')),'Original A79 has started model cases; inspect before new protocol')
    files.require(not any((failed_source/'workers').glob('*/load_*.json')),'Original A79 has model-load attempts; inspect cumulative cost')
    files.require(all(v==0 for v in previous_summary['actual_new_cost'].values()),'Original A79 has nonzero model costs')
    folder=Path(args.ffhq_dir) if args.ffhq_dir else Path(old['cases'][0]['image']).parent
    cases=rules.cases(dict(ffhq=folder,imagenet=args.imagenet_dir));inputs={};checkpoints={}
    for c in cases:
        files.require(Path(c['image']).is_file(),'Existing image missing: '+c['image'])
        inputs[c['image']]=files.digest(c['image'])
    for dataset,(name,h) in rules.CHECKPOINTS.items():
        path=PROJECTS/'modules/models'/name
        files.require(path.is_file() and files.digest(path)==h,'Missing/different existing checkpoint: '+str(path))
        checkpoints[dataset]=dict(path=str(path),sha256=h)
    motion=source/'motion_kernel.npy';files.require(files.digest(motion)==old['motion_kernel_sha256'],'Motion kernel changed')
    inputs[str(motion)]=files.digest(motion)
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB free')
    root.mkdir(exist_ok=True)
    g=np.arange(61,dtype=np.float64)-30;xx,yy=np.meshgrid(g,g);gaussian=np.exp(-(xx*xx+yy*yy)/(2*3.**2));gaussian=(gaussian/gaussian.sum()).astype(np.float32)
    np.save(root/'gaussian_kernel.npy',gaussian,allow_pickle=False);shutil.copyfile(motion,root/'motion_kernel.npy')
    kernels={t:dict(path=str(root/('gaussian_kernel.npy' if t=='gaussian_deblur' else 'motion_kernel.npy')),
        sha256=files.digest(root/('gaussian_kernel.npy' if t=='gaussian_deblur' else 'motion_kernel.npy'))) for t in rules.TASKS}
    names=['utils/diffpir_numeric_transfer.py','utils/diffpir_numeric_runtime.py','utils/diag_diffpir_numeric_transfer.py',
        'utils/a79r1_session.py','utils/test_diffpir_numeric_transfer.py','utils/test_diffpir_numeric_runtime.py',
        'utils/diffpir_multiplier_transfer.py','utils/diffpir_multiplier_runtime.py','utils/diag_diffpir_multiplier_transfer.py',
        'utils/test_diffpir_multiplier_runtime.py','utils/a67_model_freeze.py','utils/noise_signal_shrink.py','utils/paper_clock_runtime.py','utils/historical_ffhq_runtime.py',
        'utils/diag_noise_feasible_zaps.py','utils/metrics.py','utils/diag_learning_rate_ablation.py','modules/main_single.py',
        'modules/diffusion_model.py','modules/dataset_loader.py','run_a79r1_20261103.sh',str(CARD.relative_to(PROJECTS))]
    names += [str(f.relative_to(PROJECTS)) for f in rules.VENDOR.rglob('*') if f.is_file() and '__pycache__' not in str(f)]
    for row in kernels.values():inputs[row['path']]=row['sha256']
    inputs[str(failed_source/'plan.json')]=files.digest(failed_source/'plan.json')
    inputs[str(failed_source/'summary.json')]=files.digest(failed_source/'summary.json')
    for f in (failed_source/'metrology').glob('*.json'):inputs[str(f)]=files.digest(f)
    for c in cases:
        meta=failed_source/'observations'/c['key']/'observation.json'
        if meta.exists():
            row=files.read(meta);files.require(row['plan_sha256']==files.digest(failed_source/'plan.json') and files.digest(row['path'])==row['sha256'],'Old observation identity changed')
            inputs[str(meta)]=files.digest(meta);inputs[row['path']]=row['sha256']
    p=dict(experiment='A79R1_diffpir_numeric_transfer',parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases,checkpoints=checkpoints,kernels=kernels,inputs=inputs,sources={n:files.digest(PROJECTS/n) for n in names},
        arms=list(rules.ARMS),sequence=rules.sequence(),actual_grid=[999-v for v in rules.sequence()],
        upstream=rules.verify_upstream(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        failed_source=str(failed_source),failed_source_plan_sha256=files.digest(failed_source/'plan.json'),source_A79_recorded_cost=previous_summary['actual_new_cost'],
        budget=dict(runs=48,method_forwards=4800,input_vjp=0,optimizer_backwards=0,probe_per_model_load=1),
        protocol=dict(scope='FFHQ and ImageNet existing3 images each x Gaussian61/sigma3 and frozenMotion61 x four factorial arms.',
            numeric_factor='Four arms: original float32/original+module; stable equivalent float32/stable+module. Native residual failures remain failures at old1e-3 criterion; not relabeled solver-pass.',
            metrology='Before NN: native32 diagnostic; original64 residual<=1e-8, stable32 residual<=1e-5 and stable32-vs-original64 relative solution<=1e-3. Gate proves stable solve, not native32 accuracy.',
            baseline='Pinned DiffPIR main_ddpir.py sampling For AST executed unmodified. Native100 quadratic sequence, eta0, U1, learned_range, clip_denoisedTrue.',
            parameters='Paper supplement Table3 NFE100/sigma_y=.05: FFHQ Gaussian(lambda7,zeta.3),Motion(7,.4);ImageNet Gaussian(8,.3),Motion(8,.7). No quality-based selection.',
            noise='Physical[0,1] sigma=.05, equivalent model-coordinate sigma=.10. Native circular FFT convolution; NEW shared y per arm, not old A75/A77 zero-boundary y.',
            insertion='q_clean=z_prox-m; d=H(q_clean), r=y-H(m) in[0,1]. Same noise_signal_shrink.select scales q_clean only. No 1/sqrt(alpha) factor because correction already in clean coordinates.',
            preserved='Original rho_t/lambda spectral direction and zeta noise injection retained. Like prior projection, positive global guidance amplitude is canceled by gamma. No optimizer/input derivative or next-denoiser probe.',
            output='Original x_0 derived from x is returned; the native last unused pred_xstart evaluation is retained and counted. No final-output replacement, clamp, extra sampling or GT selection.',
            timing='Complete100 loop: y/operator/init/FFT/NN/sampling/CPU output sync and progress bookkeeping included; model load/probe/measurement generation/metrics/output writes separate.',
            limits='Small reused development panel, local paper-parameter native-loop reference with shared fixed kernels, not author benchmark reproduction. Both model/dataset changes paired; no universal improvement or state-necessity claim.'))
    files.write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A79R1 PREPARE/FREEZE:48 runs,4800F,0 inputVJP/optimizerB; native DiffPIR clean-prox slot; probes separately counted.',flush=True)


def kernel(p,c,device):
    import numpy as np
    import torch
    row=p['kernels'][c['task']];files.require(files.digest(row['path'])==row['sha256'],'Kernel changed')
    return torch.from_numpy(np.load(row['path'],allow_pickle=False).copy()).to(device=device,dtype=torch.float32)[None,None]


def observation(root,p,c,device):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.diag_learning_rate_ablation import set_seed
    folder=root/'observations'/c['key'];meta=folder/'observation.json'
    if meta.exists():
        r=files.read(meta);files.require(r['plan_sha256']==files.digest(root/'plan.json') and files.digest(r['path'])==r['sha256'],'Observation changed');return r
    files.require(not folder.exists() or not any(folder.iterdir()),'Partial observation retained')
    old_meta=Path(p['failed_source'])/'observations'/c['key']/'observation.json'
    if old_meta.exists():
        old=files.read(old_meta);files.require(files.digest(old['path'])==old['sha256'],'Old given y changed')
        folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(old['path'],folder/'y.pt')
        for name in ('ground_truth.png','observation.png'):
            if (old_meta.parent/name).exists():shutil.copyfile(old_meta.parent/name,folder/name)
        result=dict(old,path=str(folder/'y.pt'),plan_sha256=files.digest(root/'plan.json'),source_A79_y=old['path'])
        files.write(meta,result);return result
    folder.mkdir(parents=True,exist_ok=True)
    with precision.precision('LEGACY_IEEE'),torch.no_grad():
        gt=load_image_as_tensor(c['image']).to(device);physical=(gt+1)/2
        op=runtime.CircularBlur(kernel(p,c,device),physical.shape[-2:]);mean=op.H(physical)
        set_seed(c['measurement_seed']);y=mean+c['sigma_physical']*torch.randn_like(mean)
        torch.save(dict(y=y.cpu()),folder/'y.pt')
        tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(folder/'ground_truth.png')
        tensor_to_image((2*y.cpu()-1).squeeze(0),denormalize=True).save(folder/'observation.png')
    r=dict(path=str(folder/'y.pt'),sha256=files.digest(folder/'y.pt'),plan_sha256=files.digest(root/'plan.json'),
           sigma_physical=c['sigma_physical'],sigma_model=c['sigma_model'],boundary='circular')
    files.write(meta,r);return r


def metrology(root,p,c,device):
    import torch
    f=root/'metrology'/(c['key']+'.json')
    if f.exists():files.require(files.read(f)['stable_admission_passed'],'Previous numeric audit failed');return
    obs=observation(root,p,c,device);y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(device)
    k=kernel(p,c,device);_,sr,_,_=runtime.reference()
    op32=runtime.CircularBlur(k,y.shape[-2:]);op64=runtime.CircularBlur(k.double(),y.shape[-2:])
    pre32=sr.pre_calculate(y,k,1);pre64=sr.pre_calculate(y.double(),k.double(),1)
    sigma=runtime.schedule(device)['reduced_alpha_cumprod'];tests=[]
    with torch.no_grad():
        for t in (999,500,5):
            rho=(c['lambda_']*c['sigma_physical']**2/sigma[t].square()).float().to(device).reshape(1,1,1,1)
            m=torch.full_like(y,.5)
            native=sr.data_solution(m,*pre32,rho,1)
            stable=runtime.stable_solution(m,*pre32,rho,1)
            reference=sr.data_solution(m.double(),*pre64,rho.double(),1)
            rhs64=op64.transpose(y.double())+rho.double()*m.double()
            def residual(z):return float((op64.transpose(op64.H(z.double()))+rho.double()*z.double()-rhs64).norm()/rhs64.norm().clamp(min=1e-30))
            native_rhs32=op32.transpose(y)+rho*m
            native_old=float((op32.transpose(op32.H(native))+rho*native-native_rhs32).double().norm()/native_rhs32.double().norm().clamp(min=1e-30))
            relative=float((stable.double()-reference).norm()/reference.norm().clamp(min=1e-30))
            tests.append(dict(t=t,rho=float(rho),native_original_criterion_residual=native_old,native_meets_original_1e3=native_old<=1e-3,
                              native_true64_residual=residual(native),original64_residual=residual(reference),stable32_residual=residual(stable),stable32_vs_original64_relative_solution=relative))
    passed=all(v['original64_residual']<=1e-8 and v['stable32_residual']<=1e-5 and v['stable32_vs_original64_relative_solution']<=1e-3 for v in tests)
    files.write(f,dict(stable_admission_passed=passed,native_accuracy_certified=False,tests=tests,NN=0,
                      explanation='Native32 retained solely as byte-pinned official arithmetic reference; accuracy failure is not waived or reclassified. Stable32 is separately qualified against original64.'))
    files.require(passed,'Independent stable/reference numeric audit failed BEFORE model load: '+str(f))



def completed(root,c,arm):
    f=root/'cases'/c['key']/arm/'run.json'
    if not f.exists():return None
    r=files.read(f);files.require(r['status']=='complete','Started/failed attempt retained: '+str(f))
    files.require(r['plan_sha256']==files.digest(root/'plan.json'),'Attempt plan changed')
    for n,h in r['artifacts'].items():files.require(files.digest(f.parent/n)==h,'Output changed')
    return r


def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if completed(root,c,arm):return True
    if not resource(uuid,torch.cuda.memory_reserved())['admitted']:return False
    obs=observation(root,p,c,model.device);folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),counts=None,artifacts={});files.write(folder/'run.json',r)
    def progress(step,count):
        r['counts']=dict(count);files.write(folder/'run.json',r)
        print('A79R1 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/100 clean_prox=',step,flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count,patch.object(torch.autograd,'grad',side_effect=RuntimeError('A79R1 forbids input/parameter derivatives')):
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                return runtime.run(model,y,kernel(p,c,model.device),c,arm,count,progress=progress)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        files.require(count['forwards']==count['forward_attempts']==100 and count['calls']==p['actual_grid'],'Native grid/count changed')
        files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Unexpected derivative cost')
        gt=load_image_as_tensor(c['image']).cpu();metrics=common.metrics(out,gt)
        with torch.no_grad():
            y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
            op=runtime.CircularBlur(kernel(p,c,model.device),out.shape[-2:]);error=(y-op.H((out.to(model.device)+1)/2)).double()
            metrics['measurement_mse_physical']=float(error.square().mean())
        torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png');files.write(folder/'trace.json',detail['trace'])
        r.update(status='complete',counts=count,metrics=metrics,full_seconds=seconds,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 native_output=detail['native_output'],cost_is_lower_bound=False,artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json')})
        files.write(folder/'run.json',r);print('A79R1 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'seconds=',round(seconds,3),flush=True);return True
    except BaseException as e:
        r.update(status='failed',counts=locals().get('count',r['counts']),error=repr(e),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        files.write(folder/'run.json',r);raise


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a54_session import exclusive
    global MEASURED_PEAK
    root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
    cases=[c for c in p['cases'] if c['shard']==args.shard]
    with exclusive(folder/'worker.lock'):
        try:
            if not resource(args.device_uuid)['admitted']:files.write(folder/'status.json',dict(status='resource_limited'));return
            for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
            dataset=cases[0]['dataset'];checkpoint=p['checkpoints'][dataset]
            files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Checkpoint changed')
            for c in cases:metrology(root,p,c,'cuda:0')
            started=time.perf_counter();model=load_diffusion_model(dataset,'cuda:0');model.model.eval();torch.cuda.synchronize()
            load=dict(status='running',dataset=dataset,load_seconds=time.perf_counter()-started,probe_counts={})
            f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';files.write(f,load)
            with frozen_network(model.model):
                torch.cuda.reset_peak_memory_stats();gd,_,_,_=runtime.reference()
                diffusion=gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('linear',1000),model_mean_type=gd.ModelMeanType.EPSILON,model_var_type=gd.ModelVarType.LEARNED_RANGE,loss_type=gd.LossType.MSE,rescale_timesteps=False)
                with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as probe,torch.no_grad():
                    load['probe_counts']=probe;files.write(f,load)
                    result=diffusion.p_sample(model.model,torch.zeros(1,3,256,256,device=model.device),torch.tensor([999],device=model.device),clip_denoised=True)
                    files.require(bool(torch.isfinite(result['pred_xstart']).all()),'Real forward probe nonfinite')
                MEASURED_PEAK=torch.cuda.max_memory_reserved();load.update(probe_counts=probe,probe_peak_reserved_bytes=MEASURED_PEAK);files.write(f,load)
                del result;torch.cuda.empty_cache()
                for i,c in enumerate(cases):
                    order=rules.ARMS if (i//len(rules.TASKS))%2==0 else rules.ARMS[::-1]
                    for arm in order:
                        if not measure(root,p,c,arm,model,args.device_uuid):
                            load['status']='resource_limited';files.write(f,load);files.write(folder/'status.json',dict(status='resource_limited'));return
            load['status']='complete';files.write(f,load);files.write(folder/'status.json',dict(status='complete'))
        except BaseException as e:
            if 'load' in locals():load.update(status='failed',error=repr(e));files.write(f,load)
            files.write(folder/'status.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise


def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done={(r['case']['key'],r['arm']):r for r in rows if r['status']=='complete'}
    groups=[];comparisons=[];table=[]
    for dataset in rules.DATASETS:
        for task in rules.TASKS:
            cases=[c for c in p['cases'] if c['dataset']==dataset and c['task']==task]
            for arm in rules.ARMS:
                data=[done[c['key'],arm] for c in cases if (c['key'],arm) in done]
                for r in data:table.append(dict(dataset=dataset,task=task,image_id=r['case']['image_id'],arm=arm,**r['metrics'],full_seconds=r['full_seconds'],NFE=100,input_VJP=0))
                if len(data)==3:groups.append(dict(dataset=dataset,task=task,arm=arm,cases=3,quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),NFE=100,input_VJP=0))
            for base_arm,new_arm in rules.PAIRS+(('DIFFPIR','STABLE_DIFFPIR'),('SHRINK','STABLE_SHRINK')):
                if not all((c['key'],a) in done for c in cases for a in (base_arm,new_arm)):continue
                base=[done[c['key'],base_arm] for c in cases];new=[done[c['key'],new_arm] for c in cases]
                comparisons.append(dict(dataset=dataset,task=task,baseline=base_arm,candidate=new_arm,
                    candidate_minus_baseline={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')},
                    images_PSNR_positive=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new)),
                    worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),
                    full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base)))
    counts=[r.get('counts') or {} for r in rows];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_DIFFPIR_NUMERIC_FACTORIAL',completed_runs=len(done),planned_runs=48,image_clusters=6,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(v.get('forward_attempts',0) for v in counts),input_vjp_attempts=sum(v.get('input_vjp_attempts',0) for v in counts),optimizer_backward_attempts=sum(v.get('optimizer_backward_attempts',0) for v in counts),native_data_solve_attempts=sum(v.get('native_data_solve_attempts',0) for v in counts),module_H_actions=sum(v.get('base_H_actions',0) for v in counts),probe_forward_attempts=sum(v.get('probe_counts',{}).get('forward_attempts',0) for v in loads)),
        source_A79_recorded_cost=p['source_A79_recorded_cost'],cost_is_lower_bound=any(r['status']!='complete' for r in rows) or any(v['status']=='failed' for v in loads),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# A79R1 DiffPIR 原版与噪声感知倍率','', '新圆周卷积观测，[0,1]噪声σ=.05，原文100NFE参数；不能与旧DPS观测平均值直接相减。','', '|数据集|任务|方法|PSNR|SSIM|LPIPS|完整秒数|','|---|---|---|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['dataset']}|{r['task']}|{r['arm']}|{q['psnr']:.3f}|{q['ssim']:.4f}|{q['lpips']:.4f}|{r['mean_full_seconds']:.3f}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n');print('A79R1 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH;bundle,commit=exporter.export_and_commit({'A79R1':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A79R1 native DiffPIR transfer uploaded.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a71-root',default=PARENT)
    p.add_argument('--failed-a79-root',default=rules.FAILED_ROOT)
    p.add_argument('--ffhq-dir');p.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256');p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device-uuid');p.add_argument('--check-worker-cli',action='store_true')
    group=p.add_mutually_exclusive_group(required=True)
    for n in ('prepare','worker','summarize','report','archive'):group.add_argument('--'+n,action='store_true')
    a=p.parse_args()
    if a.worker and (a.shard is None or not a.device_uuid):p.error('Worker needs explicit shard and GPU UUID')
    if a.check_worker_cli:
        files.require(a.worker,'Need worker');print(json.dumps(dict(shard=a.shard,device_uuid=a.device_uuid,model_calls=0)));return
    if a.prepare:prepare(a)
    elif a.worker:worker(a)
    elif a.summarize or a.report:summarize(a.root)
    else:archive(a.root)


if __name__=='__main__':main()
