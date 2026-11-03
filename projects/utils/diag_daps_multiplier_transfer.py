"""A81 native official DAPS FFHQ paired multiplier transfer, fixed paper-command setup."""
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
from utils import daps_multiplier_transfer as rules
from utils import daps_multiplier_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import gpu_inventory
ROOT='/data0/lzy/ZAPS-results/diag_daps_multiplier_transfer/A81_20261104'
PARENT=parent.ROOT
BRANCH='results/daps-multiplier-transfer-20261104'
CARD=PROJECTS/'experiment_cards/A81_daps_multiplier_transfer_20261104.md'
MEASURED_PEAK=0


def resource(uuid,own=0):
    row=next(v for v in gpu_inventory() if v['uuid']==uuid)
    required=max(rules.MIN_FREE,MEASURED_PEAK+files.GIB)
    return dict(**row,own_reserved_bytes=own,required_bytes=required,admitted=row['free_bytes']+own>=required)


def check(root):
    root=Path(root);p=files.read(root/'plan.json')
    files.require(p['experiment']=='A81_daps_multiplier_transfer','Wrong A81 plan')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json')==p['parent_plan_sha256'],'Parent image plan changed')
    rules.verify_upstream();return p


def prepare(args):
    root=Path(args.root).resolve();source=Path(args.a71_root).resolve()
    if (root/'plan.json').exists():check(root);print('A81 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=parent.check(source);summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==72,'Need complete A71 image panel')
    cases=rules.cases(old['cases']);inputs={}
    for c in cases:
        files.require(Path(c['image']).is_file(),'Existing image missing');inputs[c['image']]=files.digest(c['image'])
    path=PROJECTS/'modules/models'/rules.CHECKPOINT[0]
    files.require(path.is_file() and files.digest(path)==rules.CHECKPOINT[1],'Need the existing official FFHQ checkpoint, no replacement/download')
    checkpoint=dict(path=str(path),sha256=rules.CHECKPOINT[1]);inputs[str(path)]=checkpoint['sha256']
    names=['utils/daps_multiplier_transfer.py','utils/daps_multiplier_runtime.py','utils/diag_daps_multiplier_transfer.py',
           'utils/a81_session.py','utils/test_daps_multiplier_transfer.py','utils/test_daps_multiplier_runtime.py',
           'run_a81_20261104.sh',str(CARD.relative_to(PROJECTS)),*old['sources']]
    names += [str(f.relative_to(PROJECTS)) for f in rules.VENDOR.rglob('*') if f.is_file() and '__pycache__' not in str(f)]
    configs={t:rules.configuration(t) for t in rules.TASKS}
    for cfg in configs.values():
        s=cfg['sampler'];files.require(s['annealing_scheduler_config']['num_steps']==200 and s['diffusion_scheduler_config']['num_steps']==5,'Official DAPS1k grid changed')
        m=s['mcmc_sampler_config'];files.require(m['num_steps']==100 and m['prior_solver']=='gaussian' and m['mc_algo']=='langevin' and m['tau']==.01,'Official MCMC configuration changed')
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB free');root.mkdir(exist_ok=True)
    p=dict(experiment='A81_daps_multiplier_transfer',parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases,checkpoint=checkpoint,configs=configs,model_config=rules.model_config(path),inputs=inputs,
        sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)},arms=list(rules.ARMS),upstream=rules.verify_upstream(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        budget=dict(runs=12,method_forwards=12000,operator_gradient_attempts=240000,input_vjp=0,optimizer_backwards=0,extra_module_H_actions=120000,probe_per_model_load=1),
        protocol=dict(scope='Three reused FFHQ images x SR4/Motion, original FFHQ pixel DDPM prior. Initial transfer stage only; no new datasets/checkpoints.',
            baseline='Official pinned DAPS/PFODE/MCMC/VPPrecond/DDPM/operator AST and unmodified Euler torchdiffeq0.2.5; config commands/pixel_ffhq.sh. Batch1 and fixed existing image panel, not author benchmark mean.',
            grids='200 annealing outer steps;5 Euler network calls/outer=1000F;100 Langevin updates/outer=20000 measurement gradients. Gaussian prior needs no additional NN.',
            parameters='Original pixel tau=.01 (not measurement sigma=.05); lr SR1e-4/Motion5e-5, lr_min_ratio=.01. Anneal sigma100/min.1, diffusion min.01, poly7; no tuning.',
            observations='NEW y from official operators on model[-1,1] sigma=.05; SR uses original antialiased Resizer, Motion61/intensity.5 reflection convolution and seeded original kernel constructor. Shared given y/operator state per pair; not old zero/circular y.',
            insertion='Only original operator gradient/data drift scaled. q=-lr*gradient(loss)/tau^2 in clean x coordinates, r=y-Hx, d=Hq. Same A71 select; no alpha conversion, extra NN, VJP or finite differences.',
            preserved='Original prior score, xt term, lr schedule, Langevin random term, annealing re-noising and returned xt retained. Positive global data-drift amplitude can be canceled by projection.',
            gradient_cost='Original measurement gradients counted separately; no UNet input derivative. Candidate reuses original measurement prediction and adds ONE Hq per MCMC update; RNG untouched.',
            failure='Native NaN zero-output fallback is rejected as a failed run, never accepted as reconstructed quality. No automatic failed-case rerun or policy fallback.',
            timing='Same-session complete1000 includes y/operator-state load, original operator construction, init/PFODE/MCMC/re-noising/CPU sync and progress. Model load/probe/measurement generation/metrics/writes separate.',
            limits='State-dependent likelihood drift changes the MCMC target; no inherited exact-posterior theorem. Small reused development panel; no universal improvement/state-necessity claim.',
            decision='Mean PSNR>=+.1dB, >=2/3 PSNR wins, SSIM nondecrease and LPIPS nonincrease per task, paired init/final RNG and NN-call fingerprints equal. No hyperparameter scan.'))
    files.write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A81 PREPARE/FREEZE:12 fresh runs;12000F +240000 native measurement gradients;0 UNet VJP/optimizerB; module adds120000 Hq actions; probes separately counted.',flush=True)


def observation(root,p,c,device):
    import torch
    from modules.dataset_loader import tensor_to_image
    from utils.diag_learning_rate_ablation import set_seed
    folder=root/'observations'/c['key'];meta=folder/'observation.json'
    if meta.exists():
        row=files.read(meta);files.require(row['plan_sha256']==files.digest(root/'plan.json'),'Observation plan changed')
        for name,h in row['artifacts'].items():files.require(files.digest(folder/name)==h,'Given observation/operator changed')
        return row
    files.require(not folder.exists() or not any(folder.iterdir()),'Partial given y retained');folder.mkdir(parents=True,exist_ok=True)
    with precision.precision('LEGACY_IEEE'),torch.no_grad():
        gt=runtime.load_image(c['image'],device);op=runtime.make_operator(c,device)
        set_seed(c['measurement_seed']);y=op.measure(gt)
        torch.save(dict(y=y.cpu()),folder/'y.pt');torch.save(runtime.operator_state(op),folder/'operator_state.pt')
        tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(folder/'ground_truth.png')
        tensor_to_image(y.cpu().squeeze(0),denormalize=True).save(folder/'observation.png')
    row=dict(path=str(folder/'y.pt'),operator_state=str(folder/'operator_state.pt'),plan_sha256=files.digest(root/'plan.json'),sigma_model=c['sigma'],
        artifacts={name:files.digest(folder/name) for name in ('y.pt','operator_state.pt','ground_truth.png','observation.png')})
    files.write(meta,row);return row


def fresh_operator(c,device,obs):
    import torch
    op=runtime.make_operator(c,device);expected=torch.load(obs['operator_state'],map_location='cpu',weights_only=True);actual=runtime.operator_state(op)
    files.require(set(actual)==set(expected) and all(torch.equal(actual[k],expected[k]) for k in expected),'Original operator state/RNG changed')
    return op


def completed(root,c,arm):
    f=root/'cases'/c['key']/arm/'run.json'
    if not f.exists():return None
    r=files.read(f);files.require(r['status']=='complete','Started/failed attempt retained: '+str(f))
    files.require(r['plan_sha256']==files.digest(root/'plan.json'),'Attempt plan changed')
    for n,h in r['artifacts'].items():files.require(files.digest(f.parent/n)==h,'Output changed')
    return r


def fingerprint(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def measure(root,p,c,arm,model,uuid):
    import numpy as np
    import torch
    from modules.dataset_loader import tensor_to_image
    if completed(root,c,arm):return True
    if not resource(uuid,torch.cuda.memory_reserved())['admitted']:return False
    obs=observation(root,p,c,model.device);folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),counts=None,artifacts={});files.write(folder/'run.json',r)
    def progress(step,count):
        r['counts']=dict(count);files.write(folder/'run.json',r)
        print('A81 PROGRESS:',c['key'],arm,'outer=',step,'/200 F=',count['forwards'],'/1000 measurement_grad=',count['operator_gradient_attempts'],'/20000',flush=True)
    raw=model.model.model
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        versions=[v._version for v in raw.parameters()]
        with precision.precision('LEGACY_IEEE'),runtime.counters(raw) as count:
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=fresh_operator(c,model.device,obs)
                return runtime.run(model,op,y,c,arm,count,sampler_config=p['configs'][c['task']]['sampler'],progress=progress)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        files.require(count['forwards']==count['forward_attempts']==1000,'Official DAPS1k NFE changed')
        files.require(count['operator_gradient_attempts']==count['operator_gradients_completed']==count['mcmc_updates']==20000 and count['annealing_steps']==200,'Incomplete native MCMC/annealing budget')
        files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Unexpected network/optimizer derivative')
        files.require(count['module_H_actions']==(20000 if arm=='SHRINK' else 0),'Unexpected module response cost')
        files.require(all(v.grad is None and v._version==version and not v.requires_grad for v,version in zip(raw.parameters(),versions)),'Denoiser weights/flags changed')
        gt=runtime.load_image(c['image'],'cpu');metrics=common.metrics(out,gt)
        with torch.no_grad():
            y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
            error=(y-fresh_operator(c,model.device,obs)(out.to(model.device))).double()
            metrics['measurement_mse_model']=float(error.square().mean())
        torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace']);np.save(folder/'dense_trace.npy',detail['dense_trace'],allow_pickle=False)
        files.write(folder/'dense_trace_columns.json',detail['dense_columns'])
        rng=dict(initial=fingerprint(detail['initial']),before_cpu=fingerprint(detail['cpu_rng']),before_cuda=[fingerprint(x) for x in detail['cuda_rng']],
            final_cpu=fingerprint(detail['final_cpu_rng']),final_cuda=[fingerprint(x) for x in detail['final_cuda_rng']],calls=count['calls'])
        r.update(status='complete',counts=count,metrics=metrics,full_seconds=seconds,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            native_output=detail['native_output'],paired_fingerprints=rng,cost_is_lower_bound=False,
            artifacts={name:files.digest(folder/name) for name in ('output.pt','output.png','trace.json','dense_trace.npy','dense_trace_columns.json')})
        files.write(folder/'run.json',r);print('A81 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'full_seconds=',round(seconds,3),flush=True);return True
    except BaseException as error:
        r.update(status='failed',counts=locals().get('count',r['counts']),error=repr(error),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        files.write(folder/'run.json',r);raise


def worker(args):
    import torch
    from utils.a54_session import exclusive
    global MEASURED_PEAK
    root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
    cases=[c for c in p['cases'] if c['shard']==args.shard]
    with exclusive(folder/'worker.lock'):
        try:
            if not resource(args.device_uuid)['admitted']:files.write(folder/'status.json',dict(status='resource_limited'));return
            for name,h in p['inputs'].items():files.require(files.digest(name)==h,'Input changed: '+name)
            for c in cases:observation(root,p,c,'cuda:0')
            started=time.perf_counter();model=runtime.load_model(p['checkpoint']['path'],'cuda:0');torch.cuda.synchronize()
            load=dict(status='running',load_seconds=time.perf_counter()-started,probe_counts={})
            f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';files.write(f,load)
            with frozen_network(model.model.model):
                torch.cuda.reset_peak_memory_stats()
                with precision.precision('LEGACY_IEEE'),runtime.counters(model.model.model) as probe,torch.no_grad():
                    load['probe_counts']=probe;files.write(f,load)
                    result=model.tweedie(torch.zeros(1,3,256,256,device=model.device),1.)
                    files.require(bool(torch.isfinite(result).all()) and tuple(result.shape)==(1,3,256,256),'Official model/preconditioner probe failed')
                MEASURED_PEAK=torch.cuda.max_memory_reserved();load.update(probe_counts=probe,probe_peak_reserved_bytes=MEASURED_PEAK);files.write(f,load)
                del result;torch.cuda.empty_cache()
                for i,c in enumerate(cases):
                    for arm in (rules.ARMS if i%2==0 else rules.ARMS[::-1]):
                        if not measure(root,p,c,arm,model,args.device_uuid):
                            load['status']='resource_limited';files.write(f,load);files.write(folder/'status.json',dict(status='resource_limited'));return
            load['status']='complete';files.write(f,load);files.write(folder/'status.json',dict(status='complete'))
        except BaseException as error:
            if 'load' in locals():load.update(status='failed',error=repr(error));files.write(f,load)
            files.write(folder/'status.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()));raise


def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done={(r['case']['key'],r['arm']):r for r in rows if r['status']=='complete'}
    groups=[];comparisons=[];table=[]
    for task in rules.TASKS:
        cases=[c for c in p['cases'] if c['task']==task]
        for arm in rules.ARMS:
            data=[done[c['key'],arm] for c in cases if (c['key'],arm) in done]
            for r in data:table.append(dict(task=task,image_id=r['case']['image_id'],arm=arm,**r['metrics'],full_seconds=r['full_seconds'],NFE=1000,operator_gradients=20000,input_VJP=0))
            if len(data)==3:groups.append(dict(task=task,arm=arm,cases=3,quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),NFE=1000,operator_gradients=20000,input_VJP=0))
        if all((c['key'],a) in done for c in cases for a in rules.ARMS):
            base=[done[c['key'],'DAPS'] for c in cases];new=[done[c['key'],'SHRINK'] for c in cases]
            delta={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')}
            paired=all(n['paired_fingerprints']==b['paired_fingerprints'] for b,n in zip(base,new))
            wins=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new))
            comparisons.append(dict(task=task,candidate='SHRINK',minus_DAPS=delta,images_PSNR_positive=wins,
                worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),paired_noise_and_grid_passed=paired,
                full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base),
                positive_development_candidate=paired and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
    counts=[r.get('counts') or {} for r in rows];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_OFFICIAL_DAPS_TRANSFER',completed_runs=len(done),planned_runs=12,image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(v.get('forward_attempts',0) for v in counts),input_vjp_attempts=sum(v.get('input_vjp_attempts',0) for v in counts),optimizer_backward_attempts=sum(v.get('optimizer_backward_attempts',0) for v in counts),
            operator_gradient_attempts=sum(v.get('operator_gradient_attempts',0) for v in counts),operator_gradients_completed=sum(v.get('operator_gradients_completed',0) for v in counts),base_H_actions=sum(v.get('base_H_actions',0) for v in counts),module_H_actions=sum(v.get('module_H_actions',0) for v in counts),probe_forward_attempts=sum(v.get('probe_counts',{}).get('forward_attempts',0) for v in loads)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows) or any(v['status'] not in ('complete','resource_limited') for v in loads),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# A81 官方像素DAPS倍率迁移','','两臂均完整1000F/20000观测梯度，同场clock。未改变随机项；候选不继承原后验保证。','','|任务|方法|PSNR|SSIM|LPIPS|完整秒数|','|---|---|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['arm']}|{q['psnr']:.3f}|{q['ssim']:.4f}|{q['lpips']:.4f}|{r['mean_full_seconds']:.3f}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n');print('A81 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH;bundle,commit=exporter.export_and_commit({'A81':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A81 official DAPS transfer uploaded.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a71-root',default=PARENT)
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device-uuid');p.add_argument('--check-worker-cli',action='store_true')
    group=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','worker','summarize','report','archive'):group.add_argument('--'+name,action='store_true')
    a=p.parse_args()
    if a.worker and (a.shard is None or not a.device_uuid):p.error('Worker needs explicit shard and GPU UUID')
    if a.check_worker_cli:
        files.require(a.worker,'Need worker');print(json.dumps(dict(shard=a.shard,device_uuid=a.device_uuid,model_calls=0)));return
    if a.prepare:prepare(a)
    elif a.worker:worker(a)
    elif a.summarize or a.report:summarize(a.root)
    else:archive(a.root)


if __name__=='__main__':main()
