"""A80 host-objective-aware DiffPIR multiplier test, same-y and same-session clocks."""
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
from utils import diffpir_host_aware_transfer as rules
from utils import diffpir_host_aware_runtime as runtime
from utils import diag_diffpir_numeric_transfer as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import gpu_inventory
ROOT='/data0/lzy/ZAPS-results/diag_diffpir_host_aware/A80_20261103'
PARENT=parent.ROOT
BRANCH='results/diffpir-host-aware-transfer-20261103'
CARD=PROJECTS/'experiment_cards/A80_diffpir_host_aware_20261103.md'
MEASURED_PEAK=0


def resource(uuid,own=0):
    row=next(v for v in gpu_inventory() if v['uuid']==uuid)
    required=max(rules.MIN_FREE,MEASURED_PEAK+files.GIB)
    return dict(**row,own_reserved_bytes=own,required_bytes=required,admitted=row['free_bytes']+own>=required)


def check(root):
    root=Path(root);p=files.read(root/'plan.json')
    files.require(p['experiment']=='A80_diffpir_host_aware','Wrong A80 plan')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json')==p['parent_plan_sha256'],'A79R1 plan changed')
    rules.verify_upstream();return p


def prepare(args):
    root=Path(args.root).resolve();source=Path(args.a79r1_root).resolve()
    if (root/'plan.json').exists():check(root);print('A80 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=parent.check(source);summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==48,'Need completed A79R1, no subset selected by outcomes')
    cases=rules.cases(old['cases']);inputs={str(source/'plan.json'):files.digest(source/'plan.json'),str(source/'summary.json'):files.digest(source/'summary.json')}
    for c in cases:
        files.require(Path(c['image']).is_file(),'Image missing');inputs[c['image']]=files.digest(c['image'])
        for arm in (rules.REFERENCE_SOURCE_ARM,rules.CACHED_SOURCE_ARM):
            files.require(parent.completed(source,c,arm) is not None,'Incomplete parent arm')
            for name in ('run.json','trace.json'):
                f=source/'cases'/c['key']/arm/name;inputs[str(f)]=files.digest(f)
        meta=source/'observations'/c['key']/'observation.json';row=files.read(meta)
        files.require(row['plan_sha256']==files.digest(source/'plan.json') and files.digest(row['path'])==row['sha256'],'Old y identity changed')
        inputs[str(meta)]=files.digest(meta);inputs[row['path']]=row['sha256']
        f=source/'metrology'/(c['key']+'.json')
        files.require(files.read(f)['stable_admission_passed'],'Old stable numerical gate missing/failed');inputs[str(f)]=files.digest(f)
    for row in old['checkpoints'].values():
        files.require(files.digest(row['path'])==row['sha256'],'Checkpoint changed');inputs[row['path']]=row['sha256']
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB free')
    root.mkdir(exist_ok=True);kernels={}
    for task,row in old['kernels'].items():
        files.require(files.digest(row['path'])==row['sha256'],'Old kernel changed')
        dest=root/Path(row['path']).name;shutil.copyfile(row['path'],dest)
        kernels[task]=dict(path=str(dest),sha256=row['sha256']);inputs[str(dest)]=row['sha256']
    names=['utils/host_aware_shrink.py','utils/diffpir_host_aware_transfer.py','utils/diffpir_host_aware_runtime.py',
           'utils/diag_diffpir_host_aware.py','utils/a80_session.py','utils/test_diffpir_host_aware.py',
           'utils/test_diffpir_host_aware_runtime.py','run_a80_20261103.sh',str(CARD.relative_to(PROJECTS))]
    sources=dict(old['sources']);sources.update({n:files.digest(PROJECTS/n) for n in names})
    protocol=dict(old['protocol'])
    protocol.update(scope='Same6 A79R1 images x2blur tasks; fresh paired STABLE DiffPIR and host-aware runs, cached old-policy quality.',
        numeric_factor='Stable equivalent sf1 solver fixed in both NEW arms; no new numerical-form scan.',
        noise='Exact given A79R1 y reused; physical[0,1] sigma=.05, model-coordinate sigma=.10; circular convolution.',
        adaptation='Same gamma objective plus known host quadratic kappa=rho*||q||^2. Gamma=omega*positive(r.d)/(||d||^2+kappa), no new fitted coefficient.',
        identity='For exact linear proximal solve, r.d=||d||^2+rho||q||^2 and gamma=omega. This is noise-aware damping, not a new denoiser-response or uncertainty estimator.',
        insertion='Clean-coordinate prox correction only; stable sf1 solution, native sampling AST/grid/zeta/lambda/RNG/output unchanged.',
        preserved='Native rho/lambda/zeta and proximal q retained; on exact proximal states gamma=omega. No new gradients, next-denoiser probes or coefficient fitting.',
        cost='24 NEW complete100 runs=2400F/0VJP/0optimizerB plus model probes; old A79R1 4800F recorded separately.',
        reuse='Same y/kernels/images/checkpoints and frozen sources; completed old STABLE_SHRINK quality only. Historical clock is labeled and excluded from fresh paired ratios.',
        reference_gate='Fresh baseline vs cached STABLE_DIFFPIR: relative output L2<=1e-3; PSNR<=.02dB, SSIM/LPIPS<=.001. Failure retains outputs and makes support inconclusive, no rerun.',
        timing='Same-session complete100 per arm, alternating order. Includes y/kernel/init/FFT/NN/CPU sync/progress. Excludes load/probe/metrics/parent comparisons/writes.',
        decision='Per task/dataset: mean PSNR>=+.1dB, >=2/3 PSNR wins, SSIM nondecrease and LPIPS nonincrease; all paired baseline repeat gates must pass. No parameter scan.')
    p=dict(experiment='A80_diffpir_host_aware',parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases,checkpoints=old['checkpoints'],kernels=kernels,inputs=inputs,sources=sources,arms=list(rules.ARMS),
        sequence=old['sequence'],actual_grid=old['actual_grid'],upstream=rules.verify_upstream(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        budget=dict(runs=24,method_forwards=2400,input_vjp=0,optimizer_backwards=0,probe_per_model_load=1),
        source_A79R1_recorded_cost=summary['actual_new_cost'],protocol=protocol)
    files.write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A80 PREPARE/FREEZE:24 new runs,2400F,0VJP/optimizerB; two priors, same given y; completed old policy reused for quality.',flush=True)


def kernel(p,c,device):
    return parent.kernel(p,c,device)


def observation(root,p,c,device):
    folder=root/'observations'/c['key'];meta=folder/'observation.json'
    if meta.exists():
        row=files.read(meta);files.require(row['plan_sha256']==files.digest(root/'plan.json') and files.digest(row['path'])==row['sha256'],'Observation changed');return row
    old_meta=Path(p['parent'])/'observations'/c['key']/'observation.json';old=files.read(old_meta)
    files.require(files.digest(old['path'])==old['sha256'],'Given y changed')
    files.require(not folder.exists() or not any(folder.iterdir()),'Partial observation retained')
    folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(old['path'],folder/'y.pt')
    for name in ('ground_truth.png','observation.png'):
        if (old_meta.parent/name).exists():shutil.copyfile(old_meta.parent/name,folder/name)
    row=dict(old,path=str(folder/'y.pt'),plan_sha256=files.digest(root/'plan.json'),source_A79R1_y=old['path'])
    files.write(meta,row);return row


def metrology(root,p,c,device):
    observation(root,p,c,device)
    old=Path(p['parent'])/'metrology'/(c['key']+'.json');row=files.read(old)
    files.require(row['stable_admission_passed'],'Parent stable gate failed')
    folder=root/'metrology';folder.mkdir(exist_ok=True)
    files.write(folder/old.name,dict(row,reused_A79R1_path=str(old),reused_sha256=files.digest(old),new_NN=0))


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
        print('A80 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/100 clean_prox=',step,flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count,patch.object(torch.autograd,'grad',side_effect=RuntimeError('A80 forbids input/parameter derivatives')):
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
        if arm=='DIFFPIR_REFERENCE':
            source_run=parent.completed(Path(p['parent']),c,rules.REFERENCE_SOURCE_ARM)
            reference=torch.load(Path(p['parent'])/'cases'/c['key']/rules.REFERENCE_SOURCE_ARM/'output.pt',map_location='cpu',weights_only=True)
            relative=float((out-reference).double().norm()/reference.double().norm().clamp(min=1e-30))
            deltas={k:metrics[k]-source_run['metrics'][k] for k in ('psnr','ssim','lpips')}
            passed=relative<=1e-3 and abs(deltas['psnr'])<=.02 and abs(deltas['ssim'])<=.001 and abs(deltas['lpips'])<=.001
            r['reference_repeat_gate']=dict(passed=passed,relative_output_l2=relative,quality_deltas=deltas)
        files.write(folder/'run.json',r);print('A80 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'seconds=',round(seconds,3),flush=True);return True
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
    root=Path(root);p=check(root)
    rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done={(r['case']['key'],r['arm']):r for r in rows if r['status']=='complete'}
    cached={c['key']:parent.completed(Path(p['parent']),c,rules.CACHED_SOURCE_ARM) for c in p['cases']}
    groups=[];comparisons=[];table=[];identities=[]
    for dataset in rules.DATASETS:
        for task in rules.TASKS:
            cases=[c for c in p['cases'] if c['dataset']==dataset and c['task']==task]
            for arm in (*rules.ARMS,'CACHED_OLD_PROJECTION'):
                data=([cached[c['key']] for c in cases] if arm=='CACHED_OLD_PROJECTION'
                      else [done[c['key'],arm] for c in cases if (c['key'],arm) in done])
                for r in data:table.append(dict(dataset=dataset,task=task,image_id=r['case']['image_id'],arm=arm,**r['metrics'],full_seconds=r['full_seconds'],clock_source='A79R1_ARCHIVED' if arm=='CACHED_OLD_PROJECTION' else 'A80_FRESH',NFE=100,input_VJP=0))
                if len(data)==3:
                    groups.append(dict(dataset=dataset,task=task,arm=arm,cases=3,quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),clock_source='A79R1_ARCHIVED' if arm=='CACHED_OLD_PROJECTION' else 'A80_FRESH',NFE=100,input_VJP=0))
            for base_arm in ('DIFFPIR_REFERENCE','CACHED_OLD_PROJECTION'):
                if not all((c['key'],'HOST_AWARE') in done and (base_arm=='CACHED_OLD_PROJECTION' or (c['key'],base_arm) in done) for c in cases):continue
                base=[cached[c['key']] if base_arm=='CACHED_OLD_PROJECTION' else done[c['key'],base_arm] for c in cases]
                new=[done[c['key'],'HOST_AWARE'] for c in cases]
                delta={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')}
                wins=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new))
                repeat=all((c['key'],'DIFFPIR_REFERENCE') in done and done[c['key'],'DIFFPIR_REFERENCE'].get('reference_repeat_gate',{}).get('passed',False) for c in cases)
                comparisons.append(dict(dataset=dataset,task=task,baseline=base_arm,candidate='HOST_AWARE',candidate_minus_baseline=delta,images_PSNR_positive=wins,
                    worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),reference_repeat_gate_passed=repeat,
                    positive_development_candidate=repeat and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0,
                    same_session_full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base) if base_arm=='DIFFPIR_REFERENCE' else None))
    for c in p['cases']:
        if (c['key'],'HOST_AWARE') not in done:continue
        trace=files.read(root/'cases'/c['key']/'HOST_AWARE'/'trace.json');resolved=[v for v in trace if v['resolved']]
        identities.append(dict(case=c['key'],resolved_steps=len(resolved),max_stationarity_relative_error=max((v['stationarity_relative_error'] for v in resolved),default=0),max_abs_gamma_minus_weight=max((abs(v['gamma_minus_noise_weight']) for v in resolved),default=0),gamma_min=min(v['gamma'] for v in trace),gamma_max=max(v['gamma'] for v in trace)))
    counts=[r.get('counts') or {} for r in rows];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_DIFFPIR_HOST_OBJECTIVE_ADAPTATION',completed_runs=len(done),planned_runs=24,image_clusters=6,groups=groups,comparisons=comparisons,proximal_identity_diagnostics=identities,
        actual_new_cost=dict(method_forward_attempts=sum(v.get('forward_attempts',0) for v in counts),input_vjp_attempts=sum(v.get('input_vjp_attempts',0) for v in counts),optimizer_backward_attempts=sum(v.get('optimizer_backward_attempts',0) for v in counts),native_data_solve_attempts=sum(v.get('native_data_solve_attempts',0) for v in counts),module_H_actions=sum(v.get('base_H_actions',0) for v in counts),module_q_norms=sum(v.get('module_q_norms',0) for v in counts),probe_forward_attempts=sum(v.get('probe_counts',{}).get('forward_attempts',0) for v in loads)),
        source_A79R1_recorded_cost=p['source_A79R1_recorded_cost'],cost_is_lower_bound=any(r['status']!='complete' for r in rows) or any(v['status'] not in ('complete','resource_limited') for v in loads),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# A80 DiffPIR原目标适配验证','','旧倍率质量复用A79R1；仅FRESH两臂耗时可作同场比值。','','|数据集|任务|方法|PSNR|SSIM|LPIPS|秒数|clock来源|','|---|---|---|---:|---:|---:|---:|---|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['dataset']}|{r['task']}|{r['arm']}|{q['psnr']:.3f}|{q['ssim']:.4f}|{q['lpips']:.4f}|{r['mean_full_seconds']:.3f}|{r['clock_source']}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n');print('A80 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH;bundle,commit=exporter.export_and_commit({'A80':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A80 host-aware DiffPIR adaptation uploaded.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a79r1-root',default=PARENT)
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device-uuid');p.add_argument('--check-worker-cli',action='store_true')
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
