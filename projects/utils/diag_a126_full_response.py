"""A126: full A118 coefficient refinement through the native next clean predictor."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from utils import full_coefficient_response as policy
from utils import diagonal_parameter_tuning as proxy
from utils import diag_a118_blur_transfer as previous
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native
from utils import native_ray_transfer_runtime as shared
from utils import historical_ffhq_runtime as precision
from utils.a54_session import exclusive
from utils.complete_diagonal_transfer_session import running

PROJECTS=previous.PROJECTS
ROOT=previous.history.BASE/'diag_a126_full_response'
CARD=PROJECTS/'experiment_cards/A126_full_coefficient_response.md'


def prepare(host,root):
    source=previous.OUTPUTS/f'A124R1_{host}_ffhq'
    files.require(files.read(source/'summary.json')['status']=='complete','Need completed A124R1')
    old=files.read(source/'plan.json')
    files.require(old['host']==host and len(old['cases'])==20,'A124R1 source differs')
    for name,sha in old['sources'].items():
        files.require(files.digest(name)==sha,'Bound A124R1 source changed: '+name)
    a118=files.read(Path(old['parent'])/'plan.json')
    files.require(files.digest(Path(old['parent'])/'plan.json')==old['parent_plan_sha256'],'A118 plan changed')
    files.require(files.digest(old['motion_kernel'])==old['motion_kernel_sha256'],'A118 kernel changed')
    files.require(files.digest(old['checkpoint']['path'])==old['checkpoint']['sha256'],'FFHQ weights changed')
    grid=list(range(1000)) if host=='dps' else list(range(0,1000,10))
    active=proxy.mapped_times(a118['grid'],grid)
    ids=sorted({c['image_id'] for c in old['cases']})
    files.require(len(ids)==10,'Need the fixed ten-image panel')
    cases=[]
    for case in old['cases']:
        if case['image_id'] not in ids[:3]: continue
        files.require(files.digest(case['image'])==case['image_sha256'],'Input image changed')
        if host=='dps':
            y,sha=case['a118_y'],case['a118_y_sha256']
        else:
            obs=files.read(source/'observations'/case['key']/'observation.json')
            y,sha=obs['path'],obs['sha256']
        files.require(files.digest(y)==sha,'A124 observation changed')
        baseline=source/'cases'/case['key']/'BASE'/'run.json'
        row=files.read(baseline)
        files.require(row['status']=='complete' and row['case']==case and row['y_sha256']==sha,'Original baseline differs')
        cases.append(dict(case,y_path=y,y_sha256=sha,baseline=str(baseline),baseline_sha256=files.digest(baseline)))
    plan=dict(experiment='A126',host=host,nfe=old['nfe'],source=str(source),source_plan_sha256=files.digest(source/'plan.json'),
              checkpoint=old['checkpoint'],motion_kernel=old['motion_kernel'],dps_configs=old['dps_configs'],
              cases=cases,target_times=a118['grid'],active_times=active,
              ray_iterations=policy.RAY_ITERATIONS,diagonal_iterations=policy.DIAGONAL_ITERATIONS,
              max_line_trials=policy.MAX_TRIALS,sources={str(p):files.digest(p) for p in
                  (Path(__file__),Path(policy.__file__),Path(policy.response.__file__),Path(proxy.__file__),CARD)},
              protocol='Six existing FFHQ cases; four independently timed arms; no parameter selection')
    if (root/'plan.json').exists():
        files.require(files.read(root/'plan.json')==plan,'Frozen tuning protocol changed')
    else:
        files.write(root/'plan.json',plan)
    return plan


def records(root):
    return [files.read(path) for path in sorted((root/'cases').glob('*/*/run.json'))]


def measure(root,plan,case,arm,model,wavelet,kernel):
    import torch
    from modules.dataset_loader import tensor_to_image
    from utils.diag_noise_feasible_zaps import metrics
    folder=root/'cases'/case['key']/arm
    path=folder/'run.json'
    if path.exists():
        row=files.read(path)
        files.require(row['status']=='complete' and row['plan_sha256']==files.digest(root/'plan.json') and
                      all(files.digest(folder/k)==v for k,v in row['artifacts'].items()),'Prior unfinished/changed attempt retained')
        return
    if torch.cuda.mem_get_info()[0]<2*1024**3:
        raise MemoryError('Need 2GiB current free before a new run')
    folder.mkdir(parents=True,exist_ok=False)
    row=dict(case=case,arm=arm,status='running',plan_sha256=files.digest(root/'plan.json'),parameters=dict(ray_iterations=policy.RAY_ITERATIONS,diagonal_iterations=policy.DIAGONAL_ITERATIONS))
    files.write(path,row)
    count={}; trace=[]
    try:
        scope=native.dps_reference_scope() if plan['host']=='dps' else nullcontext()
        with scope,precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count:
            native.initialize_counts(count)
            versions=[p._version for p in model.model.parameters()]
            def tick(step,counters):
                row.update(native_step=step,counts=dict(counters)); files.write(path,row)
            def run():
                y=torch.load(case['y_path'],map_location=model.device,weights_only=True)['y']
                op=previous.make_operator(plan,case,model.device,kernel,count)
                rt=previous.history.native(dict(case=case),plan['host'])
                if arm=='PROXY_DIAG':
                    apply=proxy.refine(wavelet,1,1.,count,trace)
                    sampler=proxy.sampler(rt,plan['host'],proxy.label(1,1.),plan['active_times'],apply)
                else:
                    apply=policy.controller(model,wavelet,arm,count,plan['active_times'],trace) if arm!='SCALAR' else None
                    sampler=policy.instrument(rt,plan['host'],arm,plan['active_times'],apply)
                if plan['host']=='pigdm':
                    return sampler('A126',model,op,y,case,'SHRINK',count,progress=tick)
                config=plan['dps_configs'][case['task']]
                return sampler(model,op,y,case,'SHRINK',config['diffusion_config'],config['dps_root'],count,progress=tick)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,run)
            final=shared.rng()
        files.require(count['forwards']==count['forward_attempts']==plan['nfe']+count.get('response_network_forwards',0),
                      'Unaccounted denoiser forward')
        files.require(count['input_vjp_attempts']==plan['nfe']+count.get('response_vjp_attempts',0) and
                      count.get('input_jvp_attempts',0)==count.get('response_jvp_forwards',0) and
                      count['optimizer_backward_attempts']==0,'Derivative accounting differs')
        expected=[t for t in detail['grid'] if t in plan['active_times']]
        files.require(len(expected)==30 and [r['t'] for r in detail['trace'] if 'weight' in r]==expected and
                      [r['t'] for r in trace]==(expected if arm!='SCALAR' else []),'A118-aligned feedback coverage differs')
        pairing=previous.history.fingerprint(detail,final,count,plan['host'])
        reference=files.read(case['baseline'])['pairing'].copy()
        reference.pop('feedback_entry_sha256',None)
        reference.pop('calls',None); pairing.pop('calls',None)
        files.require(pairing==reference,'Native initial/final RNG or grid differs from baseline')
        files.require(all(p.grad is None and p._version==v for p,v in zip(model.model.parameters(),versions)),
                      'NN weights/gradients changed')
        started=time.perf_counter(); quality=metrics(out,shared.load_image(case['image']))
        metric_seconds=time.perf_counter()-started
        started=time.perf_counter()
        torch.save(out,folder/'output.pt'); tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',trace)
        row.update(status='complete',counts=count,metrics=quality,full_seconds=seconds,metric_seconds=metric_seconds,
                   artifact_seconds=time.perf_counter()-started,
                   pairing=pairing,artifacts={k:files.digest(folder/k) for k in ('output.pt','output.png','trace.json')})
        files.write(path,row)
        print('A126 COMPLETE',plan['host'],case['key'],arm,quality,round(seconds,2),flush=True)
    except BaseException as error:
        row.update(status='failed',counts=count,error=repr(error),traceback=traceback.format_exc())
        files.write(path,row); files.write(folder/'partial_trace.json',trace); raise


def summarize(root,plan,status):
    rows=records(root); groups=[]; comparisons=[]
    for task in previous.TASKS:
        means={}
        selected={arm:{r['case']['key']:r for r in rows if r['status']=='complete' and
                      r['case']['task']==task and r['arm']==arm}
                  for arm in ('BASE','SCALAR','PROXY_DIAG','RESPONSE_RAY','RESPONSE_DIAG')}
        for arm,items in selected.items():
            if items:
                means[arm]=dict(images=len(items),quality={k:statistics.mean(r['metrics'][k] for r in items.values())
                    for k in previous.METRICS},seconds=statistics.mean(r['full_seconds'] for r in items.values()),
                    nfe=sorted({r['counts']['forwards'] for r in items.values()}),
                    vjp=sorted({r['counts']['input_vjp_attempts'] for r in items.values()}),
                    jvp=sorted({r['counts'].get('input_jvp_attempts',0) for r in items.values()}),
                    timing_source='historical' if arm=='BASE' else 'same_session')
        groups.append(dict(task=task,arms=means))
        for reference in ('SCALAR','PROXY_DIAG','RESPONSE_RAY'):
            a,b=selected['RESPONSE_DIAG'],selected[reference]; keys=a.keys() & b.keys()
            if keys:
                delta={k:statistics.mean(a[i]['metrics'][k]-b[i]['metrics'][k] for i in keys) for k in previous.METRICS}
                wins=sum(a[i]['metrics']['psnr']>b[i]['metrics']['psnr'] for i in keys)
                comparisons.append(dict(task=task,reference=reference,images=len(keys),delta=delta,psnr_wins=wins,
                    clock_ratio=sum(a[i]['full_seconds'] for i in keys)/sum(b[i]['full_seconds'] for i in keys),
                    positive=len(keys)==3 and wins>=2 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
    cost={k:sum(r.get('counts',{}).get(k,0) for r in rows if not r.get('reused')) for k in
          ('forwards','forward_attempts','input_vjp_attempts','input_jvp_attempts','refinement_H','refinement_Ht',
           'response_network_forwards','terminal_response_vjp','terminal_response_jvp')}
    report=dict(experiment='A126',status=status,host=plan['host'],planned_new_runs=24,
                new_completed=sum(r['status']=='complete' and not r.get('reused') for r in rows),
                groups=groups,comparisons=comparisons,actual_new_cost=cost,
                cost_is_lower_bound=any(r['status']!='complete' for r in rows),
                limits='Same three reused FFHQ images/two blurs. Actual NFE/AD differ across arms and are reported; no equal-NFE or final-quality guarantee.')
    files.write(root/'summary.json',report)
    return report


def checkpoint_probe(root,plan,model,kernel):
    import torch
    started=time.perf_counter()
    case=plan['cases'][0]
    scope=native.dps_reference_scope() if plan['host']=='dps' else nullcontext()
    with scope, clock.counters(model.model) as count:
        native.initialize_counts(count)
        op=previous.make_operator(plan,case,model.device,kernel,count)
        rt=previous.history.native(dict(case=case),plan['host'])
        t=max(plan['active_times'])
        if plan['host']=='pigdm':
            diffusion=rt.original_diffusion(); s=t-10
            predict=lambda xx:policy.response.pigdm_predict(model,diffusion,xx,s)
        else:
            cfg=plan['dps_configs'][case['task']]
            sampler=rt.reference(str(Path(cfg['dps_root']).resolve()))[0].create_sampler(**cfg['diffusion_config'])
            index=sampler.timestep_map.index(t)-1
            predict=lambda xx:policy.response.dps_predict(sampler,model,xx,index)
        x=torch.zeros(1,3,256,256,device=model.device)
        provider=policy.response.Provider(model.model,predict,op,x,op.H(x+.1),case['sigma'],count)
        value,grad=provider.gradient(x)
        primal,tangent=provider.jvp(x,torch.ones_like(x)*.01)
        error=float((value.double()-primal.double()).norm()/value.double().norm().clamp_min(1e-30))
        files.require(error<=1e-4 and bool(torch.isfinite(grad).all()) and bool(torch.isfinite(tangent).all()),
                      'Actual checkpoint derivative gate failed')
    torch.cuda.synchronize()
    files.write(root/'checkpoint_probe.json',dict(passed=True,relative_primal_error=error,counts=count,
                seconds=time.perf_counter()-started,
                scope='Separate real-checkpoint derivative admission; not reconstruction method cost'))


def execute(host,root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    plan=prepare(host,root)
    for case in plan['cases']:
        original=files.read(case['baseline'])
        files.require(files.digest(case['baseline'])==case['baseline_sha256'],'Original baseline record changed')
        target=root/'cases'/case['key']/'BASE'/'run.json'
        if not target.exists():
            files.write(target,dict(original,case=case,reused=True,source_record=case['baseline']))
    files.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Allocate exactly one visible CUDA GPU')
    if torch.cuda.mem_get_info()[0]<6*1024**3 or shutil.disk_usage(root).free<1024**3:
        raise MemoryError('Need 6GiB GPU free and 1GiB output disk space')
    subprocess.run([sys.executable,'-m','unittest','-v','utils.test_full_coefficient_response'],cwd=PROJECTS,check=True)
    started=time.perf_counter()
    model=load_diffusion_model('ffhq','cuda:0')
    model.model.load_state_dict(torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval(); wavelet=OrthogonalDWT2D(wave='db4',level=3).to(model.device)
    kernel=np.load(plan['motion_kernel'],allow_pickle=False)
    torch.cuda.synchronize()
    files.write(root/'load.json',dict(seconds=time.perf_counter()-started,scope='Model/weights/wavelet/kernel loading only'))
    with frozen_network(model.model),input_vjp_parameters(model.model):
        checkpoint_probe(root,plan,model,kernel)
        for i,case in enumerate(plan['cases']):
            arms=['SCALAR','PROXY_DIAG','RESPONSE_RAY','RESPONSE_DIAG']
            for arm in arms[i%4:]+arms[:i%4]:
                measure(root,plan,case,arm,model,wavelet,kernel)
            print('A126 PROGRESS',host,summarize(root,plan,'running')['new_completed'],'/24',flush=True)
    report=summarize(root,plan,'complete')
    files.require(report['new_completed']==24,'Incomplete full-response comparison')
    return report


def worker(host,root):
    with exclusive(root/'worker.lock'):
        state=dict(status='running',pid=os.getpid()); files.write(root/'session.json',state)
        try:
            report=execute(host,root); state['status']='complete'; print('A126 SUMMARY',report,flush=True)
        except BaseException as error:
            state.update(status='failed',error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            files.write(root/'session.json',state)
            if (root/'plan.json').exists():
                summarize(root,files.read(root/'plan.json'),state['status'])


def archive(host,root):
    import tempfile
    from utils import archive_refinement_confirmation as exporter
    summary=files.read(root/'summary.json')
    files.require(summary['status']=='complete' and summary['new_completed']==24,'Require completed A126 results')
    with tempfile.TemporaryDirectory(prefix='A126_export_',dir=root.parent) as directory:
        repo=Path(directory)/'repo'
        subprocess.run(['git','clone','--shared','--no-checkout',str(PROJECTS.parent),str(repo)],check=True)
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        for key in ('user.name','user.email'):
            git('config',key,subprocess.check_output(['git','-C',str(PROJECTS.parent),'config','--get',key],text=True).strip())
        git('remote','set-url','origin',subprocess.check_output(['git','-C',str(PROJECTS.parent),'remote','get-url','origin'],text=True).strip())
        branch=f'results/a126-full-coefficient-response-{host}'
        if git('ls-remote','--heads','origin',branch):
            git('fetch','origin',f'refs/heads/{branch}:refs/heads/{branch}')
        (repo/'projects').mkdir()
        exporter.BRANCH=branch
        _,commit=exporter.export_and_commit({host:root},projects=repo/'projects')
        git('push','origin',branch)
    files.write(root/'upload.json',dict(status='complete',branch=branch,commit=commit))
    print('RESULTS_PUSH_PASS',commit,branch,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','pigdm')); parser.add_argument('--gpu',default='0')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--status',action='store_true'); modes.add_argument('--archive',action='store_true')
    modes.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(); root=ROOT/f'A126_{args.host}_ffhq'
    if args.status:
        state=files.read(root/'session.json') if (root/'session.json').exists() else {'status':'not_started'}
        print(state)
        if (root/'plan.json').exists():
            print('A126 SUMMARY',summarize(root,files.read(root/'plan.json'),state['status']))
        for row in records(root):
            if row['status']!='complete':
                print('A126 ACTIVE',row['case']['key'],row['arm'],row.get('native_step',0),row.get('error',''))
        return
    root.mkdir(parents=True,exist_ok=True)
    if args.archive:
        with exclusive(root/'worker.lock'):
            archive(args.host,root)
        return
    if args.worker:
        worker(args.host,root); return
    with exclusive(root/'client.lock'):
        old=files.read(root/'session.json') if (root/'session.json').exists() else {}
        if old.get('status')=='complete':
            print('A126 SUMMARY',files.read(root/'summary.json')); return
        launch=files.read(root/'launcher.json') if (root/'launcher.json').exists() else {}
        pid=launch.get('pid'); log=root/'worker.log'; offset=0
        if not pid or not running(pid):
            files.require(old.get('status') not in ('failed','running'),'Prior attempt retained; inspect --status')
            offset=log.stat().st_size if log.exists() else 0
            with log.open('ab') as stream:
                process=subprocess.Popen([sys.executable,'-u','-m','utils.diag_a126_full_response',args.host,'--worker'],
                    cwd=PROJECTS,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu),stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True)
            pid=process.pid; files.write(root/'launcher.json',dict(pid=pid,gpu=args.gpu))
    while True:
        active=running(pid)
        if log.exists():
            with log.open() as stream:
                stream.seek(offset); data=stream.read(); offset=stream.tell()
            if data: print(data,end='',flush=True)
        if not active: break
        time.sleep(5)
    files.require((root/'session.json').exists() and files.read(root/'session.json')['status']=='complete',f'A126 stopped; see {log}')


if __name__=='__main__': main()
