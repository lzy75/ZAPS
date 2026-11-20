"""A123 complete FFHQ trajectories; reuse A122 controls, preserve original code."""
import argparse
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from utils import full_diagonal_transfer as rules
from utils import matched_response_transfer_runtime as response
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import native_ray_transfer_runtime as shared
from utils import historical_ffhq_runtime as precision
from utils import transfer_diagonal_runtime as native_control
from utils.complete_diagonal_transfer_session import running
from utils.a54_session import exclusive

previous=rules.sparse.previous
PROJECTS=previous.PROJECTS
ARMS=('BASE','SHRINK','OLD_DIAGONAL','SPARSE_DIAGONAL','FULL_DIAGONAL')


def prepare(host,root):
    parent=rules.parent(host); p=files.read(parent/'plan.json')
    files.require(files.read(parent/'session.json')['status']=='complete','A122 must be complete')
    files.require(p['host']==host and p['dataset']=='ffhq','A122 method/dataset differs')
    for name,sha in p['inputs'].items():
        files.require(files.digest(Path(name))==sha,'Frozen A122 implementation changed: '+name)
    files.require(files.read(parent/'tests.json')['returncode']==0,'A122 numerical gates did not pass')
    entries=previous.load_sources(host,'ffhq'); manifests=[]
    sources={r['case']['key']:r for r in p['sources']}
    files.require(len(entries)==len(sources)==12,'Need the same 12 FFHQ cases')
    for entry in entries:
        case=entry['case']; source=sources[case['key']]
        files.require(source['case']==case and case['sigma']==.05,'A122 case differs')
        files.require(source['source']==str(entry['root']) and
                      files.digest(entry['root']/'plan.json')==source['plan_sha256'],'A122 source plan changed')
        files.require(files.digest(case['image'])==entry['plan']['inputs'][case['image']],'Image changed')
        entry['observation']=previous.observation(entry,host)
        files.require(files.digest(entry['observation']['path'])==source['observation_sha256'],'A122 y changed')
        cached={}
        for arm in ARMS[:-1]:
            old_arm='MATCHED_DIAGONAL' if arm=='SPARSE_DIAGONAL' else arm
            record=parent/'cases'/case['key']/old_arm/'run.json'
            cached[arm]=dict(path=str(record),sha256=files.digest(record))
        manifests.append(dict(case=case,cached=cached))
    paths=(Path(__file__),Path(rules.__file__),Path(response.__file__),
           PROJECTS/'experiment_cards/A123_full_diagonal_transfer_20261121.md')
    plan=dict(experiment='A123',host=host,dataset='ffhq',parent=str(parent),parent_plan_sha256=files.digest(parent/'plan.json'),
              sources=manifests,positions=list(rules.positions(host)),budget=rules.budget(host),
              native_steps=rules.native_steps(host),inputs={str(path):files.digest(path) for path in paths},
              scope='Only intervention coverage changes from A122; same solver, target and scalar rule',
              timing='New full diagonal measured now; completed scalar/native/old/sparse clocks inherited from A122')
    if (root/'plan.json').exists(): files.require(files.read(root/'plan.json')==plan,'Frozen A123 plan changed')
    else: files.write(root/'plan.json',plan)
    shutil.copyfile(paths[-1],root/'experiment_card.md')
    for entry,manifest in zip(entries,manifests):
        for arm,source in manifest['cached'].items():
            path=Path(source['path']); cached=files.read(path)
            files.require(cached['case']==entry['case'],'Cached case identity differs')
            resolved,row=rules.resolve_record(path)
            target=root/'cases'/entry['case']['key']/arm/'run.json'
            imported=dict(row,arm=arm,reused=True,source_record=str(resolved),source_sha256=files.digest(resolved),
                          a122_record=str(path),a122_record_sha256=source['sha256'])
            if target.exists(): files.require(files.read(target)==imported,'Cached control record changed')
            else: files.write(target,imported)
    return entries


def sample(entry,host,model,wavelet,count,trace,tick):
    import torch
    y=torch.load(entry['observation']['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
    op=previous.operator(entry,entry['observation'],host,model.device,count)
    rt=previous.native(entry,host)
    apply=response.controller(model,wavelet,'RESPONSE_DIAG',count,rules.positions(host),trace)
    run=response.instrument(rt,apply,host)
    if host=='pigdm':
        return run('A123',model,op,y,entry['case'],'SHRINK',count,progress=tick)
    p=entry['plan']
    return run(model,op,y,entry['case'],'SHRINK',p['diffusion_config'],p['dps_root'],count,progress=tick)


def measure(root,entry,host,model,wavelet):
    import torch
    from modules.dataset_loader import tensor_to_image
    from utils.diag_noise_feasible_zaps import metrics
    case=entry['case']; folder=root/'cases'/case['key']/'FULL_DIAGONAL'; path=folder/'run.json'
    if path.exists():
        files.require(files.read(path)['status']=='complete','Unfinished attempt retained; no automatic retry')
        return
    free,_=torch.cuda.mem_get_info()
    if free < 2*1024**3: raise MemoryError('Need 2GiB current free margin before a new attempt')
    folder.mkdir(parents=True,exist_ok=False)
    row=dict(case=case,arm='FULL_DIAGONAL',status='running',counts={}); files.write(path,row)
    count={}; trace=[]
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        versions=[p._version for p in model.model.parameters()]
        scope=native_control.dps_reference_scope() if host=='dps' else nullcontext()
        with scope,precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count:
            native_control.initialize_counts(count)
            def tick(step,measured):
                row.update(counts=dict(measured),native_steps=step,diagonal_steps=len(trace))
                files.write(path,row)
                print('A123 PROGRESS:',host,case['key'],step,'/',rules.native_steps(host),
                      'F=',measured['forwards'],'/',rules.budget(host)['forwards'],flush=True)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,
                lambda:sample(entry,host,model,wavelet,count,trace,tick))
            final=shared.rng()
        expected=rules.budget(host)
        for key in ('forwards','input_vjp_attempts','input_jvp_attempts','response_vjp_attempts',
                    'response_vjp_forwards','response_jvp_forwards','response_value_forwards'):
            files.require(count.get(key,0)==expected[key],f'A123 budget differs: {key}')
        files.require(count['forward_attempts']==count['forwards'] and count['optimizer_backward_attempts']==0,
                      'Incomplete forward or unexpected optimizer backward')
        files.require([r['step'] for r in trace]==list(rules.positions(host)) and trace[-1]['next_time']==-1,
                      'Full coverage/terminal identity differs')
        files.require(all(p.grad is None and p._version==v for p,v in zip(model.model.parameters(),versions)),
                      'Model parameters/gradients changed')
        pairing=previous.fingerprint(detail,final,count,host); pairing.pop('calls',None)
        reference=files.read(rules.parent(host)/'cases'/case['key']/'SHRINK'/'run.json')
        files.require(reference['pairing']==pairing,'Native initial/final RNG or grid differs from A122 scalar')
        start=time.perf_counter(); quality=metrics(out,shared.load_image(case['image']))
        metric_seconds=time.perf_counter()-start; start=time.perf_counter()
        torch.save(out,folder/'output.pt'); tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',trace)
        row.update(status='complete',metrics=quality,full_seconds=seconds,counts=count,pairing=pairing,
                   metric_seconds=metric_seconds,artifact_seconds=time.perf_counter()-start,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   response_derivative_split={k:expected[k] for k in ('network_response_vjp','network_response_jvp',
                       'terminal_operator_vjp','terminal_identity_jvp')},
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','output.png','trace.json')})
        files.write(path,row); print('A123 COMPLETE:',host,case['key'],quality,'seconds=',round(seconds,2),flush=True)
    except BaseException as error:
        files.write(folder/'partial_trace.json',trace)
        row.update(status='failed',counts=count,error=repr(error),traceback=traceback.format_exc())
        files.write(path,row); raise


def summarize(root):
    rows=[files.read(p) for p in (root/'cases').glob('*/*/run.json')]
    groups=[]; comparisons=[]
    for task in previous.TASKS:
        selected={arm:{r['case']['key']:r for r in rows if r['case']['task']==task and r['arm']==arm and
                       r['status']=='complete'} for arm in ARMS}
        for arm,data in selected.items():
            if data:
                groups.append(dict(task=task,arm=arm,images=len(data),
                    quality={k:statistics.mean(r['metrics'][k] for r in data.values()) for k in ('psnr','ssim','lpips')},
                    mean_full_seconds=statistics.mean(r['full_seconds'] for r in data.values()),
                    clock_source='new_full_run' if arm=='FULL_DIAGONAL' else 'inherited_A122',
                    nfe=sorted({r['counts']['forwards'] for r in data.values()})))
        for reference in ARMS[:-1]:
            a,b=selected['FULL_DIAGONAL'],selected[reference]; keys=a.keys() & b.keys()
            if keys:
                delta={k:statistics.mean(a[key]['metrics'][k]-b[key]['metrics'][k] for key in keys)
                       for k in ('psnr','ssim','lpips')}
                wins=sum(a[key]['metrics']['psnr']>b[key]['metrics']['psnr'] for key in keys)
                comparisons.append(dict(task=task,reference=reference,images=len(keys),delta=delta,psnr_wins=wins,
                    positive=len(keys)==3 and wins>=2 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
    cost={}
    for row in rows:
        if row.get('reused'): continue
        for key,value in row.get('counts',{}).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool): cost[key]=cost.get(key,0)+value
    result=dict(experiment='A123',host=files.read(root/'plan.json')['host'],planned_runs=60,
                completed_runs=sum(r['status']=='complete' for r in rows),planned_new_runs=12,
                new_complete_runs=sum(r['status']=='complete' and not r.get('reused') for r in rows),
                groups=groups,comparisons=comparisons,actual_new_cost=cost,
                cost_is_lower_bound=any(r['status']!='complete' for r in rows),
                separate_model_gate=files.read(root/'load.json') if (root/'load.json').exists() else {},
                scope='A122 same FFHQ 3 images x4 tasks; unchanged diagonal algorithm used at ALL positions',
                timing='New full clocks; controls inherited, not same-session retiming; no acceleration claim')
    files.write(root/'summary.json',result); print('A123 SUMMARY:',result,flush=True); return result


def execute(host,root):
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    files.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Allocate one visible CUDA device')
    entries=prepare(host,root); checkpoint=entries[0]['plan']['checkpoint']
    files.require(all(e['plan']['checkpoint']==checkpoint for e in entries),'Checkpoint identities differ')
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Checkpoint changed')
    free,_=torch.cuda.mem_get_info()
    if free < 6*1024**3: raise MemoryError('Need >=6GiB free before model load')
    command=[sys.executable,'-m','unittest','-v','utils.test_full_diagonal_transfer']
    started=time.perf_counter(); tests=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True)
    print(tests.stdout+tests.stderr,end='',flush=True)
    files.write(root/'tests.json',dict(command=command,returncode=tests.returncode,seconds=time.perf_counter()-started,
                                     output=tests.stdout+tests.stderr,cost_scope='Toy gate, separate from method cost'))
    tests.check_returncode()
    started=time.perf_counter(); model=load_diffusion_model('ffhq','cuda:0')
    model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval(); wavelet=OrthogonalDWT2D(wave='db4',level=3).to(model.device)
    torch.cuda.synchronize(); load=dict(model_load_seconds=time.perf_counter()-started)
    files.write(root/'load.json',load)
    with frozen_network(model.model),input_vjp_parameters(model.model):
        from utils.pigdm_noisy_r1_runtime import original_diffusion
        gate={}
        try:
            with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as gate:
                x=torch.zeros(1,3,256,256,device=model.device)
                predict=lambda a:response.pigdm_predict(model,original_diffusion(),a,499)
                gate['input_jvp_attempts']=1
                with torch.no_grad(),response.forward_ad_checkpoint_scope(model.model):
                    primal,tangent=torch.func.jvp(predict,(x,),(torch.ones_like(x),))
                files.require(bool(torch.isfinite(primal).all() and torch.isfinite(tangent).all()),'Checkpoint JVP failed')
        finally:
            load.update(checkpoint_probe=gate,peak_reserved_bytes=torch.cuda.max_memory_reserved())
            files.write(root/'load.json',load)
        del x,primal,tangent; torch.cuda.empty_cache()
        for entry in entries:
            measure(root,entry,host,model,wavelet); summarize(root)


def worker(host,root):
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'worker.lock'):
        old=files.read(root/'session.json') if (root/'session.json').exists() else {}
        files.require(old.get('status') not in ('running','failed'),'Prior unfinished/failed run retained; use --status')
        state=dict(status='running',pid=os.getpid()); files.write(root/'session.json',state)
        try:
            if old.get('status')!='complete': execute(host,root)
            files.require(summarize(root)['completed_runs']==60,'Incomplete A123 results')
            state['status']='complete'
        except BaseException as error:
            state.update(status='resource_limited' if isinstance(error,MemoryError) else 'failed',
                         error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            files.write(root/'session.json',state)
            if (root/'plan.json').exists(): summarize(root)
        from utils import archive_refinement_confirmation as exporter
        exporter.BRANCH=f'results/{host}-full-diagonal-ffhq-20261121'
        _,commit=exporter.export_and_commit({host:root})
        subprocess.run(['git','push','origin',exporter.BRANCH],cwd=PROJECTS.parent,check=True)
        files.write(root/'upload.json',dict(status='complete',branch=exporter.BRANCH,commit=commit))
        print('RESULTS_PUSH_PASS: A123',host,commit,flush=True)


def follow(host,gpu,root):
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'client.lock'):
        launcher=root/'launcher.json'; old=files.read(launcher) if launcher.exists() else {}
        pid=old.get('pid'); offset=0; log=root/'worker.log'
        if not pid or not running(pid):
            state=files.read(root/'session.json') if (root/'session.json').exists() else {}
            files.require(state.get('status') not in ('running','failed'),'Prior run retained; use --status')
            offset=log.stat().st_size if log.exists() else 0
            with log.open('ab') as stream:
                proc=subprocess.Popen([sys.executable,'-u','-m','utils.diag_full_diagonal_transfer',host,'--worker'],
                    cwd=PROJECTS,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
            pid=proc.pid; files.write(launcher,dict(pid=pid,gpu=gpu))
    print('A123 FOLLOW:',host,'PID',pid,'log',log,flush=True)
    while True:
        if log.exists():
            with log.open() as stream:
                stream.seek(offset); text=stream.read(); offset=stream.tell()
            if text: print(text,end='',flush=True)
        if not running(pid): break
        time.sleep(5)
    if log.exists():
        with log.open() as stream:
            stream.seek(offset); print(stream.read(),end='',flush=True)
    state=files.read(root/'session.json') if (root/'session.json').exists() else {}
    files.require(state.get('status')=='complete',f'A123 stopped; see {log}')
    files.require((root/'upload.json').exists() and files.read(root/'upload.json')['status']=='complete',
                  f'Reconstruction complete, upload incomplete; see {log}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('host',choices=('dps','pigdm')); p.add_argument('--gpu',default='0')
    p.add_argument('--status',action='store_true'); p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(); root=rules.directory(args.host)
    if args.status:
        print(files.read(root/'session.json') if (root/'session.json').exists() else {'status':'not_started'})
        for path in (root/'cases').glob('*/FULL_DIAGONAL/run.json'):
            row=files.read(path)
            if row['status']!='complete':
                print('A123 CASE:',row['case']['key'],row['status'],
                      'native_steps=',row.get('native_steps',0),'/',rules.native_steps(args.host),
                      'F=',row.get('counts',{}).get('forwards',0),'/',rules.budget(args.host)['forwards'],
                      row.get('error',''))
        if (root/'plan.json').exists(): summarize(root)
    elif args.worker: worker(args.host,root)
    else: follow(args.host,args.gpu,root)


if __name__=='__main__': main()
