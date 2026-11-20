"""A122: adapt only the diagonal module on the unchanged FFHQ transfer panel."""
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
from utils import diag_complete_diagonal_transfer as previous
from utils import matched_response_transfer_runtime as response
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import native_ray_transfer_runtime as shared
from utils import historical_ffhq_runtime as precision
from utils import transfer_diagonal_runtime as original
from utils.a54_session import exclusive

PROJECTS = previous.PROJECTS
ARMS = ('BASE', 'SHRINK', 'OLD_DIAGONAL', 'MATCHED_DIAGONAL')


def directory(host):
    return previous.BASE/'diag_matched_response_transfer'/f'A122_{host}_ffhq_20261120'


def positions(host):
    return (200,500,800) if host == 'dps' else (20,50,80)


def import_result(root, entry, arm, host):
    if arm == 'BASE':
        row, folder = previous.existing(entry, 'BASE', host)
    else:
        folder = previous.directory(host,'ffhq')/'cases'/entry['case']['key']/'DIAGONAL'
        row = files.read(folder/'run.json')
        files.require(row['status']=='complete' and row['case']==entry['case'], 'Old diagonal identity differs')
    if row.get('reused'):
        folder = Path(row['source_record']).parent
        files.require(files.digest(folder/'run.json')==row['source_sha256'], 'Reused source record changed')
        row = files.read(folder/'run.json')
    target = root/'cases'/entry['case']['key']/arm/'run.json'
    files.write(target, dict(row, arm=arm, reused=True, source_record=str(folder/'run.json'),
                            source_sha256=files.digest(folder/'run.json')))


def sample(entry, obs, host, model, wavelet, arm, count, tick=None):
    import torch
    y = torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
    op = previous.operator(entry,obs,host,model.device,count)
    rt = previous.native(entry,host)
    trace = []
    run = rt.run
    if arm == 'MATCHED_DIAGONAL':
        apply = response.controller(model,wavelet,'RESPONSE_DIAG',count,positions(host),trace)
        run = response.instrument(rt,apply,host)
    if host == 'pigdm':
        out, detail = run('A122',model,op,y,entry['case'],'SHRINK',count,progress=tick)
    else:
        p = entry['plan']
        out, detail = run(model,op,y,entry['case'],'SHRINK',p['diffusion_config'],p['dps_root'],count,progress=tick)
    return out, detail, trace


def measure(root,entry,host,model,wavelet,arm):
    import torch
    from modules.dataset_loader import tensor_to_image
    from utils.diag_noise_feasible_zaps import metrics
    case = entry['case']; folder = root/'cases'/case['key']/arm
    record_path = folder/'run.json'
    if record_path.exists():
        files.require(files.read(record_path)['status']=='complete','Unfinished attempt retained; no automatic retry')
        return
    free,_ = torch.cuda.mem_get_info()
    files.require(free >= 2*1024**3, 'Insufficient memory margin before new attempt')
    folder.mkdir(parents=True,exist_ok=False)
    row = dict(case=case,arm=arm,status='running',counts={})
    files.write(record_path,row); count = {}
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        scope = original.dps_reference_scope() if host=='dps' else nullcontext()
        versions = [p._version for p in model.model.parameters()]
        with scope, precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count:
            original.initialize_counts(count)
            def tick(step, measured):
                row.update(counts=dict(measured),native_steps=step)
                files.write(record_path,row)
                print('A122 PROGRESS:',host,case['key'],arm,step,flush=True)
            (out,detail,trace), seconds = clock.clocks(torch.cuda.synchronize,
                lambda:sample(entry,entry['observation'],host,model,wavelet,arm,count,tick))
            final = shared.rng()
        native = 1000 if host=='dps' else 100
        extra = 24 if arm=='MATCHED_DIAGONAL' else 0
        files.require(count['forwards']==count['forward_attempts']==native+extra,'Actual NFE differs from fixed budget')
        files.require(count['input_vjp_attempts']==native+(3 if extra else 0),'Input VJP count differs')
        files.require(count.get('input_jvp_attempts',0)==(6 if extra else 0),'Input JVP count differs')
        files.require(count['optimizer_backward_attempts']==0,'Unexpected optimizer backward')
        files.require([r['step'] for r in trace]==(list(positions(host)) if extra else []),'Intervention positions differ')
        files.require(all(p.grad is None and p._version==v for p,v in zip(model.model.parameters(),versions)),
                      'Network weights/gradients changed')
        pairing = previous.fingerprint(detail,final,count,host)
        pairing.pop('calls',None)  # Extra queries differ, native grid/RNG must match.
        replay_error = None
        if arm=='SHRINK':
            _,source = previous.existing(entry,'SHRINK',host)
            expected = torch.load(source/'output.pt',map_location='cpu',weights_only=True)
            replay_error = float((out.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30))
            files.require(replay_error <= 1e-4,f'Unchanged scalar replay differs: {replay_error:.3g}')
        other = folder.parent/('SHRINK' if extra else 'MATCHED_DIAGONAL')/'run.json'
        if other.exists():
            files.require(files.read(other)['pairing']==pairing,'Native grid/init/final sampling RNG differs')
        start = time.perf_counter()
        quality = metrics(out,shared.load_image(case['image']))
        metric_seconds = time.perf_counter()-start
        start = time.perf_counter()
        torch.save(out,folder/'output.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',trace)
        row.update(status='complete',metrics=quality,full_seconds=seconds,counts=count,pairing=pairing,
                   replay_relative_l2=replay_error,metric_seconds=metric_seconds,
                   artifact_seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','output.png','trace.json')})
        files.write(record_path,row)
        print('A122 COMPLETE:',host,case['key'],arm,quality,'seconds=',round(seconds,2),flush=True)
    except BaseException as error:
        row.update(status='failed',error=repr(error),counts=count,traceback=traceback.format_exc())
        files.write(record_path,row); raise


def summarize(root):
    rows = [files.read(p) for p in (root/'cases').glob('*/*/run.json')]
    groups, comparisons = [], []
    for task in previous.TASKS:
        selected = {arm:{r['case']['key']:r for r in rows if r['case']['task']==task and r['arm']==arm and
                         r['status']=='complete'} for arm in ARMS}
        for arm,data in selected.items():
            if data:
                groups.append(dict(task=task,arm=arm,images=len(data),
                    quality={k:statistics.mean(r['metrics'][k] for r in data.values()) for k in ('psnr','ssim','lpips')},
                    mean_full_seconds=statistics.mean(r['full_seconds'] for r in data.values()),
                    clock_source='historical' if arm in ('BASE','OLD_DIAGONAL') else 'same_session',
                    nfe=sorted({r['counts']['forwards'] for r in data.values() if 'forwards' in r.get('counts',{})})))
        for reference in ('BASE','SHRINK','OLD_DIAGONAL'):
            keys = selected['MATCHED_DIAGONAL'].keys() & selected[reference].keys()
            if keys:
                candidate = selected['MATCHED_DIAGONAL']; ref = selected[reference]
                delta = {k:statistics.mean(candidate[key]['metrics'][k]-ref[key]['metrics'][k] for key in keys)
                         for k in ('psnr','ssim','lpips')}
                wins = sum(candidate[key]['metrics']['psnr']>ref[key]['metrics']['psnr'] for key in keys)
                comparisons.append(dict(task=task,reference=reference,images=len(keys),delta=delta,psnr_wins=wins,
                    positive=len(keys)==3 and wins>=2 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
    cost = {}
    for row in rows:
        if row.get('reused'): continue
        for name,value in row.get('counts',{}).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool): cost[name]=cost.get(name,0)+value
    result = dict(experiment='A122',host=files.read(root/'plan.json')['host'],completed_runs=sum(r['status']=='complete' for r in rows),
                  planned_runs=48,new_complete_runs=sum(r['status']=='complete' and not r.get('reused') for r in rows),
                  planned_new_runs=24,groups=groups,comparisons=comparisons,actual_new_cost=cost,
                  cost_is_lower_bound=any(r['status']!='complete' for r in rows),
                  scope='Existing FFHQ three images x four tasks; only diagonal adaptation changes. No ZAPS changes.',
                  timing='New unchanged scalar and matched diagonal same-session full clocks; original/old diagonal clocks inherited.',
                  numerical_tests=files.read(root/'tests.json') if (root/'tests.json').exists() else None,
                  separate_model_gate=files.read(root/'load.json') if (root/'load.json').exists() else None,
                  limitations='Three interventions only; extra NFE/AD disclosed; actual response loss is not full image risk.')
    files.write(root/'summary.json',result)
    print('A122 SUMMARY:',result,flush=True)
    return result


def execute(args,root):
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    files.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Allocate exactly one visible CUDA GPU')
    entries = previous.load_sources(args.host,'ffhq')
    manifests = []
    for entry in entries:
        previous.existing(entry,'BASE',args.host); previous.existing(entry,'SHRINK',args.host)
        entry['observation'] = previous.observation(entry,args.host)
        p,c = entry['plan'],entry['case']
        files.require(c['sigma']==.05,'A122 freezes original .05 noise; no new noise scan')
        files.require(files.digest(c['image'])==p['inputs'][c['image']],'Original image changed')
        manifests.append(dict(case=c,source=str(entry['root']),plan_sha256=files.digest(entry['root']/'plan.json'),
                              observation_sha256=files.digest(entry['observation']['path'])))
    plan = dict(experiment='A122',host=args.host,dataset='ffhq',sources=manifests,positions=positions(args.host),
                scope='Only diagonal adaptation; original scalar/native loops unchanged',
                nfe={'SHRINK':1000 if args.host=='dps' else 100,'MATCHED_DIAGONAL':1024 if args.host=='dps' else 124},
                extra_derivatives={'input_vjp':3,'input_jvp':6},
                inputs={str(p):files.digest(p) for p in (Path(__file__),Path(response.__file__),
                    Path(previous.__file__),Path(original.__file__),Path(response.forward_ad_checkpoint_scope.__code__.co_filename),
                    PROJECTS/'experiment_cards/A122_matched_response_transfer_20261120.md')})
    # JSON normalizes tuples to lists; freeze identities before any continuation.
    plan['positions']=list(plan['positions'])
    if (root/'plan.json').exists():
        files.require(files.read(root/'plan.json')==plan,'Frozen A122 protocol or sources changed')
    else:
        files.write(root/'plan.json',plan)
    shutil.copyfile(PROJECTS/'experiment_cards/A122_matched_response_transfer_20261120.md',root/'experiment_card.md')
    for entry in entries:
        for arm in ('BASE','OLD_DIAGONAL'): import_result(root,entry,arm,args.host)
    checkpoint = entries[0]['plan']['checkpoint']
    files.require(all(e['plan']['checkpoint']==checkpoint for e in entries),'Source checkpoints differ')
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Pretrained weights changed')
    free,_ = torch.cuda.mem_get_info()
    if free < 6*1024**3: raise MemoryError('Need >=6GiB free before FFHQ model load')
    command=[sys.executable,'-m','unittest','-v','utils.test_matched_response_transfer']
    started=time.perf_counter()
    tests=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True)
    print(tests.stdout+tests.stderr,end='',flush=True)
    files.write(root/'tests.json',dict(command=command,returncode=tests.returncode,seconds=time.perf_counter()-started,
                                     output=tests.stdout+tests.stderr,
                                     cost_scope='Toy numerical gates, not method cost; individual NN totals not aggregated'))
    tests.check_returncode()
    start = time.perf_counter()
    model = load_diffusion_model('ffhq','cuda:0')
    model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval(); wavelet = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
    torch.cuda.synchronize()
    load = dict(model_load_seconds=time.perf_counter()-start)
    files.write(root/'load.json',load)
    with frozen_network(model.model),input_vjp_parameters(model.model):
        # Validate the actual checkpoint's custom-checkpoint/JVP interface before trajectories.
        with clock.counters(model.model) as gate:
            from utils.pigdm_noisy_r1_runtime import original_diffusion
            x = torch.zeros(1,3,256,256,device=model.device)
            predict = lambda a:response.pigdm_predict(model,original_diffusion(),a,499)
            with torch.no_grad(),response.forward_ad_checkpoint_scope(model.model):
                primal,tangent = torch.func.jvp(predict,(x,),(torch.ones_like(x),))
            files.require(bool(torch.isfinite(primal).all() and torch.isfinite(tangent).all()),'Actual checkpoint JVP gate failed')
            gate['input_jvp_attempts']=1
        load['checkpoint_probe']=gate
        load['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        files.write(root/'load.json',load)
        del x,primal,tangent; torch.cuda.empty_cache()
        for index,entry in enumerate(entries):
            # Replay unchanged scalar first: a failed source/parity gate stops before candidate.
            measure(root,entry,args.host,model,wavelet,'SHRINK')
            measure(root,entry,args.host,model,wavelet,'MATCHED_DIAGONAL')
            summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','pigdm'))
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--archive',action='store_true')
    args = parser.parse_args(); root = directory(args.host)
    if args.status:
        print(files.read(root/'session.json') if (root/'session.json').exists() else {'status':'not_started'})
        if (root/'plan.json').exists(): summarize(root)
        return
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'worker.lock'):
        previous_state = files.read(root/'session.json') if (root/'session.json').exists() else {}
        files.require(previous_state.get('status') not in ('failed','running'),'Prior failed/unfinished attempt retained; use --status')
        state = dict(status='running',pid=os.getpid()); files.write(root/'session.json',state)
        try:
            if previous_state.get('status')!='complete': execute(args,root)
            summary = summarize(root)
            files.require(summary['completed_runs']==48,'Incomplete four-task table')
            state['status']='complete'
        except BaseException as error:
            state.update(status='resource_limited' if isinstance(error,MemoryError) else 'failed',
                         error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            files.write(root/'session.json',state)
            if (root/'plan.json').exists(): summarize(root)
        if args.archive:
            from utils import archive_refinement_confirmation as exporter
            exporter.BRANCH=f'results/{args.host}-matched-diagonal-ffhq-20261120'
            _,commit = exporter.export_and_commit({args.host:root})
            subprocess.run(['git','push','origin',exporter.BRANCH],cwd=PROJECTS.parent,check=True)
            files.write(root/'upload.json',dict(status='complete',commit=commit,branch=exporter.BRANCH))
            print('RESULTS_PUSH_PASS: A122',args.host,commit,flush=True)


if __name__=='__main__':
    main()
