"""A124: paired low-cost transfer on the frozen A118 ten-image blur panel."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from utils import a118_blur_transfer_runtime as runtime
from utils import diag_complete_diagonal_transfer as history
from utils import historical_ffhq_runtime as precision
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native
from utils.a54_session import exclusive
from utils.complete_diagonal_transfer_session import running
from utils.shared_prefix_feedback import ROOT as A102

PROJECTS = history.PROJECTS
PARENT = Path(A102).with_name('A118_20261120')
TASKS = ('gaussian_deblur', 'motion_deblur')
METRICS = ('psnr', 'ssim', 'lpips')
OUTPUTS = history.BASE / 'diag_a118_blur_transfer'
CARD = PROJECTS / 'experiment_cards/A124_a118_blur_transfer.md'
REVISION = 'A124R1'


def failed_preflight(host):
    root = OUTPUTS/f'A124_{host}_ffhq'
    session = root/'session.json'
    if not session.exists():
        return None
    state = files.read(session)
    files.require(state['status'] != 'running' and not any((root/'cases').glob('*/*')),
                  'A124 has an active session or attempted reconstructions; inspect existing results before rerunning')
    error = state.get('error','')
    files.require(state['status']=='resource_limited' or
                  (state['status']=='failed' and 'CalledProcessError' in error and 'test_a118_blur_transfer' in error),
                  'A124 failure is unrelated to the repaired CUDA preflight')
    return dict(path=str(root),session_sha256=files.digest(session))


def load_plan(host, parent):
    summary = files.read(parent / 'summary.json')
    source = files.read(parent / 'plan.json')
    files.require(summary['status']=='complete' and summary['completed_cases']==summary['planned_cases']==20
                  and source['experiment']=='A118' and source['method_nfe']==300, 'Need completed A118 parent')
    for name, sha in source['sources'].items():
        files.require(files.digest(PROJECTS / name)==sha, 'A118 source changed: ' + name)
    kernel = parent / 'motion_kernel.npy'
    files.require(files.digest(kernel)==source['motion_kernel_sha256'], 'A118 kernel changed')
    cases = []
    for case in source['cases']:
        obs = files.read(parent / 'observations' / case['key'] / 'observation.json')
        files.require(case['task'] in TASKS and case['sigma']==obs['sigma']==.05 and
                      files.digest(case['image'])==case['image_sha256'], 'A118 case identity changed')
        files.require(files.digest(obs['path'])==obs['sha256'], 'A118 observation changed')
        cases.append(dict(case, a118_y=obs['path'], a118_y_sha256=obs['sha256']))
    ids_by_task = [{c['image_id'] for c in cases if c['task']==t} for t in TASKS]
    files.require(len(cases)==20 and len(ids_by_task[0])==10 and ids_by_task[0]==ids_by_task[1],
                  'A118 ten-image blur panel differs')
    entries = history.load_sources(host,'ffhq')
    plans = {t:next(e['plan'] for e in entries if e['case']['task']==t) for t in TASKS}
    checkpoint = plans[TASKS[0]]['checkpoint']
    files.require(all(p['checkpoint']==checkpoint for p in plans.values()) and
                  files.digest(checkpoint['path'])==checkpoint['sha256']==source['checkpoint_sha256'],
                  'Host and A118 checkpoints must match')
    dps_configs = ({t:{k:plans[t][k] for k in ('diffusion_config','dps_root')} for t in TASKS}
                   if host=='dps' else {})
    return dict(experiment=REVISION,host=host,parent=str(parent),parent_plan_sha256=files.digest(parent/'plan.json'),
                prior_preflight=failed_preflight(host),
                cases=cases,checkpoint=checkpoint,motion_kernel=str(kernel),motion_kernel_sha256=files.digest(kernel),
                dps_configs=dps_configs,arms=list(runtime.ARMS),cutoff=runtime.LAST_ACTIVE_T,
                relative_bound=runtime.RELATIVE_BOUND,nfe=1000 if host=='dps' else 100,
                sources={str(p):files.digest(p) for p in (Path(__file__),Path(runtime.__file__),CARD)},
                observations='A118 original y for DPS; same A118 noise realization on native circular PiGDM mean',
                timing='Same session full independent runs; model load, observation construction and metrics separate')


def make_operator(plan, case, device, kernel, counts=None):
    if plan['host']=='dps':
        from utils.shrink_multitask_runtime import make_operator as dps_operator
        return dps_operator(case,device,count=counts,kernel=kernel)
    from utils.pigdm_noisy_r1_runtime import Operator
    return Operator(case['task'],device,kernel=kernel if case['task']=='motion_deblur' else None,count=counts)


def observation(root, plan, case, device, kernel):
    if plan['host']=='dps':
        return case['a118_y'],case['a118_y_sha256']
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.shrink_multitask_runtime import make_operator as a118_operator

    meta = root/'observations'/case['key']/'observation.json'
    if not meta.exists():
        path = meta.with_name('y.pt')
        with precision.precision('LEGACY_IEEE'), torch.no_grad():
            gt = load_image_as_tensor(case['image']).to(device)
            original = torch.load(case['a118_y'],map_location=device,weights_only=True)['y']
            original_mean = a118_operator(case,device,kernel=kernel).H(gt)
            noise = original.double() - original_mean.double()
            files.require(.048 <= float(noise.std()) <= .052 and abs(float(noise.mean())) < .001,
                          'Recovered A118 measurement noise differs from sigma .05')
            mean = make_operator(plan,case,device,kernel).H(gt)
            y = (mean.double() + noise).to(mean.dtype)
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(y=y.cpu()),path)
        files.write(meta,dict(path=str(path),sha256=files.digest(path),source_sha256=case['a118_y_sha256']))
    record = files.read(meta)
    files.require(record['source_sha256']==case['a118_y_sha256'] and files.digest(record['path'])==record['sha256'],
                  'Frozen PiGDM observation changed')
    return record['path'],record['sha256']


def sample(plan,case,model,operator,y,arm,counters,trace,wavelet,tick):
    host = plan['host']
    rt = history.native(dict(case=case),host)
    refine = runtime.refinement(wavelet,arm,counters,trace)
    sampler = runtime.instrument(rt,host,arm,plan['cutoff'],refine)
    if host=='pigdm':
        return sampler('A124',model,operator,y,case,'BASE' if arm=='BASE' else 'SHRINK',counters,progress=tick)
    args = plan['dps_configs'][case['task']]
    return sampler(model,operator,y,case,'DPS' if arm=='BASE' else 'SHRINK',
                   args['diffusion_config'],args['dps_root'],counters,progress=tick)


def measure(root,plan,case,model,kernel,arm,y_path,y_sha,wavelet):
    import torch
    from modules.dataset_loader import tensor_to_image
    from utils.diag_noise_feasible_zaps import metrics
    from utils.dps_missing_tasks_runtime import tensor_digest

    folder = root/'cases'/case['key']/arm
    path = folder/'run.json'
    if path.exists():
        row = files.read(path)
        files.require(row['status']=='complete' and row['y_sha256']==y_sha and
                      all(files.digest(folder/n)==v for n,v in row['artifacts'].items()),
                      'Previous output failed or changed')
        return row
    if torch.cuda.mem_get_info()[0] < 2*1024**3:
        raise MemoryError('Need 2 GiB free before starting a new reconstruction')
    folder.mkdir(parents=True,exist_ok=False)
    row = dict(case=case,arm=arm,status='running',y_sha256=y_sha)
    files.write(path,row)
    trace, counts, entry_hash = [], {}, []
    host, nfe = plan['host'], plan['nfe']
    first_tail = 299 if host=='dps' else 290
    def capture_entry(_model,args):
        if counts['calls'][-1]==first_tail:
            entry_hash.append(tensor_digest(args[0]))
    try:
        scope = native.dps_reference_scope() if host=='dps' else nullcontext()
        with scope, precision.precision('LEGACY_IEEE'), clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            versions = [p._version for p in model.model.parameters()]
            def tick(step,cost):
                row.update(step=step,counts=dict(cost))
                files.write(path,row)
            def run():
                y = torch.load(y_path,map_location=model.device,weights_only=True)['y']
                op = make_operator(plan,case,model.device,kernel,counts)
                return sample(plan,case,model,op,y,arm,counts,trace,wavelet,tick)
            hook = model.model.register_forward_pre_hook(capture_entry)
            try:
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,run)
            finally:
                hook.remove()
            final_rng = shared.rng()
        files.require(len(entry_hash)==1 and counts['calls']==detail['grid'], 'Entry or native grid differs')
        files.require(counts['forwards']==counts['forward_attempts']==counts['input_vjp_attempts']==nfe and
                      counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0,
                      'Native denoiser/derivative budget differs')
        expected_tail = [t for t in detail['grid'] if t <= plan['cutoff']]
        scalar_tail = [r['t'] for r in detail['trace'] if 'weight' in r]
        expected_steps = 300 if host=='dps' else 30
        files.require(len(expected_tail)==expected_steps and
                      scalar_tail==(expected_tail if arm!='BASE' else []) and
                      [r['t'] for r in trace]==(expected_tail if arm.endswith('DIAGONAL') else []),
                      'Feedback coverage differs')
        files.require(all(p.grad is None and p._version==v for p,v in zip(model.model.parameters(),versions)),
                      'Network weights or gradients changed')
        pairing = history.fingerprint(detail,final_rng,counts,host)
        pairing['feedback_entry_sha256'] = entry_hash[0]
        for old in folder.parent.glob('*/run.json'):
            saved = files.read(old)
            if saved['status']=='complete':
                files.require(saved['pairing']==pairing and saved['y_sha256']==y_sha, 'Paired trajectories differ before feedback')
        start = time.perf_counter()
        quality = metrics(out,shared.load_image(case['image']))
        metric_seconds = time.perf_counter()-start
        torch.save(out,folder/'output.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',trace)
        row.update(status='complete',counts=counts,metrics=quality,full_seconds=seconds,pairing=pairing,
                   metric_seconds=metric_seconds,artifacts={name:files.digest(folder/name) for name in
                       ('output.pt','output.png','trace.json')})
        files.write(path,row)
        print('A124 COMPLETE', host, case['key'], arm, quality, round(seconds,2), flush=True)
        return row
    except BaseException as error:
        row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=counts)
        files.write(path,row)
        files.write(folder/'partial_trace.json',trace)
        raise


def summarize(root, plan, status):
    import numpy as np
    rows = [files.read(p) for p in sorted((root/'cases').glob('*/*/run.json'))]
    groups = []
    for task in TASKS:
        cells = {arm:{r['case']['image_id']:r for r in rows if r['case']['task']==task and
                      r['arm']==arm and r['status']=='complete'} for arm in runtime.ARMS}
        means = {arm:dict(images=len(c),quality={k:statistics.mean(r['metrics'][k] for r in c.values()) for k in METRICS},
                          seconds=statistics.mean(r['full_seconds'] for r in c.values())) for arm,c in cells.items() if c}
        comparisons = []
        for cand, ref in (('TAIL_SCALAR','BASE'),('A118_DIAGONAL','BASE'),('BOUNDED_DIAGONAL','BASE'),
                          ('A118_DIAGONAL','TAIL_SCALAR'),('BOUNDED_DIAGONAL','TAIL_SCALAR'),
                          ('BOUNDED_DIAGONAL','A118_DIAGONAL')):
            a, b = cells[cand],cells[ref]; ids = sorted(a.keys() & b.keys())
            if not ids:
                continue
            paired = {i:{k:a[i]['metrics'][k]-b[i]['metrics'][k] for k in METRICS} for i in ids}
            delta = {k:statistics.mean(v[k] for v in paired.values()) for k in METRICS}
            wins = sum(v['psnr']>0 for v in paired.values())
            ratio = sum(a[i]['full_seconds'] for i in ids)/sum(b[i]['full_seconds'] for i in ids)
            ci = None
            if len(ids)==10:
                samples = np.random.default_rng(20261120).integers(0,10,size=(20000,10))
                ci = {k:np.quantile(np.array([paired[i][k] for i in ids])[samples].mean(1), [.025,.975]).tolist()
                      for k in METRICS}
            comparisons.append(dict(candidate=cand,reference=ref,images=len(ids),delta=delta,psnr_wins=wins,
                per_image=paired,paired_bootstrap_95=ci,clock_ratio=ratio,within_10_percent=ratio<=1.1,
                meets_quality_criterion=len(ids)==10 and wins>=7 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
        groups.append(dict(task=task,arms=means,comparisons=comparisons))
    done = sum(r['status']=='complete' for r in rows)
    keys = ('forwards','forward_attempts','input_vjp_attempts','input_jvp_attempts','refinement_H','refinement_Ht',
            'module_dwt_analysis','module_dwt_synthesis')
    result = dict(experiment=REVISION,status=status,host=plan['host'],planned_runs=80,completed_runs=done,
                  per_arm_nfe=plan['nfe'],groups=groups,actual_cost={k:sum(r.get('counts',{}).get(k,0) for r in rows) for k in keys},
                  cost_is_lower_bound=any(r['status']!='complete' for r in rows))
    files.write(root/'summary.json',result)
    return result


def execute(host, root, parent):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters

    plan = load_plan(host,parent)
    plan_path = root/'plan.json'
    if plan_path.exists():
        files.require(files.read(plan_path)==plan,'Frozen A124 plan changed')
    else:
        files.write(plan_path,plan)
        shutil.copyfile(CARD,root/'experiment_card.md')
    files.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose one allocated CUDA device')
    if torch.cuda.mem_get_info()[0] < 6*1024**3:
        raise MemoryError('Need 6 GiB free before model load')
    if shutil.disk_usage(root).free < 1024**3:
        raise OSError('Need 1 GiB output space on the experiment disk')
    subprocess.run([sys.executable,'-m','unittest','-v','utils.test_a118_blur_transfer'],cwd=PROJECTS,check=True)
    started = time.perf_counter()
    model = load_diffusion_model('ffhq','cuda:0')
    model.model.load_state_dict(torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval()
    wavelet = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
    torch.cuda.synchronize()
    files.write(root/'load.json',dict(model_load_seconds=time.perf_counter()-started))
    kernel = np.load(plan['motion_kernel'],allow_pickle=False)
    with frozen_network(model.model), input_vjp_parameters(model.model):
        for i,case in enumerate(plan['cases']):
            y_path,y_sha = observation(root,plan,case,model.device,kernel)
            # Rotate arm order to avoid confounding timing with position in each case.
            arms = runtime.ARMS[i%4:]+runtime.ARMS[:i%4]
            for arm in arms:
                measure(root,plan,case,model,kernel,arm,y_path,y_sha,wavelet)
            result = summarize(root,plan,'running')
            print('A124 PROGRESS',host,result['completed_runs'],'/80',flush=True)
    result = summarize(root,plan,'complete')
    files.require(result['completed_runs']==80,'Incomplete A124 panel')
    return result


def archive(host,root):
    from utils import archive_refinement_confirmation as exporter
    result = files.read(root/'summary.json')
    files.require(result['status']=='complete' and result['completed_runs']==80,'Require complete A124 result')
    branch = f'results/a124r1-a118-blur-transfer-{host}'
    # Keep Git staging and the export copy on the result disk.
    with tempfile.TemporaryDirectory(prefix='A124_export_',dir=root.parent) as stage:
        repo = Path(stage)/'repo'
        original = PROJECTS.parent
        subprocess.run(['git','clone','--shared','--no-checkout',str(original),str(repo)],check=True)
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        for key in ('user.name','user.email'):
            git('config',key,subprocess.check_output(['git','-C',str(original),'config','--get',key],text=True).strip())
        git('remote','set-url','origin',subprocess.check_output(['git','-C',str(original),'remote','get-url','origin'],text=True).strip())
        ref = f'refs/heads/{branch}'
        if git('ls-remote','--heads','origin',ref):
            git('fetch','origin',f'{ref}:{ref}')
        (repo/'projects').mkdir()
        exporter.BRANCH = branch
        _,commit = exporter.export_and_commit({host:root},projects=repo/'projects')
        git('push','origin',ref)
    files.write(root/'upload.json',dict(status='complete',branch=branch,commit=commit))
    print('RESULTS_PUSH_PASS',commit,branch,flush=True)


def worker(host,root,parent):
    with exclusive(root/'worker.lock'):
        state = dict(status='running',pid=os.getpid())
        files.write(root/'session.json',state)
        try:
            result = execute(host,root,parent)
            state['status'] = 'complete'
            print('A124 SUMMARY',result,flush=True)
        except BaseException as error:
            state.update(status='resource_limited' if isinstance(error,MemoryError) else 'failed',error=repr(error))
            if (root/'plan.json').exists():
                summarize(root,files.read(root/'plan.json'),state['status'])
            raise
        finally:
            files.write(root/'session.json',state)


def follow(host,root,parent,gpu):
    with exclusive(root/'client.lock'):
        session = root/'session.json'; launch = root/'launcher.json'; log = root/'worker.log'
        old = files.read(session) if session.exists() else {}
        if old.get('status')=='complete':
            print('A124 SUMMARY',files.read(root/'summary.json'),flush=True)
            return
        pid = files.read(launch)['pid'] if launch.exists() else None
        offset = 0
        if not pid or not running(pid):
            files.require(old.get('status') not in ('failed','running'),'Prior attempt retained; see --status')
            offset = log.stat().st_size if log.exists() else 0
            cmd = [sys.executable,'-u','-m','utils.diag_a118_blur_transfer',host,'--parent',str(parent),'--worker']
            with log.open('ab') as stream:
                proc = subprocess.Popen(cmd,cwd=PROJECTS,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=stream,
                                        stderr=subprocess.STDOUT,start_new_session=True)
            pid = proc.pid
            files.write(launch,dict(pid=pid,gpu=gpu))
    while True:
        active = running(pid)
        if log.exists():
            with log.open() as stream:
                stream.seek(offset); data=stream.read(); offset=stream.tell()
            if data:
                print(data,end='',flush=True)
        if not active:
            break
        time.sleep(5)
    files.require(session.exists() and files.read(session)['status']=='complete',f'A124 stopped: {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','pigdm'))
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--gpu',default='0')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--status',action='store_true')
    modes.add_argument('--archive',action='store_true')
    modes.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = OUTPUTS/f'{REVISION}_{args.host}_ffhq'
    if args.status:
        state = files.read(root/'session.json') if (root/'session.json').exists() else dict(status='not_started')
        print(state)
        if (root/'summary.json').exists():
            print('A124 SUMMARY',files.read(root/'summary.json'))
        for path in (root/'cases').glob('*/*/run.json'):
            row = files.read(path)
            if row['status']!='complete':
                print('A124 CASE',row['case']['key'],row['arm'],row['status'],row.get('step',0),row.get('error',''))
        return
    root.mkdir(parents=True,exist_ok=True)
    if args.archive:
        with exclusive(root/'worker.lock'):
            archive(args.host,root)
    elif args.worker:
        worker(args.host,root,args.parent.resolve())
    else:
        follow(args.host,root,args.parent.resolve(),args.gpu)


if __name__=='__main__':
    main()
