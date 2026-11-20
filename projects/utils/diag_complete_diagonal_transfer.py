"""Complete DPS/PiGDM four-task scalar and diagonal transfer tables."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import native_ray_transfer_runtime as shared
from utils import historical_ffhq_runtime as precision
from utils import transfer_diagonal_runtime as control
from utils.a54_session import exclusive

PROJECTS = Path(__file__).resolve().parents[1]
TASKS = ('super_resolution', 'gaussian_deblur', 'motion_deblur', 'inpainting')
ARMS = ('BASE', 'SHRINK', 'DIAGONAL')
BASE = Path('/data0/lzy/ZAPS-results')
ROOT = BASE / 'diag_complete_diagonal_transfer'
SOURCES = dict(
    dps_ffhq=BASE / 'diag_dps_multiplier_transfer/A75_20261031',
    dps_imagenet=BASE / 'diag_dps_imagenet_transfer/A77_20261101',
    pigdm_ffhq=BASE / 'diag_pigdm_noisy_transfer/A85R1_ffhq_20261104',
    pigdm_imagenet=BASE / 'diag_pigdm_noisy_transfer/A85R1_imagenet_20261104')


def directory(host, dataset):
    return ROOT / f'{host}_{dataset}_20261120'


def load_sources(host, dataset):
    if host == 'pigdm':
        from utils import diag_pigdm_noisy_r1 as driver
        roots = [(SOURCES[f'{host}_{dataset}'], driver)]
    else:
        from utils import diag_dps_multiplier_transfer as ffhq
        from utils import diag_dps_imagenet_transfer as imagenet
        from utils import diag_dps_missing_tasks as missing
        roots = [(SOURCES[f'{host}_{dataset}'], ffhq if dataset == 'ffhq' else imagenet),
                 (BASE / f'diag_dps_missing_tasks/A88_{dataset}_20261104', missing)]
    entries = []
    for root, driver in roots:
        plan = driver.check(root)
        for case in plan['cases']:
            entries.append(dict(root=root, driver=driver, plan=plan, case=case))
    for task in TASKS:
        files.require(sum(row['case']['task'] == task for row in entries) == 3,
                      f'Need three completed source images for {task}')
    return entries


def existing(entry, arm, host):
    name = 'DPS' if host == 'dps' and arm == 'BASE' else arm
    row = entry['driver'].completed(entry['root'], entry['case'], name)
    files.require(row is not None, 'Missing completed original result')
    return row, entry['root'] / 'cases' / entry['case']['key'] / name


def observation(entry, host):
    case, plan, root = entry['case'], entry['plan'], entry['root']
    if host == 'pigdm':
        meta = root / 'observations' / case['key'] / 'observation.json'
        obs = files.read(meta)
        files.require(obs['plan_sha256'] == files.digest(root / 'plan.json'), 'Observation plan differs')
        for name, sha in obs['artifacts'].items():
            files.require(files.digest(meta.parent / name) == sha, 'Original observation changed')
        return obs
    path = Path(plan['observations'][case['key']])
    if str(path) in plan['inputs']:
        files.require(files.digest(path) == plan['inputs'][str(path)], 'DPS original y differs')
    else:
        meta = files.read(path.parent / 'observation.json')
        files.require(meta['sha256'] == files.digest(path), 'DPS observation changed')
        files.require(meta['plan_sha256'] == files.digest(root / 'plan.json'), 'DPS observation plan differs')
    return dict(path=str(path))


def operator(entry, obs, host, device, count):
    plan, case = entry['plan'], entry['case']
    if host == 'pigdm':
        p = dict(plan, root=str(entry['root']))
        return entry['driver'].fresh_operator(p, case, device, obs, count)
    from utils import diag_shrink_multitask
    return diag_shrink_multitask.make_operator(
        Path(plan.get('operator_root', plan['parent'])), case, device, count)


def native(entry, host):
    if host == 'pigdm':
        from utils import pigdm_noisy_r1_runtime as rt
    elif entry['case']['task'] in ('gaussian_deblur', 'inpainting'):
        from utils import dps_missing_tasks_runtime as rt
    else:
        from utils import dps_multiplier_transfer_runtime as rt
    return rt


def fingerprint(detail, final, count, host):
    from utils.diag_pigdm_noisy_r1 import fingerprint as digest
    if host == 'pigdm':
        before = detail['entry_rng']
        initial = digest(detail['initial'])
    elif 'pairing' in detail:
        return dict(**detail['pairing'], calls=count['calls'])
    else:
        before = dict(cpu=detail['cpu_rng'], cuda=detail['cuda_rng'])
        initial = digest(detail['initial'])
    return dict(initial=initial, entry_cpu=digest(before['cpu']), entry_cuda=[digest(v) for v in before['cuda']],
                final_cpu=digest(final['cpu']), final_cuda=[digest(v) for v in final['cuda']],
                grid=detail['grid'], calls=count['calls'])


def reuse(root, entry, host, arm):
    if arm == 'BASE':
        row, folder = existing(entry, arm, host)
    else:
        parent = BASE / 'diag_pigdm_noisy_transfer/A119_ffhq_20261120'
        plan = files.read(parent / 'plan.json')
        files.require(plan['parent_plan'] == files.digest(entry['root'] / 'plan.json'), 'A119 source plan differs')
        folder = parent / 'cases' / entry['case']['key'] / arm
        row = files.read(folder / 'run.json')
        files.require(row['status'] == 'complete' and row['case'] == entry['case'], 'A119 reuse identity differs')
    target = root / 'cases' / entry['case']['key'] / arm / 'run.json'
    files.require(not target.exists(), 'Reuse record exists')
    files.write(target, {**row, 'arm': arm, 'reused': True, 'source_record': str(folder / 'run.json'),
                         'source_sha256': files.digest(folder / 'run.json')})


def measure(root, entry, obs, host, model, wavelet, arm):
    import torch
    from modules.dataset_loader import tensor_to_image
    from utils.diag_noise_feasible_zaps import metrics
    case, plan = entry['case'], entry['plan']
    folder = root / 'cases' / case['key'] / arm
    record_file = folder / 'run.json'
    if record_file.exists():
        files.require(files.read(record_file)['status'] == 'complete', 'Prior unfinished attempt retained')
        return
    free, _ = torch.cuda.mem_get_info()
    needed = max(0, torch.cuda.max_memory_reserved() - torch.cuda.memory_reserved()) + 1024**3
    if free < needed:
        raise MemoryError('Insufficient current memory margin before attempt')
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(case=case, arm=arm, status='running', counts={})
    files.write(record_file, record)
    count = {}
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        versions = [p._version for p in model.model.parameters()]
        with precision.precision('LEGACY_IEEE'), clock.counters(model.model) as count:
            count.update(module_dwt_analysis=0, module_dwt_synthesis=0,
                         module_adjoint_actions=0, refinement_H=0, refinement_Ht=0)
            def tick(step, measured):
                record['counts'] = dict(measured)
                files.write(record_file, record)
                print('TRANSFER PROGRESS:', host, case['key'], arm, step, flush=True)
            def operation():
                y = torch.load(obs['path'], map_location='cpu', weights_only=True)['y'].to(model.device)
                op = operator(entry, obs, host, model.device, count)
                rt = native(entry, host)
                sampler = rt.run
                if arm == 'DIAGONAL':
                    sampler = control.instrument(rt, control.feedback(wavelet, count, []), host)
                if host == 'pigdm':
                    return sampler('A121', model, op, y, case, 'BASE' if arm == 'DIAGONAL' else 'SHRINK',
                                   count, progress=tick)
                return sampler(model, op, y, case, 'SHRINK', plan['diffusion_config'],
                               plan['dps_root'], count, progress=tick)
            (out, detail), seconds = clock.clocks(torch.cuda.synchronize, operation)
            final = shared.rng()
        nfe = 100 if host == 'pigdm' else 1000
        files.require(count['forwards'] == count['forward_attempts'] == count['input_vjp_attempts'] == nfe,
                      'Network evaluation budget differs')
        files.require(count['optimizer_backward_attempts'] == 0, 'Unexpected optimizer backward')
        files.require(all(p.grad is None and p._version == version for p, version in
                          zip(model.model.parameters(), versions)), 'Network parameters changed')
        pairing = fingerprint(detail, final, count, host)
        replay_error = None
        if arm == 'SHRINK':
            _, original = existing(entry, 'SHRINK', host)
            expected = torch.load(original / 'output.pt', map_location='cpu', weights_only=True)
            replay_error = float((out.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))
            files.require(replay_error <= 1e-4, f'Original scalar replay differs: {replay_error:.3g}')
        other = folder.parent / ('SHRINK' if arm == 'DIAGONAL' else 'DIAGONAL') / 'run.json'
        if other.exists():
            files.require(files.read(other)['pairing'] == pairing, 'Initial/final RNG or grid pairing failed')
        start = time.perf_counter()
        quality = metrics(out, shared.load_image(case['image']))
        metric_seconds = time.perf_counter() - start
        start = time.perf_counter()
        torch.save(out, folder / 'output.pt')
        tensor_to_image(out.squeeze(0), denormalize=True).save(folder / 'output.png')
        files.write(folder / 'trace.json', detail['trace'])
        record.update(status='complete', counts=count, metrics=quality, full_seconds=seconds,
                      pairing=pairing, original_scalar_relative_l2=replay_error, metric_seconds=metric_seconds,
                      artifacts={name: files.digest(folder / name) for name in ('output.pt', 'output.png', 'trace.json')},
                      artifact_seconds=time.perf_counter()-start, peak_reserved_bytes=torch.cuda.max_memory_reserved())
        files.write(record_file, record)
        print('TRANSFER COMPLETE:', case['key'], arm, quality, round(seconds, 2), flush=True)
    except BaseException as error:
        record.update(status='failed', counts=count, error=repr(error), traceback=traceback.format_exc())
        files.write(record_file, record)
        raise


def summarize(root):
    plan = files.read(root / 'plan.json')
    rows = [files.read(f) for f in (root / 'cases').glob('*/*/run.json')]
    groups, comparisons = [], []
    for task in TASKS:
        selected = {arm: {r['case']['key']: r for r in rows if r['case']['task'] == task and
                          r['arm'] == arm and r['status'] == 'complete'} for arm in ARMS}
        for arm, cases in selected.items():
            if cases:
                groups.append(dict(task=task, arm=arm, images=len(cases),
                    quality={k: statistics.mean(r['metrics'][k] for r in cases.values()) for k in ('psnr','ssim','lpips')},
                    seconds=statistics.mean(r['full_seconds'] for r in cases.values()),
                    nfe=plan['nfe'], input_vjp=plan['nfe'], reused=all(r.get('reused', False) for r in cases.values())))
        for candidate, reference in (('SHRINK','BASE'), ('DIAGONAL','SHRINK')):
            keys = selected[candidate].keys() & selected[reference].keys()
            if keys:
                delta = {k: statistics.mean(selected[candidate][key]['metrics'][k]-selected[reference][key]['metrics'][k]
                                            for key in keys) for k in ('psnr','ssim','lpips')}
                wins = sum(selected[candidate][key]['metrics']['psnr'] > selected[reference][key]['metrics']['psnr'] for key in keys)
                comparisons.append(dict(task=task, candidate=candidate, reference=reference, images=len(keys),
                    delta=delta, psnr_wins=wins, positive=len(keys)==3 and wins>=2 and delta['psnr']>=.1 and
                    delta['ssim']>=0 and delta['lpips']<=0))
    result = dict(host=plan['host'], dataset=plan['dataset'], completed_runs=sum(r['status']=='complete' for r in rows),
        planned_runs=36, groups=groups, comparisons=comparisons,
        actual_new_cost={key: sum((r.get('counts') or {}).get(key,0) for r in rows if not r.get('reused'))
                         for key in ('forward_attempts','forwards','input_vjp_attempts','optimizer_backward_attempts',
                                     'refinement_H','refinement_Ht','module_dwt_analysis','module_dwt_synthesis')},
        timing='Same-session scalar/diagonal full loops. BASE clocks and A119 clocks are inherited, not newly timed.',
        separate_model_gate=files.read(root/'load.json') if (root/'load.json').exists() else {},
        failures=[r.get('error') for r in rows if r['status']=='failed'])
    files.write(root / 'summary.json', result)
    print('TRANSFER SUMMARY:', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def execute(args, root):
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    entries = load_sources(args.host, args.dataset)
    manifests = []
    for entry in entries:
        existing(entry, 'BASE', args.host)
        existing(entry, 'SHRINK', args.host)
        entry['observation'] = observation(entry, args.host)
        manifests.append(dict(case=entry['case'], source=str(entry['root']),
                              plan_sha256=files.digest(entry['root']/'plan.json'),
                              observation_sha256=files.digest(entry['observation']['path'])))
    plan = dict(host=args.host, dataset=args.dataset, nfe=1000 if args.host=='dps' else 100, sources=manifests,
                files={str(p):files.digest(p) for p in (Path(__file__),Path(control.__file__))})
    if (root/'plan.json').exists():
        files.require(files.read(root/'plan.json') == plan, 'Frozen plan differs')
    else:
        files.write(root/'plan.json', plan)
    files.require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'Allocate one visible CUDA device')
    free, _ = torch.cuda.mem_get_info()
    if free < (10 if args.dataset=='imagenet' else 6)*1024**3:
        raise MemoryError('Insufficient free memory before load')
    checkpoint = entries[0]['plan']['checkpoint']
    for entry in entries:
        files.require(entry['plan']['checkpoint']==checkpoint, 'Source models differ')
        files.require(files.digest(entry['case']['image'])==entry['plan']['inputs'][entry['case']['image']], 'Input image changed')
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'], 'Checkpoint differs')
    subprocess.run([sys.executable,'-m','unittest','-v','utils.test_transfer_diagonal_runtime'], cwd=PROJECTS, check=True)
    load_start = time.perf_counter()
    model = load_diffusion_model(args.dataset, 'cuda:0')
    model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval()
    torch.cuda.synchronize()
    wavelet = SimpleNamespace(dwt=OrthogonalDWT2D(wave='db4',level=3).to(model.device))
    load = dict(model_load_seconds=time.perf_counter()-load_start)
    files.write(root/'load.json',load)
    with frozen_network(model.model), input_vjp_parameters(model.model):
        started = time.perf_counter()
        with precision.precision('LEGACY_IEEE'), clock.counters(model.model) as probe:
            x = torch.zeros(1,3,256,256,device=model.device,requires_grad=True)
            value = model.model(x,torch.tensor([999],device=model.device))
            files.require(bool(torch.isfinite(value).all()), 'Actual checkpoint forward gate failed')
            probe['input_vjp_attempts'] += 1
            derivative = torch.autograd.grad(value[:,:3].square().mean(),x)[0]
            files.require(bool(torch.isfinite(derivative).all()), 'Actual checkpoint input VJP gate failed')
            del derivative,value,x
        torch.cuda.synchronize()
        load.update(probe_counts=probe,probe_seconds=time.perf_counter()-started,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
        files.write(root/'load.json',load)
        torch.cuda.empty_cache()
        for index, entry in enumerate(entries):
            key = entry['case']['key']
            if not (root/'cases'/key/'BASE'/'run.json').exists():
                reuse(root, entry, args.host, 'BASE')
            if args.host=='pigdm' and args.dataset=='ffhq' and entry['case']['task']=='motion_deblur':
                for arm in ('SHRINK','DIAGONAL'):
                    if not (root/'cases'/key/arm/'run.json').exists():
                        reuse(root, entry, args.host, arm)
                continue
            for arm in (('SHRINK','DIAGONAL') if index%2==0 else ('DIAGONAL','SHRINK')):
                measure(root, entry, entry['observation'], args.host, model, wavelet, arm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host', choices=('dps','pigdm'))
    p.add_argument('dataset', choices=('ffhq','imagenet'))
    p.add_argument('--status', action='store_true')
    p.add_argument('--archive', action='store_true')
    args = p.parse_args()
    root = directory(args.host, args.dataset)
    if args.status:
        print(files.read(root/'session.json') if (root/'session.json').exists() else dict(status='not_started'))
        if (root/'plan.json').exists(): summarize(root)
        return
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'worker.lock'):
        previous = files.read(root/'session.json') if (root/'session.json').exists() else {}
        files.require(previous.get('status') not in ('failed','running'), 'Prior failed/unfinished session retained')
        state = dict(status='running',pid=os.getpid())
        files.write(root/'session.json',state)
        try:
            if previous.get('status')!='complete': execute(args,root)
            result = summarize(root)
            files.require(result['completed_runs']==36, 'Incomplete transfer table')
            state['status']='complete'
        except BaseException as error:
            state.update(status='resource_limited' if isinstance(error, MemoryError) else 'failed',
                         error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            files.write(root/'session.json',state)
            if (root/'plan.json').exists(): summarize(root)
        if args.archive:
            from utils import archive_refinement_confirmation as exporter
            exporter.BRANCH=f'results/{args.host}-diagonal-four-tasks-{args.dataset}-20261120'
            bundle, commit = exporter.export_and_commit({args.host+'_'+args.dataset:root})
            subprocess.run(['git','push','origin',exporter.BRANCH],cwd=PROJECTS.parent,check=True)
            files.write(root/'upload.json',dict(status='complete',commit=commit,branch=exporter.BRANCH))
            print('RESULTS_PUSH_PASS:',commit,exporter.BRANCH,flush=True)


if __name__=='__main__':
    main()
