"""A35: full improved-path geometry and paired final guidance responses; 1296 diagnostic NFE."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_continuous_response_feedback as parent
from utils import continuous_response_feedback as feedback
from utils import trajectory_state_geometry as geometry

source, P = parent.source, parent.P
ROOT = '/data0/lzy/ZAPS-results/diag_trajectory_geometry_value/A35_20261021'
CARD = PROJECTS/'experiment_cards/A35_trajectory_geometry_value_20261021.md'


def prepare(args):
    root, old = Path(args.root).resolve(), Path(args.a33_dir).resolve()
    if root.is_relative_to(old) or old.is_relative_to(root):
        raise ValueError('New output overlaps A33')
    if root.exists():
        p = source.check_plan(root)
        if p['source_A33'] != str(old):raise ValueError('Existing parent differs')
        print('A35 frozen plan reused; complete cases will not rerun.', flush=True)
        return
    previous = source.check_plan(old)
    summary = source.load_signed(old/'summary.json')
    if previous['experiment'] != 'A33_continuous_response_feedback' or summary['completed_cases'] != 12:
        raise ValueError('Need twelve complete A33 cases')
    identities = dict(previous['identity_files'])
    baseline = {}
    profiles = parent.read_profiles(old, previous)
    for spec in previous['cases']:
        P.verify_case_sources(previous, spec)
        for arm in ('I_cont2', 'STAGE_cont2'):
            signature = profiles['policy_sha256']['I_cont2'] if arm == 'STAGE_cont2' else None
            item = parent.task_done(old, previous, spec, arm, signature)
            if item is None:raise ValueError('Incomplete A33 reference')
            path, _ = item
            if arm == 'I_cont2':baseline[spec['case']['key']] = str(path)
            for f in (path, path.with_suffix('.sha256'), path.parent/'output.pt'):
                identities[str(f)] = source.digest(f)
    for name in ('plan.json', 'plan.sha256', 'summary.json', 'summary.sha256'):
        identities[str(old/name)] = source.digest(old/name)
    workers = dict(previous['worker_identity_files'])
    for f in (Path(__file__), Path(geometry.__file__), CARD, PROJECTS/'run_a35_20261021.sh',
              PROJECTS/'utils/test_trajectory_state_geometry.py'):
        workers[str(f)] = source.digest(f)
    if any(source.digest(path) != sha for path, sha in workers.items()):
        raise ValueError('Frozen worker changed')
    plan = dict(previous)
    plan.update(experiment='A35_trajectory_geometry_value', source_A33=str(old), baseline_records=baseline,
                identity_files=identities, worker_identity_files=workers,
                geometry_points=[list(v) for v in geometry.POINTS], pulse_factors=list(geometry.FACTORS),
                geometry_window_u=geometry.WINDOW_U, planned_method_nfe=0, planned_diagnostic_nfe=1296,
                calls_per_case=geometry.CALLS, scientific_nfe_per_arm=300,
                scope='Six reused development images, two seeds; causal geometry on improved I_cont2 paths. '
                      'Observed clean-path bending, not true stochastic-process or ODE curvature. '
                      'Conditional-mean steps exclude new draws but retain past noise dependence. '
                      'One fixed-point guidance pulse, then actual feedback suffix; no new controller or schedule.')
    root.parent.mkdir(parents=True, exist_ok=True)
    P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir(); source.signed(root/'plan.json', plan)
    (root/'experiment_card.md').write_text(CARD.read_text())
    print('A35 PREPARE PASS: 12 cases x108 =1296 diagnostic NFE; all scientific arms=300. '
          'Full geometry capture on I_cont2, ten final pulse outcomes/case.', flush=True)


def completed(root, plan, spec):
    complete = []
    for path in sorted((root/spec['case']['key']).glob('attempt_*/run.json')):
        r = source.read(path)
        if r['status'] != 'complete':continue
        r = source.load_signed(path)
        expected = [(k, t) for k, t in enumerate(reversed(plan['grid']))]
        if (r['plan_sha256'] != source.digest(root/'plan.json') or r['case'] != spec['case']
                or r['method_nfe'] != 0 or r['diagnostic_nfe'] != geometry.CALLS or len(r['rows']) != 30
                or [(v['step'], v['t']) for v in r['rows']] != expected
                or not r['gates'] or not all(v['passed'] for v in r['gates'].values())
                or source.digest(path.parent/'outputs.pt') != r['outputs_sha256']
                or source.digest(path.parent/'trajectory.pt') != r['trajectory_sha256']):
            raise ValueError('Completed A35 differs: '+str(path))
        expected_calls = [('baseline', k, t) for k, t in expected]
        for k, t in geometry.POINTS:
            for factor in geometry.FACTORS:
                expected_calls.extend((f'T{t}_M{factor:g}', j, tj) for j, tj in expected[k+1:])
        if ([(c['kind'], c['step'], c['t']) for c in r['calls']] != expected_calls
                or set(r['metrics']) != {'BASE'} | {f'T{t}_M{f:g}' for _, t in geometry.POINTS for f in geometry.FACTORS}):
            raise ValueError('Completed A35 arm/call schedule differs')
        complete.append((path, r))
    if len(complete) > 1:raise ValueError('Multiple complete cases; no output selection')
    return complete[0] if complete else None


def run_case(root, plan, spec, model, device):
    import torch
    from unittest.mock import patch
    from modules.main_single import load_image_as_tensor
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics

    if completed(root, plan, spec):return
    folder = root/spec['case']['key'];folder.mkdir(exist_ok=True)
    attempt = folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir()
    path = attempt/'run.json'; started = time.perf_counter(); context = {}
    record = dict(status='running', case=spec['case'], plan_sha256=source.digest(root/'plan.json'),
                  diagnostic_nfe=0, method_nfe=0, calls=[], rows=[], gates={})
    def save():
        record['seconds'] = time.perf_counter()-started; source.WRITE(path, record)
    def count(_module, inputs):
        t = int(inputs[1].flatten()[0]);record['diagnostic_nfe'] += 1
        record['calls'].append(dict(**context, t=t));save()
        if t != context['expected_t'] or record['diagnostic_nfe'] > geometry.CALLS:
            raise RuntimeError('Budget/time mismatch')
    save();hook = model.model.register_forward_pre_hook(count)
    try:
        P.verify_case_sources(plan, spec)
        reference_path = Path(plan['baseline_records'][spec['case']['key']])
        if source.digest(reference_path) != plan['identity_files'][str(reference_path)]:
            raise ValueError('A33 baseline record changed')
        reference = source.load_signed(reference_path)
        if source.digest(reference_path.parent/'output.pt') != reference['output_sha256']:
            raise ValueError('A33 baseline tensor changed')
        target = torch.load(reference_path.parent/'output.pt', map_location='cpu', weights_only=True)
        z, y, state, old_cache, positions = parent.old.setup(plan, spec, model, device)
        original = torch.load(Path(spec['source'])/'last_opt.pt', map_location='cpu', weights_only=True)
        native = source.read(Path(spec['source'])/'run.json')
        noises, noise_gate = P.coupled.replay_noises(state, old_cache, dict(rows=native['rows']), z, y, device, original)
        history = geometry.History();cache = {n: [] for n in ('latent_inputs', 'clean', 'guidance', 'prior')}
        outputs = {}; scalar_error = 0.; external = rng_state(device)
        def advance(x, f, k, gamma):
            t = positions.nominal[k]
            base = P.terminal.terminal_output(z, x, f) if k == 29 else P.coupled.posterior(
                x, f['clean'], t, positions.nominal[k+1], z.dm.alphas_cumprod, noises[k])+f['q']
            result = base if gamma == 1. else base+(gamma-1.)*f['q']
            if not torch.isfinite(result).all():raise RuntimeError('Nonfinite endpoint')
            return result
        with torch.no_grad(), model_rng_guard(model.model, device), patch(
                'torch.randn_like', side_effect=RuntimeError('Unexpected noise draw')):
            x = state['init_noise'].to(device)
            for k, t in enumerate(positions.nominal):
                context.update(kind='baseline', step=k, expected_t=t)
                f = source.fields.field(z, y, x, t, positions)
                readout = feedback.sensor(z, y, f['clean'], f['q'], t, 2.)
                gamma = readout['gamma']
                old_row = reference['rows'][k]
                for name, value in (('gamma', gamma), ('R', f['R']), ('qRMS', readout['qRMS']), ('clip', f['clip_fraction'])):
                    scalar_error = max(scalar_error, abs(value-old_row[name])/max(abs(old_row[name]), 1e-12))
                prior = f['clean']-x if k == 29 else P.coupled.posterior(
                    x, f['clean'], t, positions.nominal[k+1], z.dm.alphas_cumprod, torch.zeros_like(x))-x
                local = dict(latent_inputs=x[0].cpu(), clean=f['clean'][0].cpu(),
                             guidance=f['q'][0].cpu(), prior=prior[0].cpu())
                for name, value in local.items():cache[name].append(value)
                u = float(positions.u[t])
                observation = history.append(u, local['clean'].numpy(), local['guidance'].numpy(),
                                             local['prior'].numpy(), gamma, readout, f['clip_fraction'])
                record['rows'].append(dict(step=k, t=t, u=u, gamma=gamma, R=f['R'], state=observation))
                x = advance(x, f, k, gamma)
            outputs['BASE'] = x.cpu()
            parity = relative_error(outputs['BASE'], target)
            if max(parity, scalar_error) > 1e-6 or not math.isfinite(parity+scalar_error):
                raise RuntimeError('Improved I_cont2 baseline parity failed')
            cache = {n: torch.stack(v) for n, v in cache.items()}
            for k, t in geometry.POINTS:
                if positions.nominal[k] != t:raise ValueError('Diagnostic anchor differs')
                f = dict(clean=cache['clean'][k:k+1].to(device), q=cache['guidance'][k:k+1].to(device))
                for factor in geometry.FACTORS:
                    arm = f'T{t}_M{factor:g}';before = record['diagnostic_nfe']
                    x = advance(cache['latent_inputs'][k:k+1].to(device), f, k, record['rows'][k]['gamma']*factor)
                    for j in range(k+1, 30):
                        tj = positions.nominal[j]
                        context.update(kind=arm, step=j, expected_t=tj)
                        field = source.fields.field(z, y, x, tj, positions)
                        gamma = feedback.sensor(z, y, field['clean'], field['q'], tj, 2.)['gamma']
                        x = advance(x, field, j, gamma)
                    if record['diagnostic_nfe']-before != 29-k:raise RuntimeError('Suffix budget differs')
                    outputs[arm] = x.cpu()
        if (record['diagnostic_nfe'] != geometry.CALLS or not same_rng(external, rng_state(device))
                or P.precision.precision_state(torch) != plan['native_policy']
                or not all(torch.equal(a.cpu(), state[n]) for a, n in ((z.tau, 'tau'), (z.zeta, 'zeta'), (z.D, 'D')))):
            raise RuntimeError('Budget/RNG/precision/parameters gate failed')
        for name, value in (('trajectory.pt', cache), ('outputs.pt', outputs)):
            tmp = attempt/(name+'.tmp');torch.save(value, tmp);tmp.replace(attempt/name)
        record.update(status='sampled', outputs_sha256=source.digest(attempt/'outputs.pt'),
                      trajectory_sha256=source.digest(attempt/'trajectory.pt'));save()
        # No GT in geometry, pulse magnitudes, continuation, or output selection.
        gt = load_image_as_tensor(spec['case']['image']).cpu()
        metrics = {arm: dict(loss=float((y-z.A.H(value.to(device))).double().square().mean()/(2*.05**2)),
                             **compute_all_metrics(value, gt, lpips_net='alex')) for arm, value in outputs.items()}
        if any(not math.isfinite(v) for m in metrics.values() for v in m.values()):raise RuntimeError('Nonfinite metric')
        if any(abs(metrics['BASE'][m]-reference['metrics'][m]) > tol for m, tol in
               (('psnr', .01), ('ssim', .001), ('lpips', .001))):raise RuntimeError('Metric protocol drift')
        P.verify_case_sources(plan, spec)
        record.update(status='complete', metrics=metrics, gates=dict(
            baseline=dict(passed=True, tensor_error=parity, scalar_error=scalar_error),
            noise=noise_gate, budget_rng_precision=dict(passed=True), fixed_parameters=dict(passed=True),
            causal_geometry=dict(passed=True), GT_after_outputs=dict(passed=True)))
        save();source.signed(path, record)
        print('A35 CASE:', spec['case']['key'], 'complete; 108 calls; full geometry and 10 real final outcomes.', flush=True)
    except BaseException as error:
        record.update(status='failed', error=repr(error));save();raise
    finally:
        hook.remove()


def worker(args):
    import fcntl
    import torch
    from utils.diag_state_weight_scan import preserved_rng
    root = Path(args.root).resolve();plan = source.check_plan(root)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = [s for i, s in enumerate(sorted(plan['cases'], key=lambda s: s['case']['key'])) if (i//2)%2 == args.shard]
        if all(completed(root, plan, s) for s in specs):return
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(args.device), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != plan['runtime'] or P.precision.precision_state(torch) != plan['native_policy']:
            raise ValueError('Runtime/precision differs')
        torch.set_num_threads(2)
        with preserved_rng(), torch.no_grad():
            model = P.coupled.archive.recovery.load_model(plan, args.device)
            for spec in specs:run_case(root, plan, spec, model, args.device)
        source.check_plan(root)


def pictures(folder, spec, record):
    import torch
    from PIL import Image, ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    outputs = torch.load(folder/'outputs.pt', map_location='cpu', weights_only=True)
    cache = torch.load(folder/'trajectory.pt', map_location='cpu', weights_only=True)
    gt = _to_uint8_numpy(load_image_as_tensor(spec['case']['image']))
    for factor in geometry.FACTORS:
        names = ['GT', 'BASE']+[f'T{t}_M{factor:g}' for _, t in geometry.POINTS]
        sheet = Image.new('RGB', (20+276*len(names), 660), 'white');draw = ImageDraw.Draw(sheet)
        draw.text((20, 8), spec['case']['key']+f' | one guidance pulse x{factor:g}; every arm 300 NFE', fill='black', font=font(18))
        for j, arm in enumerate(names):
            x = 20+276*j;draw.text((x, 42), arm, fill='black', font=font(17))
            if arm == 'GT':raw = Image.fromarray(gt)
            else:
                m = record['metrics'][arm];raw = Image.fromarray(_to_uint8_numpy(outputs[arm]))
                draw.text((x, 66), f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}', fill='black', font=font(13))
            sheet.paste(raw, (x, 100));sheet.paste(raw.crop((64,48,192,176)).resize((256,256), Image.Resampling.NEAREST), (x, 380))
        sheet.save(folder/f'comparison_M{factor:g}.png')
    names = [('GT', gt)]+[(f'clean t={record["rows"][k]["t"]}', _to_uint8_numpy(cache['clean'][k])) for k in (0,8,19,24,27,28,29)]+[('FINAL BASE', _to_uint8_numpy(outputs['BASE']))]
    sheet = Image.new('RGB', (20+276*len(names), 310), 'white');draw = ImageDraw.Draw(sheet)
    for j, (name, arr) in enumerate(names):
        x = 20+j*276;draw.text((x, 12), name, fill='black', font=font(17));sheet.paste(Image.fromarray(arr), (x, 45))
    sheet.save(folder/'trajectory_frames.png')
    plot = Image.new('RGB', (1200, 760), 'white');draw = ImageDraw.Draw(plot)
    panels = [('Physical residual and gamma', [('R','#1674a5'),('gamma','#d66b22')]),
              ('Clean-path curvature ln(1+K)', [('log_curvature','#1674a5'),('log_high_curvature','#d66b22')]),
              ('Path backtracking: full / high', [('backtrack','#1674a5'),('high_backtrack','#d66b22')]),
              ('Prior-guidance alignment: full / high', [('prior_guide_cos','#1674a5'),('high_prior_guide_cos','#d66b22')])]
    for panel, (title, lines) in enumerate(panels):
        left, top = 65+(panel%2)*600, 60+(panel//2)*370;w,h=480,245
        draw.text((left, top-40), title, fill='black', font=font(17))
        series = []
        for key, color in lines:
            vals = [r[key] if key in ('R','gamma') else r['state']['features'][key] for r in record['rows']]
            series.append((key,color,vals))
        values = [v for _,_,vs in series for v in vs if v is not None]
        lo,hi=(min(values),max(values)) if values else (0.,1.)
        if hi <= lo:hi=lo+1.
        draw.rectangle((left,top,left+w,top+h),outline='#888')
        draw.text((left-50,top),f'{hi:.2g}',fill='black',font=font(14));draw.text((left-50,top+h-15),f'{lo:.2g}',fill='black',font=font(14))
        for line_no,(key,color,vals) in enumerate(series):
            previous=None
            for row,value in zip(record['rows'],vals):
                if value is None:previous=None;continue
                point=(left+w*row['u'],top+h*(hi-value)/(hi-lo))
                if previous is not None:draw.line((previous,point),fill=color,width=2)
                draw.ellipse((point[0]-2,point[1]-2,point[0]+2,point[1]+2),fill=color);previous=point
            draw.text((left,top+h+20+line_no*20),key,fill=color,font=font(13))
        draw.text((left+w-180,top+h+5),'u: actual log-noise progress',fill='black',font=font(12))
    plot.save(folder/'geometry_trajectory.png')


def summarize(args):
    root=Path(args.root).resolve();plan=source.check_plan(root);records=[];paths={}
    for spec in plan['cases']:
        item=completed(root,plan,spec)
        if item:paths[spec['case']['key']]=item[0];records.append(item[1])
    attempts=[source.read(p) for p in root.glob('*/attempt_*/run.json')]
    result=dict(experiment=plan['experiment'],completed_cases=len(records),planned_cases=12,
                method_nfe=0,diagnostic_nfe=sum(r['diagnostic_nfe'] for r in attempts),planned_diagnostic_nfe=1296,
                deployment_nfe_per_arm=300,partial_attempts=sum(r['status']!='complete' for r in attempts),
                cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] in ('running','sampled') for r in attempts),limits=plan['scope'])
    if len(records)==12:
        analysis=geometry.analyze(records);source.signed(root/'analysis.json',analysis)
        result.update(classification=analysis['classification'],supported_families=analysis['supported_families'],
                      prediction_mae={factor:{m:data['mae'] for m,data in group['targets'].items()}
                                      for factor,group in analysis['actions'].items()})
        effects={}
        for k,t in geometry.POINTS:
            for factor in geometry.FACTORS:
                arm=f'T{t}_M{factor:g}'
                delta={m:statistics.mean(r['metrics'][arm][m]-r['metrics']['BASE'][m] for r in records) for m in ('psnr','ssim','lpips','loss')}
                effects[arm]=dict(mean_delta=delta,quality_guard=delta['ssim']>=-.002 and delta['lpips']<=.005)
        result['fixed_pulse_effects']=effects
        result['information_tests']= {f:{m:v['comparisons'] for m,v in a['targets'].items()} for f,a in analysis['actions'].items()}
        result['action_audit'] = dict(
            models={name:{k:v for k,v in data.items() if k not in ('choices','per_image')}
                    for name,data in analysis['action_audit']['models'].items()},
            comparisons=analysis['action_audit']['comparisons'],limits=analysis['action_audit']['limits'])
    else:result['classification']='INCOMPLETE_PILOT'
    source.signed(root/'summary.json',result)
    if args.archive:
        for spec in plan['cases']:
            key=spec['case']['key']
            if key in paths:pictures(paths[key].parent,spec,next(r for r in records if r['case']['key']==key))
    print('A35 Summary:',json.dumps(result,ensure_ascii=False),flush=True);print('Records:',root,flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A35':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a33-dir',default=parent.ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
