"""A2d: direction refresh versus matched-amplitude direction; 30 diagnostic calls per case.

Reuses completed A2a snapshots and audits completed A2c. All endpoints share
the archived prior and noise. No training, noise-grid refinement or GT selection.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import archive_guidance_replication_stage as archive
from utils.diag_state_trajectory import digest

CARD = PROJECTS/'experiment_cards/A2d_direction_refresh_feasibility_20261019.md'
from utils import diag_one_step_guidance_scale as previous
TIMES = (903, 641, 319, 54, 14)
WRITE = archive.recovery.write_json
METRICS = ('psnr', 'ssim', 'lpips')


def branch_inputs(original_next, q0, qh, a0, ah):
    """F/M have identical second-stage RMS; supports NumPy and Torch tensors."""
    if not (math.isfinite(a0) and math.isfinite(ah) and a0 > 0 and ah >= 0):
        raise ValueError('Need finite positive original RMS and nonnegative refreshed RMS')
    if original_next.shape != q0.shape or qh.shape != q0.shape:
        raise ValueError('Branch tensor shapes differ')
    return dict(C=original_next+0, F=original_next+.5*(qh-q0),
                M=original_next+.5*((ah/a0)-1.)*q0)


def geometry(a0, ah, chord, repeat_chord):
    if any(not math.isfinite(v) for v in (a0, ah, repeat_chord)) or a0 <= 0 or ah < 0 or repeat_chord < 0:
        raise ValueError('Invalid geometry amplitude/floor')
    if chord is not None and (not math.isfinite(chord) or chord < 0 or chord > 2.+1e-10):
        raise ValueError('Invalid unit-vector chord')
    resolved = chord is not None and chord > 10*max(repeat_chord, 1e-12)
    return dict(a0=a0, ah=ah, amplitude_ratio=ah/a0, chord=chord,
                cosine=None if chord is None else max(-1., min(1., 1.-.5*chord*chord)),
                chi=None if chord is None else chord/(.5*a0),
                repeat_chord=repeat_chord, resolved=resolved,
                expected_F_M_rms=None if chord is None else .5*ah*chord)


def next_row(rows, probe):
    k = probe['step']
    if len(rows) != 30 or not 0 <= k < 29 or rows[k]['t'] != probe['t']:
        raise ValueError('Need original nonterminal snapshot')
    row = rows[k+1]
    if rows[k]['t_prev'] != row['t'] or row['parameter_index'] != 28-k:
        raise ValueError('Next-time/index identity differs')
    return row


def effect(candidate, baseline, floors):
    delta = {k: candidate[k]-baseline[k] for k in ('loss', *METRICS)}
    if not all(math.isfinite(v) for v in delta.values()):
        raise ValueError('Non-finite intervention outcome')
    limits = {k: 10*max(floors[k], dict(loss=1e-10, psnr=1e-6, ssim=1e-7, lpips=1e-7)[k]) for k in delta}
    nonworse = delta['psnr'] >= -limits['psnr'] and delta['ssim'] >= -limits['ssim'] and delta['lpips'] <= limits['lpips']
    better = delta['psnr'] > limits['psnr'] or delta['ssim'] > limits['ssim'] or delta['lpips'] < -limits['lpips']
    return dict(delta=delta, numerical_limits=limits, quality_pareto=nonworse and better,
                joint_local_candidate=nonworse and better and delta['loss'] <= limits['loss'])


def summarize(records):
    complete = [r for r in records if r['status'] == 'complete']
    layers = []
    for t in TIMES:
        points = [(r['case'], p) for r in complete for p in r['points'] if p['source_t'] == t]
        images = {c['image_id'] for c, p in points}
        valid = [(c, p) for c, p in points if p['shift_resolved']]
        hits = [(c, p) for c, p in valid if p['joint_direction_candidate']]
        consistent = sum(len([p for c, p in points if c['image_id'] == image]) == 2 and
                         len([p for c, p in hits if c['image_id'] == image]) == 2 for image in images)
        effects = {comparison: {k: statistics.mean(p['effects'][comparison]['delta'][k] for c, p in points) if points else None
                               for k in ('loss', *METRICS)} for comparison in ('F_C', 'M_C', 'F_M')}
        layers.append(dict(source_t=t, n=len(points), image_clusters=len(images), resolved=len(valid),
                           joint_local_points=len(hits), images_both_seeds_joint=consistent, mean_delta=effects))
    supported = len(complete) == 6 and any(l['images_both_seeds_joint'] >= 2 for l in layers)
    return dict(completed_cases=len(complete), planned_cases=6,
                method_nfe=sum(r['method_nfe'] for r in records), diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
                classification=('LOCAL_DIRECTION_REFRESH_CANDIDATE_ONLY' if supported else 'LOCAL_DIRECTION_REFRESH_NOT_SUPPORTED') if len(complete) == 6 else 'INCOMPLETE_PILOT',
                layers=layers, limits='Three development image clusters. Next-time clean endpoints only; no final reconstruction, scheduling, passive-cosine control or GT selection claim.')


def direction_candidate(effects, resolved):
    return bool(resolved and effects['F_C']['joint_local_candidate'] and effects['F_M']['joint_local_candidate'])


def bind(args):
    root = Path(args.source_plan_dir).resolve()
    summary, active = archive.audit_source(root)
    parent_plan = json.loads((root/'plan.json').read_text())
    registry = archive.recovery.registry_for(root)
    a2c = Path(args.a2c_dir).resolve()
    a2c_plan = json.loads((a2c/'plan.json').read_text())
    if (Path(a2c_plan['source']).resolve() != root or set(a2c_plan['source_cases']) != set(active) or
            len(previous.completed(a2c, a2c_plan)) != 6):
        raise ValueError('Need six completed compatible A2c cases; do not rerun them')
    if any(digest(p) != sha for p, sha in a2c_plan['identity_files'].items()):
        raise ValueError('Frozen A2c sources changed')
    a2c_records = [json.loads(p.read_text()) for p in a2c.glob('*/run.json')]
    if previous.summarize(a2c_records) != json.loads((a2c/'summary.json').read_text()):
        raise ValueError('A2c saved summary differs from complete cases')
    paths = [root/'plan.json', root/'recovery_registry.json', root/'recovery_summary.json', CARD, Path(__file__),
             PROJECTS/'utils/metrics.py', PROJECTS/'utils/diag_guidance_local_probe.py',
             PROJECTS/'utils/diag_cached_guidance_geometry.py', PROJECTS/'run_a2d_20261019.sh', Path(archive.__file__)]
    paths.extend([a2c/'plan.json', a2c/'summary.json', Path(previous.__file__)])
    paths.extend(a2c.glob('*/run.json')); paths.extend(a2c.glob('*/run.sha256'))
    sources = {}
    for key, record in active.items():
        source = root/registry['active'][key]['directory'] if key in registry['active'] else root/key
        sources[key] = str(source)
        paths.extend(source/name for name in ('run.json', 'pre_last_state.pt', 'trajectory.pt'))
        paths.append(Path(record['measurement']['path']))
        if tuple(p['t'] for p in record['probes']) != TIMES:
            raise ValueError('Original snapshots differ')
    plan = dict(experiment='direction_refresh_plan', source=str(root), source_cases=sources,
                identity_files={**parent_plan['identity_files'], **{str(p): digest(p) for p in paths}}, times=list(TIMES), midpoint_fraction=.5, arms=['C', 'F', 'M'], a2c_source=str(a2c),
                planned_method_nfe=0, planned_diagnostic_nfe=180,
                parent_cost=dict(method=summary['method_nfe'], diagnostic=summary['diagnostic_nfe']),
                preceding_a2c_cost={k: previous.summarize(a2c_records)[k] for k in ('method_nfe', 'diagnostic_nfe')})
    directory = Path(args.run_dir).resolve()
    if any(directory.is_relative_to(p) or p.is_relative_to(directory) for p in (root, a2c)):
        raise ValueError('New run must be separate from A2a and A2c sources')
    if directory.exists():
        existing = json.loads((directory/'plan.json').read_text())
        if existing != plan:
            raise ValueError('Frozen plan/source/card/implementation changed; do not overwrite')
    elif not args.inspect:
        directory.mkdir(parents=True, exist_ok=False)
        WRITE(directory/'plan.json', plan)
    return root, directory, plan, active


def completed(directory, plan):
    result = {}
    for key in plan['source_cases']:
        path = directory/key/'run.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if (digest(path) != (path.parent/'run.sha256').read_text().strip() or r['case']['key'] != key or
                r['status'] != 'complete' or r['plan_sha256'] != digest(directory/'plan.json') or
                r['method_nfe'] != 0 or r['diagnostic_nfe'] != 30 or
                len(r['points']) != 5 or len(r['calls']) != 30 or tuple(p['source_t'] for p in r['points']) != TIMES or
                not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('Existing interrupted/failed/incompatible case retained: '+key)
        if digest(path.parent/'branches.pt') != r['branch_archive']['sha256']:
            raise ValueError('Saved branch inputs changed; do not rerun or silently reuse: '+key)
        result[key] = r
    return result


def run_case(parent_plan, parent, source, directory, plan, model, device):
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.diag_cached_guidance_geometry import rms, unit_direction
    from utils.metrics import compute_all_metrics
    directory.mkdir(exist_ok=False)
    (directory/'experiment_card.md').write_text(CARD.read_text(), encoding='utf-8')
    record = dict(experiment='direction_refresh_case', status='running', case=parent['case'],
                  plan_sha256=digest(directory.parent/'plan.json'), method_nfe=0, diagnostic_nfe=0,
                  calls=[], points=[], gates={}, config=parent['config'],
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
                  runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
                  experiment_card=CARD.read_text(), outcome='Next-time clean endpoints; no rollout or GT branch selection')
    started = time.perf_counter()
    def save():
        record['seconds'] = time.perf_counter()-started
        WRITE(directory/'run.json', record)
    save()
    context = {}
    def count(_module, inputs):
        record['diagnostic_nfe'] += 1
        record['calls'].append(dict(context, t=int(inputs[1].flatten()[0]), cumulative=record['diagnostic_nfe']))
        if record['diagnostic_nfe'] > 30 or record['calls'][-1]['t'] != context['eval_t']:
            raise RuntimeError('UNet budget/time gate failed')
    hook = model.model.register_forward_pre_hook(count)
    try:
        before = rng_state(device)
        baseline_checks, midpoint_checks, input_checks = [], [], []
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.no_grad():
            state = torch.load(source/'pre_last_state.pt', map_location='cpu', weights_only=True)
            cache = torch.load(source/'trajectory.pt', map_location='cpu', weights_only=True)
            if any(tuple(cache[k].shape) != (30, 3, 256, 256) or cache[k].dtype != torch.float32 or not torch.isfinite(cache[k]).all()
                   for k in ('clean', 'guidance', 'latent_inputs')):
                raise ValueError('Unexpected source tensor contract')
            if state['tau'].tolist() != parent_plan['grid']:
                raise ValueError('PRE grid differs')
            base = get_operator('super_resolution', device=device, scale_factor=4, noise_sigma=0.)
            operator = unit_operator(TransposeMode(base, 'legacy_bicubic').to(device)).to(device)
            z = ZAPS(model, operator, img_size=256, **parent_plan['config'])
            z.tau = state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device)); z.D.copy_(state['D'].to(device))
            y = torch.load(parent['measurement']['path'], map_location='cpu', weights_only=True)['y'].to(device)
            pending, saved_inputs = [], []
            with model_rng_guard(model.model, device):
                for probe in parent['probes']:
                    k = probe['step']; row = next_row(parent['rows'], probe)
                    s, index = row['t'], row['parameter_index']
                    current_index = parent['rows'][k]['parameter_index']
                    if current_index != 29-k:
                        raise ValueError('Current parameter index differs')
                    x = cache['latent_inputs'][k:k+1].to(device)
                    x_next = cache['latent_inputs'][k+1:k+2].to(device)
                    q0 = cache['guidance'][k:k+1].to(device)
                    a0 = rms(q0)
                    if not math.isclose(a0, probe['q_rms'], rel_tol=1e-6, abs_tol=1e-12):
                        raise ValueError('Cached correction differs from original probe')
                    n0 = unit_direction(q0)
                    # Source cohort already passed nonzero-q gates; a contradictory source is a technical error.
                    if n0 is None:
                        raise ValueError('Predeclared source q is undefined; retain source, do not replace point')
                    context.clear(); context.update(source_step=k, source_t=probe['t'], next_t=s, eval_t=s, arm='C')
                    centers = []
                    for repeat in (0, 1):
                        context['repeat'] = repeat
                        centers.append(evaluate_field(z, y, x_next.clone(), s, index))
                    check = dict(step=k, clean_error=relative_error(cache['clean'][k+1:k+2], centers[0]['clean'].cpu()),
                                 q_error=relative_error(cache['guidance'][k+1:k+2], centers[0]['q'].cpu()),
                                 repeat_error=relative_error(centers[0]['clean'], centers[1]['clean']),
                                 loss_error=abs(centers[0]['loss']-row['measurement_loss']))
                    if not all(math.isfinite(v) for v in check.values()) or max(check['clean_error'], check['q_error']) > 1e-6 or check['repeat_error'] > 1e-7 or check['loss_error'] > 1e-7+2e-5*abs(row['measurement_loss']):
                        raise RuntimeError('Baseline field parity failed: '+json.dumps(check))
                    baseline_checks.append(check)
                    xh = x+.5*q0
                    mid_rounding = rms((xh.double()-x.double())-.5*q0.double())/(.5*a0)
                    if mid_rounding > .01:
                        raise RuntimeError('Midpoint input is not resolved')
                    mids = []
                    context.update(eval_t=probe['t'], arm='midpoint')
                    for repeat in (0, 1):
                        context['repeat'] = repeat
                        mids.append(evaluate_field(z, y, xh, probe['t'], current_index))
                    qh = mids[0]['q']; ah = rms(qh)
                    nh, nr = unit_direction(qh), unit_direction(mids[1]['q'])
                    mid_check = dict(step=k, q_repeat_error=relative_error(qh, mids[1]['q']),
                                     clean_repeat_error=relative_error(mids[0]['clean'], mids[1]['clean']))
                    if mid_check['q_repeat_error'] > 1e-6 or mid_check['clean_repeat_error'] > 1e-7 or (nh is None) != (nr is None):
                        raise RuntimeError('Midpoint repeat is unstable')
                    midpoint_checks.append(mid_check)
                    chord = float((nh-n0).norm()) if nh is not None else None
                    floor = float((nh-nr).norm()) if nh is not None else 0.
                    geo = geometry(a0, ah, chord, floor)
                    geo.update(midpoint_clip=mids[0]['clip_fraction'], midpoint_loss=mids[0]['loss'], midpoint_rounding=mid_rounding)
                    inputs = branch_inputs(x_next, q0, qh, a0, ah)
                    expected = dict(F=.5*(qh.double()-q0.double()), M=.5*((ah/a0)-1.)*q0.double())
                    shifts = {}
                    for arm in ('F', 'M'):
                        actual = inputs[arm].double()-x_next.double()
                        err = rms(actual-expected[arm])
                        # Unresolved near-zero shifts are preserved as uncertainty, not amplified by a near-zero denominator.
                        limit = .01*max(rms(expected[arm]), 1e-12*a0)
                        if err > limit and rms(expected[arm]) > 1e-6*a0:
                            raise RuntimeError('Branch shift is not resolved: '+arm)
                        shifts[arm] = dict(absolute_error=err, expected_rms=rms(expected[arm]), resolved=err <= limit)
                    actual_difference = rms(inputs['F'].double()-inputs['M'].double())
                    expected_difference = geo['expected_F_M_rms'] if chord is not None else 0.
                    identity_error = abs(actual_difference-expected_difference)
                    identity_floor = shifts['F']['absolute_error']+shifts['M']['absolute_error']
                    if identity_error > 2*identity_floor+1e-8*max(a0, ah, 1e-12):
                        raise RuntimeError('Matched-amplitude identity failed')
                    geo['endpoint_direction_shift_resolved'] = expected_difference > 10*max(identity_floor, 1e-12*max(a0, ah))
                    input_checks.append(dict(step=k, shifts=shifts, identity_error=identity_error, rounding_floor=identity_floor))
                    branches = dict(C=dict(clean=centers[0]['clean'].cpu(), loss=centers[0]['loss'], clip=centers[0]['clip_fraction']))
                    context.update(eval_t=s, repeat=None)
                    for arm in ('F', 'M'):
                        context['arm'] = arm
                        field = evaluate_field(z, y, inputs[arm], s, index)
                        branches[arm] = dict(clean=field['clean'].cpu(), loss=field['loss'], clip=field['clip_fraction'])
                    saved_inputs.append(dict(source_step=k, source_t=probe['t'], next_t=s,
                                             qh=qh.cpu(), F=inputs['F'].cpu(), M=inputs['M'].cpu()))
                    pending.append((probe, row, branches, centers[1]['clean'].cpu(), abs(centers[1]['loss']-centers[0]['loss']), geo, shifts))
                    save()
                    print('PROGRESS:', parent['case']['key'], 't=', probe['t'], 'calls=', record['diagnostic_nfe'], flush=True)
            if not (torch.equal(z.zeta.cpu(), state['zeta']) and torch.equal(z.D.cpu(), state['D']) and torch.equal(z.tau.cpu(), state['tau'])):
                raise RuntimeError('Parameters/grid changed')
            artifact = dict(points=[{k:v for k,v in p.items() if k not in ('qh','F','M')} for p in saved_inputs],
                            **{key:torch.cat([p[key] for p in saved_inputs], dim=0) for key in ('qh','F','M')})
            temporary = directory/'branches.pt.tmp'
            torch.save(artifact, temporary)
            temporary.replace(directory/'branches.pt')
            record['branch_archive'] = dict(path=str(directory/'branches.pt'), sha256=digest(directory/'branches.pt'),
                                            scope='Actual midpoint correction and F/M next inputs; CPU float32, five original snapshots')
            # GT is loaded only after all branch inputs and all network calls are fixed.
            gt = load_image_as_tensor(parent['case']['image']).cpu()
            for probe, row, branches, repeated_clean, repeated_loss_error, geo, shifts in pending:
                for branch in branches.values():
                    branch.update(compute_all_metrics(branch.pop('clean'), gt, lpips_net='alex'))
                baseline = branches['C']
                repeated_metrics = compute_all_metrics(repeated_clean, gt, lpips_net='alex')
                drifts = {key: abs(baseline[key]-row[key]) for key in METRICS}
                if any(drifts[key] > limit for key, limit in (('psnr', .01), ('ssim', .001), ('lpips', .001))):
                    raise RuntimeError('Baseline metric convention/parity changed')
                floors = {key: max(drifts[key], abs(repeated_metrics[key]-baseline[key])) for key in METRICS}
                floors['loss'] = max(repeated_loss_error, abs(baseline['loss']-row['measurement_loss']))
                effects = dict(F_C=effect(branches['F'], branches['C'], floors), M_C=effect(branches['M'], branches['C'], floors),
                               F_M=effect(branches['F'], branches['M'], floors))
                resolved = geo['resolved'] and geo['endpoint_direction_shift_resolved'] and all(s['resolved'] for s in shifts.values())
                point = dict(source_step=probe['step'], source_t=probe['t'], next_t=row['t'], branches=branches,
                             floors=floors, geometry=geo, shift_resolved=resolved,
                             predictors={key: parent['rows'][probe['step']].get(key) for key in
                                         ('u', 'guidance_cosine', 'guidance_variation3_per_u', 'residual_noise_ratio', 'raw_x0_clip_fraction')},
                             effects=effects, joint_direction_candidate=direction_candidate(effects, resolved))
                record['points'].append(point)
                print('POINT:', parent['case']['key'], json.dumps(point), flush=True)
        if not same_rng(before, rng_state(device)):
            raise RuntimeError('External RNG changed')
        if record['diagnostic_nfe'] != 30 or len(record['points']) != 5:
            raise RuntimeError('Need five points and thirty diagnostic calls')
        if any(digest(p) != sha for p, sha in plan['identity_files'].items()):
            raise RuntimeError('Source/implementation changed during diagnostic')
        record['gates'] = dict(baseline=dict(passed=True, checks=baseline_checks), midpoint_repeat=dict(passed=True, checks=midpoint_checks),
                               matched_amplitude=dict(passed=True, checks=input_checks), shared_prior_and_noise=dict(passed=True),
                               parameters_grid_rng=dict(passed=True), identity_and_budget=dict(passed=True, method_nfe=0, diagnostic_nfe=30))
        record['status'] = 'complete'; save()
        (directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        print('CASE:', parent['case']['key'], 'Gates:', json.dumps(record['gates']), flush=True)
        return record
    except BaseException as error:
        record['status'] = 'failed'; record['error'] = repr(error); save(); raise
    finally:
        hook.remove()


def stored_summary(directory):
    attempts = [json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))]
    result = summarize(attempts)
    result['cost_is_lower_bound_if_abruptly_interrupted'] = any(r['status'] == 'running' for r in attempts)
    return result


def execute(args):
    root, directory, plan, parents = bind(args)
    print('Source AUDIT PASS: six A2a and six A2c cases; planned NFE=0 method +180 diagnostic.', flush=True)
    if args.inspect:
        print('A2d INSPECT PASS: six cases x five points x six calls. No training or noise-grid change.', flush=True)
        return
    records = completed(directory, plan)
    keys = sorted(parents)  # The predefined 00020_s1000 technical smoke runs first.
    todo = [key for key in keys if key not in records]
    try:
        if todo:
            archive.recovery.require_space(directory)
            parent_plan = json.loads((root/'plan.json').read_text())
            import torch
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                model = archive.recovery.load_model(parent_plan, args.device)
                for key in todo:
                    records[key] = run_case(parent_plan, parents[key], Path(plan['source_cases'][key]), directory/key, plan, model, args.device)
    finally:
        summary = stored_summary(directory)
        WRITE(directory/'summary.json', summary)
        print('A2d Summary:', json.dumps(summary), '\nRecords:', directory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir', default='results/diag_guidance_replication/A2a_20261018')
    parser.add_argument('--a2c-dir', default='/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019')
    parser.add_argument('--run-dir', default='/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019')
    parser.add_argument('--device', default='cuda')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if args.summary_only:
        if not (directory/'plan.json').is_file():
            raise ValueError('No stored A2d plan; no experiment started')
        print('A2d Summary:', json.dumps(stored_summary(directory)), '\nRecords:', directory)
    elif args.inspect:
        execute(args)
    else:
        # Per-run advisory lock survives SSH disconnection with the process.
        # Concurrent submits cannot enter bind/create or overwrite a running case.
        import fcntl
        directory.parent.mkdir(parents=True, exist_ok=True)
        with (directory.parent/(directory.name+'.lock')).open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A2d is already running; inspect the existing log instead of resubmitting') from None
            execute(args)


if __name__ == '__main__':
    main()
