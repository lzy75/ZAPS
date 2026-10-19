"""A2c: controlled one-step guidance dose; 20 diagnostic UNet calls per case.

Reuses completed A2a snapshots. gamma=0/.5/1 share the SAME archived prior and
noise. Evaluate clean estimates at the actual next time, no rollout or training.
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

CARD = PROJECTS/'experiment_cards/A2c_one_step_guidance_scale_20261019.md'
TIMES = (903, 641, 319, 54, 14)
WRITE = archive.recovery.write_json
METRICS = ('psnr', 'ssim', 'lpips')


def shifted_input(original_next, q, gamma):
    if gamma not in (0., .5, 1.):
        raise ValueError('Only predefined gamma=0/.5/1 allowed')
    return original_next.clone() if gamma == 1. else original_next+(gamma-1.)*q


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
        for gamma in ('0.5', '0.0'):
            points = [dict(case=r['case']['key'], image=r['case']['image_id'], **p['effects'][gamma])
                      for r in complete for p in r['points'] if p['source_t'] == t]
            images = {p['image'] for p in points}
            consistent = sum(len([p for p in points if p['image'] == image]) == 2 and
                             all(p['joint_local_candidate'] for p in points if p['image'] == image) for image in images)
            layers.append(dict(source_t=t, gamma=float(gamma), n=len(points), image_clusters=len(images),
                               mean_delta={k: statistics.mean(p['delta'][k] for p in points) if points else None for k in ('loss', *METRICS)},
                               joint_local_points=sum(p['joint_local_candidate'] for p in points),
                               images_both_seeds_joint=consistent))
    supported = len(complete) == 6 and any(l['gamma'] == .5 and l['images_both_seeds_joint'] >= 2 for l in layers)
    return dict(completed_cases=len(complete), planned_cases=6,
                method_nfe=sum(r['method_nfe'] for r in records), diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
                classification=('LOCAL_HALF_DOSE_CANDIDATE_ONLY' if supported else 'LOCAL_HALF_DOSE_NOT_SUPPORTED') if len(complete) == 6 else 'INCOMPLETE_PILOT',
                layers=layers, limits='Three image clusters; paired one-step clean evaluation, not final reconstruction or online state gain. No GT selection.')


def bind(args):
    root = Path(args.source_plan_dir).resolve()
    summary, active = archive.audit_source(root)
    parent_plan = json.loads((root/'plan.json').read_text())
    registry = archive.recovery.registry_for(root)
    paths = [root/'plan.json', root/'recovery_registry.json', root/'recovery_summary.json', CARD, Path(__file__),
             PROJECTS/'utils/metrics.py', Path(archive.__file__)]
    sources = {}
    for key, record in active.items():
        source = root/registry['active'][key]['directory'] if key in registry['active'] else root/key
        sources[key] = str(source)
        paths.extend(source/name for name in ('run.json', 'pre_last_state.pt', 'trajectory.pt'))
        paths.append(Path(record['measurement']['path']))
        if tuple(p['t'] for p in record['probes']) != TIMES:
            raise ValueError('Original snapshots differ')
    plan = dict(experiment='one_step_guidance_scale_plan', source=str(root), source_cases=sources,
                identity_files={**parent_plan['identity_files'], **{str(p): digest(p) for p in paths}}, times=list(TIMES), gammas=[1., .5, 0.],
                planned_method_nfe=0, planned_diagnostic_nfe=120,
                parent_cost=dict(method=summary['method_nfe'], diagnostic=summary['diagnostic_nfe']))
    directory = Path(args.run_dir).resolve()
    if directory.is_relative_to(root) or root.is_relative_to(directory):
        raise ValueError('New run must be separate from the A2a source')
    if directory.exists():
        existing = json.loads((directory/'plan.json').read_text())
        if existing != plan:
            raise ValueError('Frozen plan/source/card/implementation changed; do not overwrite')
    elif not args.inspect:
        if args.remaining:
            raise ValueError('Run the first case before --remaining')
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
                r['method_nfe'] != 0 or r['diagnostic_nfe'] != 20 or
                len(r['points']) != 5 or len(r['calls']) != 20 or tuple(p['source_t'] for p in r['points']) != TIMES or
                not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('Existing interrupted/failed/incompatible case retained: '+key)
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
    from utils.diag_cached_guidance_geometry import rms
    from utils.metrics import compute_all_metrics
    directory.mkdir(exist_ok=False)
    (directory/'experiment_card.md').write_text(CARD.read_text(), encoding='utf-8')
    record = dict(experiment='one_step_guidance_scale_case', status='running', case=parent['case'],
                  plan_sha256=digest(directory.parent/'plan.json'), method_nfe=0, diagnostic_nfe=0,
                  calls=[], points=[], gates={}, config=parent['config'],
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
                  runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
                  experiment_card=CARD.read_text(), outcome='Next-time clean estimate, not final output', gt_role='offline evaluation of all predetermined branches only')
    started = time.perf_counter()
    def save():
        record['seconds'] = time.perf_counter()-started
        WRITE(directory/'run.json', record)
    save()
    context = {}
    def count(_module, inputs):
        record['diagnostic_nfe'] += 1
        record['calls'].append(dict(context, t=int(inputs[1].flatten()[0]), cumulative=record['diagnostic_nfe']))
        if record['diagnostic_nfe'] > 20 or record['calls'][-1]['t'] != context['next_t']:
            raise RuntimeError('UNet budget/time gate failed')
    hook = model.model.register_forward_pre_hook(count)
    try:
        before = rng_state(device)
        baseline_checks = []
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.no_grad():
            state = torch.load(source/'pre_last_state.pt', map_location='cpu', weights_only=True)
            cache = torch.load(source/'trajectory.pt', map_location='cpu', weights_only=True)
            if any(tuple(cache[k].shape) != (30, 3, 256, 256) or cache[k].dtype != torch.float32 or not torch.isfinite(cache[k]).all()
                   for k in ('clean', 'guidance', 'latent_inputs')):
                raise ValueError('Unexpected archived tensor contract')
            if state['tau'].tolist() != parent_plan['grid']:
                raise ValueError('PRE grid differs')
            base = get_operator('super_resolution', device=device, scale_factor=4, noise_sigma=0.)
            operator = unit_operator(TransposeMode(base, 'legacy_bicubic').to(device)).to(device)
            z = ZAPS(model, operator, img_size=256, **parent_plan['config'])
            z.tau = state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device)); z.D.copy_(state['D'].to(device))
            y = torch.load(parent['measurement']['path'], map_location='cpu', weights_only=True)['y'].to(device)
            pending = []
            with model_rng_guard(model.model, device):
                for probe in parent['probes']:
                    k = probe['step']; row = next_row(parent['rows'], probe)
                    s, index = row['t'], row['parameter_index']
                    x_next = cache['latent_inputs'][k+1:k+2].to(device)
                    q = cache['guidance'][k:k+1].to(device)
                    if not math.isclose(rms(q), probe['q_rms'], rel_tol=1e-6, abs_tol=1e-12):
                        raise ValueError('Source correction differs from original probe')
                    context.clear(); context.update(source_step=k, source_t=probe['t'], next_t=s)
                    centers = []
                    for repeat in (0, 1):
                        context.update(gamma=1., repeat=repeat)
                        centers.append(evaluate_field(z, y, shifted_input(x_next, q, 1.), s, index))
                    check = dict(step=k, next_t=s,
                                 clean_error=relative_error(cache['clean'][k+1:k+2], centers[0]['clean'].cpu()),
                                 q_error=relative_error(cache['guidance'][k+1:k+2], centers[0]['q'].cpu()),
                                 repeat_error=relative_error(centers[0]['clean'], centers[1]['clean']),
                                 loss_error=abs(centers[0]['loss']-row['measurement_loss']))
                    if (not all(math.isfinite(v) for v in check.values()) or
                            max(check['clean_error'], check['q_error']) > 1e-6 or check['repeat_error'] > 1e-7 or
                            check['loss_error'] > 1e-7+2e-5*abs(row['measurement_loss'])):
                        raise RuntimeError('Original next-state field parity failed: '+json.dumps(check))
                    baseline_checks.append(check)
                    branches = {'1.0': dict(clean=centers[0]['clean'].cpu(), loss=centers[0]['loss'], clip=centers[0]['clip_fraction'])}
                    for gamma in (0., .5):
                        context.update(gamma=gamma, repeat=None)
                        candidate = shifted_input(x_next, q, gamma)
                        expected = (gamma-1.)*q.double()
                        rounding = rms((candidate.double()-x_next.double())-expected)/max(rms(expected), 1e-30)
                        if rounding > .01:
                            raise RuntimeError('Counterfactual input shift is not resolved')
                        field = evaluate_field(z, y, candidate, s, index)
                        branches[str(gamma)] = dict(clean=field['clean'].cpu(), loss=field['loss'], clip=field['clip_fraction'], input_shift_relative_error=rounding)
                    pending.append((probe, row, branches, centers[1]['clean'].cpu(), abs(centers[1]['loss']-centers[0]['loss'])))
            if not (torch.equal(z.zeta.cpu(), state['zeta']) and torch.equal(z.D.cpu(), state['D']) and torch.equal(z.tau.cpu(), state['tau'])):
                raise RuntimeError('Parameters/grid changed')
            # All states and all UNet calls fixed before GT loading or quality evaluation.
            gt = load_image_as_tensor(parent['case']['image']).cpu()
            for probe, row, branches, repeated_clean, repeated_loss_error in pending:
                for branch in branches.values():
                    branch.update(compute_all_metrics(branch.pop('clean'), gt, lpips_net='alex'))
                baseline = branches['1.0']
                repeated_metrics = compute_all_metrics(repeated_clean, gt, lpips_net='alex')
                drifts = {key: abs(baseline[key]-row[key]) for key in METRICS}
                if any(drifts[key] > limit for key, limit in (('psnr', .01), ('ssim', .001), ('lpips', .001))):
                    raise RuntimeError('Original metric convention/parity changed')
                floors = {key: max(drifts[key], abs(repeated_metrics[key]-baseline[key])) for key in METRICS}
                floors['loss'] = max(repeated_loss_error, abs(baseline['loss']-row['measurement_loss']))
                point = dict(source_step=probe['step'], source_t=probe['t'], next_t=row['t'], branches=branches,
                             floors=floors, predictors=dict(q_rms=probe['q_rms'], omega3=parent['rows'][probe['step']]['guidance_variation3_per_u'],
                                residual_noise_ratio=parent['rows'][probe['step']]['residual_noise_ratio'],
                                s_dir=[p['s_dir'] for p in probe['pairs']], kappa=[p['s_dir_per_state_rms'] for p in probe['pairs']]),
                             effects={g: effect(branches[g], baseline, floors) for g in ('0.5', '0.0')})
                record['points'].append(point)
                print('POINT:', parent['case']['key'], json.dumps(point), flush=True)
        if not same_rng(before, rng_state(device)):
            raise RuntimeError('External RNG changed')
        if record['diagnostic_nfe'] != 20 or len(record['points']) != 5:
            raise RuntimeError('Need exactly five points and twenty diagnostic calls')
        if any(digest(p) != sha for p, sha in plan['identity_files'].items()):
            raise RuntimeError('Source/implementation changed during diagnostic')
        record['gates'] = dict(baseline=dict(passed=True, checks=baseline_checks), shared_prior_and_noise=dict(passed=True),
                               parameters_grid_rng=dict(passed=True), identity_and_budget=dict(passed=True, method_nfe=0, diagnostic_nfe=20))
        record['status'] = 'complete'; save()
        (directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        print('CASE:', parent['case']['key'], 'Gates:', json.dumps(record['gates']), flush=True)
        return record
    except BaseException as error:
        record['status'] = 'failed'; record['error'] = repr(error); save(); raise
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--case', choices=['00020_s1000'])
    group.add_argument('--remaining', action='store_true')
    args = parser.parse_args()
    root, directory, plan, parents = bind(args)
    print('Source AUDIT PASS: six cases; planned new NFE=0 method +120 diagnostic; next-time clean outcomes only.', flush=True)
    if args.inspect:
        print('Smoke: 20 diagnostic; remaining five: 100. No training, rollout, new noise or GT selection.', flush=True)
        return
    records = completed(directory, plan)
    if args.remaining and '00020_s1000' not in records:
        raise ValueError('Complete the first technical smoke case before --remaining')
    keys = list(parents) if args.remaining else [args.case]
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
        # Include interrupted attempts and their consumed calls in the cost ledger.
        attempts = [json.loads(p.read_text()) for p in directory.glob('*/run.json')]
        summary = summarize(attempts)
        WRITE(directory/'summary.json', summary)
        print('A2c Summary:', json.dumps(summary), '\nRecords:', directory, flush=True)


if __name__ == '__main__':
    main()
