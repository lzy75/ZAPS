"""A16: one-score terminal strength control; 12 diagnostic calls, no new training."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_terminal_polish_control as parent
from utils import diag_clean_response_precision as precision
from utils.diag_state_trajectory import digest

WRITE = parent.WRITE
ROOT = '/data0/lzy/ZAPS-results/diag_terminal_state_strength/A16_20261020'
CARD = PROJECTS/'experiment_cards/A16_terminal_state_strength_20261020.md'
METRICS = ('loss', 'psnr', 'ssim', 'lpips')
FIXED = dict(F0=0., F05=.5, F1=1., F2=2.)
ARMS = (*FIXED, 'L', 'A')
GUARDS = dict(ssim=.002, lpips=.005)


def read(path):
    return json.loads(Path(path).read_text())


def strength(p, q, residual, action):
    """Float64 reductions; action is H(clean+q)-H(clean), including affine cancellation."""
    p, q, residual, action = [np.asarray(v, dtype=np.float64).reshape(-1) for v in (p, q, residual, action)]
    if p.shape != q.shape or residual.shape != action.shape or not all(np.isfinite(v).all() for v in (p, q, residual, action)):
        raise ValueError('Invalid strength vectors')
    pp, qq, pq, aa, ar = map(float, (p@p, q@q, p@q, action@action, action@residual))
    resolved = aa > 1e-20
    raw = ar/aa if resolved else None
    physical = float(np.clip(raw, 0, 2)) if resolved else 0.
    cap = min(2., .5*pp/(-pq)) if pq < 0 else 2.
    alpha = min(physical, cap)
    return dict(alpha=alpha, alpha_L=physical, alpha_raw=raw, alpha_cap=cap,
                action_resolved=resolved, guard_active=alpha < physical-1e-12,
                prior_q_cosine=pq/math.sqrt(pp*qq) if pp > 1e-20 and qq > 1e-20 else None,
                q_over_prior_norm=math.sqrt(qq/pp) if pp > 1e-20 else None,
                prior_norm2=pp, q_norm2=qq, prior_q_dot=pq, action_norm2=aa,
                prior_projection_after=pp+alpha*pq, required_projection=.5*pp)


def limits(floors):
    return {k:10*max(floors[k], 1e-6 if k == 'psnr' else 1e-10 if k == 'loss' else 1e-7) for k in METRICS}


def summarize(records):
    complete = [r for r in records if r['status'] == 'complete']
    pairs = [(a, 'C') for a in (*ARMS, 'S_native')] + [(a, b) for a in ('A', 'L') for b in (*FIXED, 'S_native', 'L') if a != b]
    comparisons = {}
    images = sorted({r['case']['image_id'] for r in complete})
    for a, b in pairs:
        delta = {k:statistics.mean(r['branches'][a][k]-r['branches'][b][k] for r in complete) if complete else None for k in METRICS}
        numerical = {k:statistics.mean(limits(r['floors'])[k] for r in complete) if complete else None for k in METRICS}
        wins = 0
        for image in images:
            group = [r for r in complete if r['case']['image_id'] == image]
            if len(group) == 2 and statistics.mean(r['branches'][a]['psnr']-r['branches'][b]['psnr']-limits(r['floors'])['psnr'] for r in group) > 0:
                wins += 1
        quality = bool(complete and delta['ssim'] >= -GUARDS['ssim'] and delta['lpips'] <= GUARDS['lpips'])
        positive = bool(complete and delta['psnr'] > numerical['psnr'])
        comparisons[a+'_'+b] = dict(mean_delta=delta, mean_numerical_limit=numerical, quality_guard=quality,
                                     psnr_positive_resolved=positive, images_psnr_better=wins,
                                     acceptable_vs_C=bool(b == 'C' and quality and positive and delta['loss'] <= numerical['loss']),
                                     pair_advantage=quality and positive and wins >= 2)
    acceptable_fixed = [a for a in (*FIXED, 'S_native') if comparisons[a+'_C']['acceptable_vs_C']]
    reference_arms = sorted(set(acceptable_fixed) | {'F1', 'S_native'})
    state = len(complete) == 6 and comparisons['A_C']['acceptable_vs_C'] and all(comparisons['A_'+b]['pair_advantage'] for b in reference_arms)
    angle = state and any(r['state']['guard_active'] for r in complete) and comparisons['A_L']['pair_advantage']
    classification = ('INCOMPLETE_PILOT' if len(complete) != 6 else 'STATE_STRENGTH_EXPLORATORY_CANDIDATE' if state else
                      'FIXED_STRENGTH_CANDIDATE_ONLY' if acceptable_fixed else 'NO_ACCEPTABLE_STRENGTH_GAIN')
    return dict(completed_cases=len(complete), planned_cases=6, image_clusters=len(images),
        method_nfe=sum(r['method_nfe'] for r in records), diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
        scientific_extra_nfe_per_case=1, parent_method_nfe_per_case=300, deployment_total_method_nfe_per_case=301,
        cumulative_queue_cost=dict(method_nfe=3600, diagnostic_nfe=2264+sum(r['diagnostic_nfe'] for r in records), total=5864+sum(r['diagnostic_nfe'] for r in records)),
        classification=classification, direction_guard_increment_candidate=angle,
        acceptable_fixed_arms=acceptable_fixed, state_reference_arms=reference_arms,
        guard_active_cases=sum(r['state']['guard_active'] for r in complete), user_mean_guards=GUARDS,
        comparisons=comparisons,
        rows=[dict(case=r['case']['key'], state=r['state'], branches=r['branches'], seconds=r['seconds'],
                   deltas_vs_C={a:{k:r['branches'][a][k]-r['branches']['C'][k] for k in METRICS} for a in (*ARMS,'S_native')},
                   gates_passed=all(v['passed'] for v in r['gates'].values())) for r in complete],
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] == 'running' for r in records),
        limits='Three previously observed image clusters. Fixed rule, one score shared by all new arms; no GT selection, scheduling, independent confirmation or curvature claim. A15 old joint classifications unchanged.')


def prepare(args):
    source = Path(args.a9_dir).resolve()
    p9 = read(source/'plan.json')
    # Validate the frozen A9 artifacts without recursively replaying any earlier experiment.
    active = parent.completed(source, p9)
    if len(active) != 6 or parent.stored_summary(source) != read(source/'summary.json'):
        raise ValueError('Need six audited complete A9 cases')
    a6 = Path(p9['a6_source']); p6 = read(a6/'plan.json')
    if p6['config'] != p9['config'] or p6['grid'] != p9['grid']:
        raise ValueError('A6/A9 protocols differ')
    identities = {}
    def bind(path, expected=None):
        path = Path(path); value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Source identity differs: '+str(path))
        identities[str(path)] = value
        return value
    bind(source/'plan.json'); bind(source/'summary.json')
    bind(a6/'plan.json', p9['identity_files'][str(a6/'plan.json')])
    sources = {}
    for c in p9['cases']:
        key = c['key']; folder = a6/key; r6 = read(folder/'run.json')
        bind(folder/'run.json', (folder/'run.sha256').read_text().strip()); bind(folder/'run.sha256')
        if r6['case'] != c or r6['status'] != 'complete' or r6['plan_sha256'] != identities[str(a6/'plan.json')] or not r6['gates'] or not all(v['passed'] for v in r6['gates'].values()):
            raise ValueError('Invalid A6 source: '+key)
        for n in ('pre_last_state.pt', 'last_opt.pt'):
            bind(folder/n, r6['artifacts_sha256'][n])
        bind(r6['measurement']['path'], r6['measurement']['sha256'])
        bind(source/key/'run.json', (source/key/'run.sha256').read_text().strip()); bind(source/key/'run.sha256')
        bind(source/key/'outputs.pt', active[key]['output_sha256'])
        bind(c['image'], c['image_sha256'])
        sources[key] = dict(a6=str(folder), a9=str(source/key), measurement=r6['measurement'])
    if {(c['image_id'],c['sampling_seed']) for c in p9['cases']} != {(i,s) for i in ('00023','00024','00025') for s in (1000,2000)}:
        raise ValueError('Fixed cohort differs')
    # Bind production/metric code already attested by A9; no need to hash unrelated old caches.
    for path, sha in p9['identity_files'].items():
        if Path(path).suffix == '.py' and Path(path).is_relative_to(PROJECTS):
            bind(path, sha)
    bind(Path(args.a13_dir)/'plan.json', (Path(args.a13_dir)/'plan.sha256').read_text().strip())
    r13 = precision.existing(Path(args.a13_dir), 14)
    if r13 is None or r13['classification'] != 'IEEE_POLICY_IMPROVES_METROLOGY_THIS_CASE':
        raise ValueError('Need completed IEEE precision context')
    bind(Path(args.a13_dir)/'t14/run.json'); bind(Path(args.a13_dir)/'t14/run.sha256')
    a15 = Path(args.a15_dir); r15 = read(a15/'run.json')
    bind(a15/'run.json', (a15/'run.sha256').read_text().strip()); bind(a15/'run.sha256')
    bind(a15/'plan.json', r15['plan_sha256'])
    if r15['status'] != 'complete' or len(r15['rows']) != 24:
        raise ValueError('Need completed A15 context')
    own = [Path(__file__), CARD, PROJECTS/'utils/test_diag_terminal_state_strength.py', PROJECTS/'run_a16_20261020.sh',
           PROJECTS/'utils/diag_clean_response_precision.py', PROJECTS/'utils/diag_terminal_polish_control.py']
    workers = {str(p):bind(p) for p in own}
    root = Path(args.root).resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (source, a6, a15, Path(args.a13_dir).resolve())):
        raise ValueError('New run overlaps parent')
    plan = dict(experiment='A16_terminal_state_strength', cases=p9['cases'], sources=sources,
        config=p9['config'], grid=p9['grid'], checkpoint_sha256=p9['checkpoint_sha256'],
        identity_files=identities, worker_identity_files=workers, ieee_policy=r13['ieee_policy'], runtime=r13['runtime'],
        fixed_coefficients=FIXED, prior_projection_fraction=.5, user_mean_guards=GUARDS,
        planned_method_nfe=0, planned_diagnostic_nfe=12, evaluation_t=0, parameter_index=0,
        source_a15_classifications=[p['classification'] for p in r15['positions']])
    if root.exists():
        if read(root/'plan.json') != plan:
            raise ValueError('Frozen A16 differs; refuse overwrite')
    else:
        root.mkdir(parents=True, exist_ok=False); WRITE(root/'plan.json', plan)
        (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
        (root/'experiment_card.md').write_text(CARD.read_text())
    print('A16 PREPARE PASS: six cases; two calls each; fixed rules and user guards frozen.', flush=True)


def load_record(root, case):
    path = root/case['key']/'run.json'
    if not path.exists():
        if path.parent.exists():
            raise ValueError('Interrupted case retained: '+case['key'])
        return None
    r = read(path)
    if (digest(path) != (path.parent/'run.sha256').read_text().strip() or r['status'] != 'complete' or r['case'] != case or
        r['plan_sha256'] != digest(root/'plan.json') or r['method_nfe'] != 0 or r['diagnostic_nfe'] != 2 or len(r['calls']) != 2 or
        not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or digest(path.parent/'outputs.pt') != r['outputs_sha256']):
        raise ValueError('Invalid/interrupted A16 retained: '+case['key'])
    return r


def run_case(root, plan, case, model, device, policy, runtime):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    folder = root/case['key']; folder.mkdir(exist_ok=False); started = time.perf_counter()
    r = dict(status='running', case=case, plan_sha256=digest(root/'plan.json'), method_nfe=0, diagnostic_nfe=0, calls=[], gates={}, policy=policy, runtime=runtime)
    def save():
        r['seconds'] = time.perf_counter()-started; WRITE(folder/'run.json', r)
    def count(_module, inputs):
        r['diagnostic_nfe'] += 1; r['calls'].append(dict(t=int(inputs[1].flatten()[0]), parameter_index=0)); save()
        if r['diagnostic_nfe'] > 2 or r['calls'][-1]['t'] != 0:
            raise RuntimeError('A16 call/time budget exceeded')
    save(); hook = model.model.register_forward_pre_hook(count)
    try:
        src = plan['sources'][case['key']]; a6 = Path(src['a6']); a9 = Path(src['a9'])
        paths = [a6/n for n in ('run.json','pre_last_state.pt','last_opt.pt')] + [a9/n for n in ('run.json','outputs.pt')] + [Path(src['measurement']['path']), Path(case['image'])]
        for path in paths:
            if digest(path) != plan['identity_files'][str(path)]:
                raise ValueError('Source changed: '+str(path))
        before = rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.no_grad():
            state = torch.load(a6/'pre_last_state.pt', map_location='cpu', weights_only=True)
            old = torch.load(a9/'outputs.pt', map_location='cpu', weights_only=True)
            original = torch.load(a6/'last_opt.pt', map_location='cpu', weights_only=True)
            c = old['C'].to(device)
            if c.shape != (1,3,256,256) or c.dtype != torch.float32 or not torch.isfinite(c).all() or relative_error(c.cpu(), original) > 1e-6 or state['tau'].tolist() != plan['grid']:
                raise ValueError('Invalid original terminal state')
            operator = unit_operator(TransposeMode(get_operator('super_resolution', device=device, scale_factor=4, noise_sigma=0.), 'legacy_bicubic').to(device)).to(device)
            z = ZAPS(model, operator, img_size=256, **plan['config']); z.tau = state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device)); z.D.copy_(state['D'].to(device))
            y = torch.load(src['measurement']['path'], map_location='cpu', weights_only=True)['y'].to(device)
            with model_rng_guard(model.model, device), patch('torch.randn_like', side_effect=RuntimeError('Unexpected sampling draw')):
                fields, repeats = parent.polish_pair(z, y, c, evaluate_field)
            f = fields[0]; clean, q = f['clean'], f['q']; h = operator.H(clean)
            residual = y-h; action = operator.H(clean+q)-h
            state_readout = strength((clean-c).cpu().numpy(), q.cpu().numpy(), residual.cpu().numpy(), action.cpu().numpy())
            outputs = {a:(clean+alpha*q).cpu() for a,alpha in FIXED.items()}
            outputs.update(L=(clean+state_readout['alpha_L']*q).cpu(), A=(clean+state_readout['alpha']*q).cpu(), C=c.cpu(), S_native=old['S'])
            checks = dict(clean_repeat=relative_error(clean, fields[1]['clean']), q_repeat=relative_error(q, fields[1]['q']),
                          output_repeat=relative_error(repeats[0], repeats[1]), core_F1=relative_error(outputs['F1'], repeats[0]))
            if any(not math.isfinite(v) or v > 1e-7 for v in checks.values()) or any(not torch.isfinite(v).all() for v in outputs.values()):
                raise RuntimeError('Repeat/core/finiteness gate failed')
            if not (torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):
                raise RuntimeError('Original parameters/grid changed')
            # All candidates frozen before GT access. This is one score reused, not per-arm inference.
            tmp = folder/'outputs.pt.tmp'
            torch.save(dict(outputs=outputs, fields=dict(clean=clean.cpu(), q=q.cpu()), repeat=repeats[1]), tmp)
            tmp.replace(folder/'outputs.pt'); r['outputs_sha256'] = digest(folder/'outputs.pt')
            gt = load_image_as_tensor(case['image']).cpu()
            def metrics(x):
                return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)), **compute_all_metrics(x,gt,lpips_net='alex'))
            branches = {a:metrics(x) for a,x in outputs.items()}; repeated = metrics(repeats[1])
            source_metrics = read(a9/'run.json')['branches']
            drift = {k:max(abs(branches[a][k]-source_metrics[b][k]) for a,b in (('C','C'),('S_native','S'))) for k in METRICS}
            if any(drift[k] > v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001),('loss',1e-7+2e-5*abs(branches['C']['loss'])))):
                raise RuntimeError('Historical metric convention drift')
            floors = {k:max(drift[k], abs(repeated[k]-branches['F1'][k])) for k in METRICS}
            affine_checks = {}
            for a in ARMS:
                alpha = FIXED[a] if a in FIXED else state_readout['alpha_L'] if a == 'L' else state_readout['alpha']
                predicted = float((residual.double()-alpha*action.double()).square().mean()/(2*.05**2))
                error = abs(predicted-branches[a]['loss']); affine_checks[a] = error
                if error > 1e-7+2e-5*abs(branches[a]['loss']):
                    raise RuntimeError('Affine operator loss model failed')
            r.update(state=state_readout, branches=branches, floors=floors)
        if not same_rng(before, rng_state(device)) or r['diagnostic_nfe'] != 2 or precision.precision_state(torch) != policy:
            raise RuntimeError('RNG/precision/call-budget gate failed')
        r['gates'] = dict(repeat_core=dict(passed=True, errors=checks), affine_loss=dict(passed=True, absolute_errors=affine_checks),
            historical_metrics=dict(passed=True, drift=drift), parameters_rng_budget=dict(passed=True))
        r['status'] = 'complete'; save(); (folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')
        print('A16 CASE:',case['key'],'alpha=',state_readout['alpha'],'guard=',state_readout['guard_active'],flush=True)
    except BaseException as error:
        r.update(status='failed', error=repr(error)); save(); raise
    finally:
        hook.remove()


def run(args):
    import fcntl
    import torch
    root = Path(args.root).resolve(); plan = read(root/'plan.json')
    if digest(root/'plan.json') != (root/'plan.sha256').read_text().strip():
        raise ValueError('Plan changed')
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo = [c for i,c in enumerate(sorted(plan['cases'], key=lambda c:c['key'])) if i%2 == args.shard and load_record(root,c) is None]
        if not todo:
            return
        for path,sha in plan['worker_identity_files'].items():
            if digest(path) != sha:
                raise ValueError('Worker code changed')
        parent.coupled.archive.recovery.require_space(root); torch.set_num_threads(2)
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'), TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != plan['runtime']:
            raise ValueError('Runtime differs from A13 context; stop before new calls')
        saved = precision.precision_state(torch)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), precision.ieee(torch) as policy:
            if policy != plan['ieee_policy']:
                raise ValueError('IEEE policy differs')
            model = parent.coupled.archive.recovery.load_model(plan,args.device)
            for case in todo:
                run_case(root,plan,case,model,args.device,policy,runtime)
        if precision.precision_state(torch) != saved:
            raise RuntimeError('Precision not restored')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--a9-dir', default='/data0/lzy/ZAPS-results/diag_terminal_polish_control/A9_20261020')
    parser.add_argument('--a13-dir', default=precision.ROOT)
    parser.add_argument('--a15-dir', default='/data0/lzy/ZAPS-results/diag_ieee_response_value/A15_20261020')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--shard', type=int, choices=(0,1))
    action.add_argument('--summary', action='store_true'); action.add_argument('--archive', action='store_true')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.prepare:
        prepare(args)
    elif args.shard is not None:
        run(args)
    else:
        plan = read(root/'plan.json')
        if digest(root/'plan.json') != (root/'plan.sha256').read_text().strip():
            raise ValueError('Plan changed')
        for path,sha in plan['worker_identity_files'].items():
            if digest(path) != sha:
                raise ValueError('Analysis code/card changed')
        records = []
        for c in plan['cases']:
            p = root/c['key']/'run.json'
            if p.exists():
                r = read(p)
                if r['status'] == 'complete':
                    load_record(root,c)
                records.append(r)
        result = summarize(records); WRITE(root/'summary.json',result)
        print('A16 Summary:',json.dumps(result),flush=True)
        if args.archive:
            from utils.archive_refinement_confirmation import export_and_commit
            bundle,commit = export_and_commit({'A16':str(root)})
            print('Archive:',bundle,'Results commit:',commit,flush=True)
        print('Records:',root,flush=True)


if __name__ == '__main__':
    main()
