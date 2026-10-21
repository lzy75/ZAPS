"""A28 multi-image state/action identification; cached states, 2736 diagnostic forwards."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_equal_nfe_path_opportunity as parent
from utils import diag_fixed_count_state_positions as fields
from utils import state_action_response as rules
from utils.diag_state_trajectory import digest

ROOT = '/data0/lzy/ZAPS-results/diag_state_action_response/A28_20261021'
CARD = PROJECTS/'experiment_cards/A28_state_action_response_20261021.md'
WRITE = parent.WRITE
ATOL, RTOL = 2./256, .01


def read(path):
    return json.loads(Path(path).read_text())


def signed(path, record):
    WRITE(path, record)
    path.with_suffix('.sha256').write_text(digest(path)+'\n')


def load_signed(path):
    if digest(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Record SHA changed: '+str(path))
    return read(path)


def check_plan(root):
    plan = load_signed(root/'plan.json')
    for path, sha in plan['worker_identity_files'].items():
        if digest(path) != sha:
            raise ValueError('Frozen worker changed: '+path)
    return plan


def prepare(args):
    root, source = Path(args.root).resolve(), Path(args.a22_dir).resolve()
    if root.exists():
        p = check_plan(root)
        if p['source_A22'] != str(source):
            raise ValueError('Existing source differs')
        for path, sha in p['identity_files'].items():
            if digest(path) != sha:
                raise ValueError('Bound source changed: '+path)
        print('A28 frozen plan reused; complete points reused, partial attempts retained.', flush=True)
        return
    previous = load_signed(source/'plan.json'); load_signed(source/'summary.json')
    if {(s['case']['image_id'], s['case']['sampling_seed']) for s in previous['cases']} != {
            (f'{i:05d}', seed) for i in range(20, 26) for seed in (1000, 2000)}:
        raise ValueError('Need original six-image/two-seed source cohort')
    identities = {}
    def bind(path, expected=None):
        path = Path(path); sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Source SHA differs: '+str(path))
        identities[str(path)] = sha
    cases = previous['cases']
    for spec in cases:
        parent.verify_case_sources(previous, spec)
        if parent.load_record(source, spec) is None:
            raise ValueError('A22 source incomplete')
        src = Path(spec['source']); native = read(src/'run.json')
        if native['status'] != 'complete' or not native['gates'] or not all(g['passed'] for g in native['gates'].values()):
            raise ValueError('Native source gates failed')
        if len(native['rows']) != 30 or [r['t'] for r in native['rows']] != list(reversed(previous['grid'])):
            raise ValueError('Native source grid differs')
        for name in ('run.json', 'trajectory.pt', 'pre_last_state.pt', 'last_opt.pt'):
            bind(src/name, previous['identity_files'][str(src/name)])
        for path in (spec['measurement']['path'], spec['case']['image']):
            bind(path, previous['identity_files'][str(Path(path))])
        for name in ('run.json', 'run.sha256'):
            bind(source/spec['case']['key']/name)
    if any(root.is_relative_to(Path(s['source']).resolve()) or Path(s['source']).resolve().is_relative_to(root) for s in cases):
        raise ValueError('Output overlaps historical source')
    for name in ('plan.json', 'plan.sha256', 'summary.json', 'summary.sha256'):
        bind(source/name)
    own = [Path(__file__), Path(rules.__file__), Path(fields.__file__), Path(fields.policy.__file__),
           Path(parent.coupled.__file__), Path(parent.coupled.contract.__file__), Path(parent.precision.__file__),
           PROJECTS/'modules/zaps_algorithm.py', PROJECTS/'modules/diffusion_model.py',
           PROJECTS/'utils/test_state_action_response.py', PROJECTS/'run_a28_20261021.sh', CARD]
    worker_files = dict(previous['worker_identity_files'])
    for path in own:
        bind(path); worker_files[str(path)] = identities[str(path)]
    for path, sha in worker_files.items():
        if digest(path) != sha:
            raise ValueError('Source implementation differs: '+path)
    plan = dict(experiment='A28_state_action_response', source_A22=str(source), cases=cases,
                config=previous['config'], grid=previous['grid'], runtime=previous['runtime'],
                native_policy=previous['native_policy'], checkpoint_sha256=previous['checkpoint_sha256'],
                steps=list(rules.STEPS), spans=list(rules.SPANS), gammas=list(rules.GAMMAS), etas=list(rules.ETAS),
                atol=ATOL, rtol=RTOL, identity_files=identities, worker_identity_files=worker_files,
                planned_method_nfe=0, planned_diagnostic_nfe=12*6*rules.CALLS_PER_POINT,
                calls_per_point=rules.CALLS_PER_POINT,
                scope='Reused six development images. Fixed-state IEEE metrology, not a new sampling policy or final reconstruction.')
    root.parent.mkdir(parents=True, exist_ok=True)
    parent.coupled.archive.recovery.require_space(root.parent)
    root.mkdir(exist_ok=False); signed(root/'plan.json', plan)
    (root/'experiment_card.md').write_text(CARD.read_text())
    print('A28 PREPARE PASS: 12 cases x6 states x38 calls =2736 diagnostic NFE; method NFE=0.', flush=True)


def rms(x):
    return float(x.double().square().mean().sqrt())


def normalized_error(a, b):
    return rms((a.double()-b.double())/(ATOL+RTOL*a.double().abs().maximum(b.double().abs())))


def metrics(z, y, clean, gt):
    residual = (y-z.A.H(clean)).double()/.05
    mse = float(((clean.double()-gt.double())/2).square().mean())
    if mse <= 0:
        raise RuntimeError('Perfect or invalid diagnostic MSE; do not emit infinite PSNR')
    return dict(loss=float(residual.square().mean()/2), psnr=-10*math.log10(mse))


def fixed_probe(x, q, direction, amplitude, evaluate, center_floor):
    """Central fixed-t probes. Zero/tiny responses remain unresolved, never amplified."""
    import torch
    size = rms(direction); valid = size > 1e-12 and rms(q) > 1e-12
    v = direction/size if valid else torch.zeros_like(x)
    pairs = []
    for eta in rules.ETAS:
        epsilon = max(eta*amplitude, 1e-8)
        plus = evaluate(x+epsilon*v)['q']; minus = evaluate(x-epsilon*v)['q']
        rp, rm = rms(plus), rms(minus)
        resolved = valid and min(rp, rm) > 1e-12
        Ldir = float((plus.double()/plus.double().norm()-minus.double()/minus.double().norm()).norm())/(2*epsilon) if resolved else 0.
        Lmag = abs(math.log(rp/rm))/(2*epsilon) if resolved else 0.
        signal = rms(plus-minus)
        floor = max(10*center_floor, 32*torch.finfo(x.dtype).eps*max(rms(q), 1e-12))
        pairs.append(dict(eta=eta, epsilon=epsilon, L_dir=Ldir, L_mag=Lmag,
                          signal_rms=signal, floor_rms=floor, resolved=bool(resolved and signal > floor)))
    spread = {kind: rules.scale_agreement([p['L_'+kind] for p in pairs]) for kind in ('dir', 'mag')}
    usable = all(p['resolved'] for p in pairs) and max(spread.values()) <= .10
    return dict(usable=bool(usable), pairs=pairs, spread=spread,
                L_dir=float(np.mean([p['L_dir'] for p in pairs])), L_mag=float(np.mean([p['L_mag'] for p in pairs])))


def diagnose_point(z, y, x, positions, step, cache_clean, cache_q, noise, auxiliary, history_q, gt, evaluate):
    """38 field calls; all endpoint decisions frozen before offline GT readout."""
    import torch
    from utils.diag_zaps_trace_audit import relative_error
    t, nominal_s = positions.nominal[step:step+2]
    native = evaluate(x, t, 'native_center')
    parity = {k: relative_error(a, b) for k, a, b in
              (('clean', native['clean'], cache_clean), ('q', native['q'], cache_q))}
    if max(parity.values()) > 1e-6 or not all(math.isfinite(v) for v in parity.values()):
        raise RuntimeError('Native cached center parity failed: '+str(parity))
    actions, refinements, endpoints = [], [], []
    with parent.precision.ieee(torch) as ieee_policy:
        center = evaluate(x, t, 'ieee_center'); repeat = evaluate(x, t, 'ieee_repeat')
        repeat_errors = {name: relative_error(center[name], repeat[name]) for name in ('clean', 'q')}
        if max(repeat_errors.values()) > 1e-6:
            raise RuntimeError('IEEE center repetition failed')
        floor = rms(center['q']-repeat['q'])
        zero = torch.zeros_like(x); clean, q = center['clean'], center['q']
        move = parent.coupled.posterior(x, clean, t, nominal_s, z.dm.alphas_cumprod, zero)+q-x
        probes = {}
        for name, direction, amplitude in (('q', q, rms(q)), ('travel', move, rms(move))):
            def probe_eval(value, n=name):
                return evaluate(value, t, n+'_probe')
            probes[name] = fixed_probe(x, q, direction, amplitude, probe_eval, floor)
        for span in rules.next_times(positions.u, t, nominal_s):
            s, factor = span['s'], span['factor']
            spec = parent.coupled.interval_spec(z.dm.alphas_cumprod.cpu().numpy(), t, s)
            mean_base = parent.coupled.posterior(x, clean, t, s, z.dm.alphas_cumprod, zero)+q
            stochastic_base = parent.coupled.posterior(x, clean, t, s, z.dm.alphas_cumprod, noise)+q
            # Both substeps use frozen source-t parameters and transported dose.
            mid_mean, frozen_mean, _, _ = parent.coupled.frozen_bridge(x, clean, q, zero, zero, spec, z.dm.alphas_cumprod)
            mid_stochastic, frozen_stochastic, _, _ = parent.coupled.frozen_bridge(x, clean, q, noise, auxiliary, spec, z.dm.alphas_cumprod)
            null = max(relative_error(mean_base, frozen_mean), relative_error(stochastic_base, frozen_stochastic))
            if not math.isfinite(null) or null > 1e-6:
                raise RuntimeError('Frozen-field subdivision contract failed')
            mid_f = evaluate(mid_mean, spec['m'], 'mean_mid')
            qmid = parent.coupled.frozen_guidance(z, y, mid_f['clean'], t, 29-step)
            fine_mean = parent.coupled.refined_endpoint(mean_base, clean, q, mid_f['clean'], qmid, spec)
            for gamma in rules.GAMMAS:
                mu = mean_base if gamma == 1. else mean_base+(gamma-1)*q
                value = stochastic_base if gamma == 1. else stochastic_base+(gamma-1)*q
                fmean = evaluate(mu, s, 'candidate_mean')
                fnoise = evaluate(value, s, 'candidate_stochastic')
                actions.append(dict(**span, gamma=gamma, mean=None, stochastic=None))
                endpoints.append((actions[-1], fmean['clean'], fnoise['clean']))
            mid_f = evaluate(mid_stochastic, spec['m'], 'stochastic_mid')
            qmid = parent.coupled.frozen_guidance(z, y, mid_f['clean'], t, 29-step)
            fine_stochastic = parent.coupled.refined_endpoint(stochastic_base, clean, q, mid_f['clean'], qmid, spec)
            ffine = evaluate(fine_stochastic, s, 'stochastic_fine_endpoint')
            refinements.append(dict(**span, m=spec['m'], frozen_error=null,
                                    mean_error=normalized_error(fine_mean, mean_base),
                                    stochastic_error=normalized_error(fine_stochastic, stochastic_base),
                                    stochastic_fine=None, guidance_factor=spec['local_guidance_factor']))
            endpoints.append((refinements[-1], None, ffine['clean']))
        # No GT metric is read until every action and query at this state is immutable.
        for row, a, b in endpoints:
            if a is not None:
                row['mean'] = metrics(z, y, a, gt); row['stochastic'] = metrics(z, y, b, gt)
            else:
                row['stochastic_fine'] = metrics(z, y, b, gt)
        current_R = center['R']
        turn = None
        # History stays entirely in the archived native precision convention.
        if min(rms(v) for v in (native['q'], history_q)) > 1e-12:
            nq = native['q'].double()
            cosine = float((nq*history_q.double()).sum()/(nq.norm()*history_q.double().norm()))
            turn = math.acos(max(-1., min(1., cosine)))/(positions.u[t]-positions.u[positions.nominal[step-1]])
    return dict(step=step, t=t, nominal_s=nominal_s, u=float(positions.u[t]), R=current_R,
                q_rms=rms(q), move_rms=rms(move), clip=center['clip_fraction'], history_turn=turn,
                probes=probes, actions=actions, refinements=refinements,
                gates=dict(native_parity=dict(passed=True, errors=parity), ieee_repeat=dict(passed=True, errors=repeat_errors),
                           subdivision=dict(passed=True), GT_after_queries=dict(passed=True)),
                ieee_policy=ieee_policy, ieee_vs_native={name: relative_error(center[name], native[name]) for name in ('clean', 'q')})


def complete_point(folder, plan_sha, case, step):
    found = []
    for path in sorted(folder.glob('attempt_*/run.json')):
        r = read(path)
        if r.get('status') != 'complete':
            continue
        r = load_signed(path)
        if (r['plan_sha256'] != plan_sha or r['case'] != case or r['step'] != step or
                r['diagnostic_nfe'] != rules.CALLS_PER_POINT or len(r['calls']) != rules.CALLS_PER_POINT or
                not r['point']['gates'] or not all(g['passed'] for g in r['point']['gates'].values())):
            raise ValueError('Completed point contract differs: '+str(path))
        found.append(r)
    if len(found) > 1:
        raise ValueError('Duplicate completed point; refuse implicit output selection')
    return found[0] if found else None


def run_point(root, plan, spec, step, z, positions, cache, noises, y, gt, model, device):
    import torch
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    folder = root/spec['case']['key']/f'k{step:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    plan_sha = digest(root/'plan.json')
    if complete_point(folder, plan_sha, spec['case'], step) is not None:
        return
    # Partial attempts are retained and included in total cost, not overwritten.
    index = len(list(folder.glob('attempt_*')))
    attempt = folder/f'attempt_{index:03d}'; attempt.mkdir(exist_ok=False)
    path = attempt/'run.json'; start = time.perf_counter(); context = {}
    record = dict(status='running', case=spec['case'], step=step, plan_sha256=plan_sha,
                  method_nfe=0, diagnostic_nfe=0, calls=[])
    def save():
        record['seconds'] = time.perf_counter()-start; WRITE(path, record)
    def count(_module, inputs):
        record['diagnostic_nfe'] += 1
        actual = int(inputs[1].flatten()[0]); record['calls'].append(dict(**context, t=actual)); save()
        if actual != context['expected_t'] or record['diagnostic_nfe'] > rules.CALLS_PER_POINT:
            raise RuntimeError('Actual query budget/time differs')
    def evaluate(x, t, kind):
        context.update(expected_t=int(t), kind=kind)
        return fields.field(z, y, x, int(t), positions)
    save(); hook = model.model.register_forward_pre_hook(count)
    before = rng_state(device)
    try:
        with torch.no_grad(), model_rng_guard(model.model, device):
            gen = torch.Generator(device=device).manual_seed(28261021+10000*int(spec['case']['image_id'])+10*step+spec['case']['sampling_seed'])
            auxiliary = torch.randn(cache['latent_inputs'][step:step+1].shape, device=device, generator=gen)
            point = diagnose_point(z, y, cache['latent_inputs'][step:step+1].to(device), positions, step,
                                   cache['clean'][step:step+1].to(device), cache['guidance'][step:step+1].to(device),
                                   noises[step], auxiliary, cache['guidance'][step-1:step].to(device), gt, evaluate)
        if (record['diagnostic_nfe'] != rules.CALLS_PER_POINT or not same_rng(before, rng_state(device)) or
                parent.precision.precision_state(torch) != plan['native_policy']):
            raise RuntimeError('Budget/RNG/precision restoration failed')
        point.update(image_id=spec['case']['image_id'], seed=spec['case']['sampling_seed'])
        point['gates']['RNG_precision_budget'] = dict(passed=True)
        record.update(status='complete', point=point); save(); signed(path, record)
        print('A28 POINT:', spec['case']['key'], 't=', point['t'], 'probes usable=',
              {k: v['usable'] for k, v in point['probes'].items()}, 'calls=38', flush=True)
    except BaseException as error:
        record.update(status='failed', error=repr(error)); save(); raise
    finally:
        hook.remove()


def worker(args, root, plan):
    import fcntl
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_state_weight_scan import preserved_rng
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = [s for i, s in enumerate(sorted(plan['cases'], key=lambda s: s['case']['key'])) if (i//2)%2 == args.shard]
        if (all(complete_point(root/s['case']['key']/f'k{k:02d}', digest(root/'plan.json'), s['case'], k)
                is not None for s in specs for k in rules.STEPS)
                and all((root/s['case']['key']/'case_gate.json').exists() for s in specs)):
            print('A28 shard already complete; UNet NFE=0.', flush=True); return
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(args.device), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != plan['runtime'] or parent.precision.precision_state(torch) != plan['native_policy']:
            raise ValueError('Original runtime/precision differs; no UNet calls')
        torch.set_num_threads(2); parent.coupled.archive.recovery.require_space(root)
        with preserved_rng():
            model = parent.coupled.archive.recovery.load_model(plan, args.device)
            for spec in specs:
                parent.verify_case_sources(plan, spec)
                src = Path(spec['source']); native = read(src/'run.json')
                state = torch.load(src/'pre_last_state.pt', map_location='cpu', weights_only=True)
                cache = torch.load(src/'trajectory.pt', map_location='cpu', weights_only=True)
                original = torch.load(src/'last_opt.pt', map_location='cpu', weights_only=True)
                if state['tau'].tolist() != plan['grid'] or any(cache[n].shape != (30, 3, 256, 256) or
                        cache[n].dtype != torch.float32 or not torch.isfinite(cache[n]).all() for n in ('latent_inputs', 'clean', 'guidance')):
                    raise ValueError('Source tensor/grid contract differs')
                operator = unit_operator(TransposeMode(get_operator('super_resolution', device=args.device,
                    scale_factor=4, noise_sigma=0.), 'legacy_bicubic').to(args.device)).to(args.device)
                z = ZAPS(model, operator, img_size=256, **plan['config']); z.tau = state['tau'].to(args.device)
                with torch.no_grad():
                    z.zeta.copy_(state['zeta'].to(args.device)); z.D.copy_(state['D'].to(args.device))
                y = torch.load(spec['measurement']['path'], map_location='cpu', weights_only=True)['y'].to(args.device)
                positions = fields.policy.Positions(fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()), list(reversed(plan['grid'])))
                for k in rules.STEPS:
                    rules.next_times(positions.u, positions.nominal[k], positions.nominal[k+1])
                noises, _ = parent.coupled.replay_noises(state, cache, dict(rows=native['rows']), z, y, args.device, original)
                gt = load_image_as_tensor(spec['case']['image']).to(args.device)
                for k in rules.STEPS:
                    run_point(root, plan, spec, k, z, positions, cache, noises, y, gt, model, args.device)
                if not all(torch.equal(a.cpu(), state[n]) for a, n in ((z.tau, 'tau'), (z.zeta, 'zeta'), (z.D, 'D'))):
                    raise RuntimeError('Original parameters mutated')
                parent.verify_case_sources(plan, spec)
                signed(root/spec['case']['key']/'case_gate.json', dict(passed=True, plan_sha256=digest(root/'plan.json'),
                       source_unchanged=True, parameters_unchanged=True, points=6))
        check_plan(root)


def summarize(root, plan):
    points = []; cost = 0; partial = 0; completed_cases = 0
    for spec in plan['cases']:
        found = []
        for step in rules.STEPS:
            folder = root/spec['case']['key']/f'k{step:02d}'
            r = complete_point(folder, digest(root/'plan.json'), spec['case'], step)
            if r is not None:
                found.append(r['point'])
            for path in folder.glob('attempt_*/run.json'):
                attempt = read(path); cost += attempt['diagnostic_nfe']; partial += attempt['status'] != 'complete'
        gate = root/spec['case']['key']/'case_gate.json'
        if len(found) == 6 and gate.exists():
            g = load_signed(gate)
            if not g['passed'] or g['plan_sha256'] != digest(root/'plan.json'):
                raise ValueError('Case gate differs')
            parent.verify_case_sources(plan, spec); completed_cases += 1
        points.extend(found)
    out = dict(experiment='A28_state_action_response', completed_cases=completed_cases, planned_cases=12,
               completed_states=len(points), planned_states=72, method_nfe=0, diagnostic_nfe=cost,
               planned_diagnostic_nfe=2736, partial_attempts=partial,
               cost_is_lower_bound_if_abruptly_interrupted=bool(partial), classification='INCOMPLETE_DIAGNOSTIC',
               limits='Six reused development image clusters, leave-one-image-out diagnosis only. Fixed-state IEEE response; one coupled bridge/state. Float clean next-endpoint PSNR, not final PSNR/SSIM/LPIPS. No online policy, NFE saving or final reconstruction gain.')
    if completed_cases == 12:
        analysis = rules.analyze(points); out.update(classification=analysis['classification'], analysis=analysis, points=points)
    signed(root/'summary.json', out)
    return out


def compact(out):
    result = {k: v for k, v in out.items() if k not in ('points', 'analysis')}
    if 'analysis' in out:
        analysis = out['analysis']; result['analysis'] = {k: v for k, v in analysis.items() if k not in ('targets', 'paid_response_all_states')}
        result['analysis']['targets'] = {t: dict(mae={n: r['mae'] for n, r in v['models'].items()},
                    comparisons=v['comparisons']) for t, v in analysis['targets'].items()}
        if 'paid_response_all_states' in analysis:
            result['analysis']['paid_response_all_states'] = {t: dict(mae={n: r['mae'] for n, r in v['models'].items()},
                    comparisons=v['comparisons']) for t, v in analysis['paid_response_all_states'].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--a22-dir', default=parent.ROOT)
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--shard', type=int, choices=(0, 1))
    mode.add_argument('--summarize', action='store_true'); mode.add_argument('--archive', action='store_true')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.prepare:
        prepare(args); return
    plan = check_plan(root)
    if args.shard is not None:
        worker(args, root, plan); return
    out = summarize(root, plan)
    print('A28 Summary:', json.dumps(compact(out), ensure_ascii=False), flush=True)
    print('Records:', root, flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle, commit = export_and_commit({'A28': str(root)})
        print('Archive:', bundle, 'Results commit:', commit, flush=True)


if __name__ == '__main__':
    main()
