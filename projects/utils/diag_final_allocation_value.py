"""A30: complete A28-matched refinements to final images; 864 new diagnostic calls."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_state_action_response as source
from utils import diag_cached_stage_action_audit as audit
from utils import final_allocation_value as rules

P = source.parent
ROOT = '/data0/lzy/ZAPS-results/diag_final_allocation_value/A30_20261021'
CARD = PROJECTS/'experiment_cards/A30_final_allocation_value_20261021.md'


def prepare(args):
    root, old = Path(args.root).resolve(), Path(args.a28_dir).resolve()
    if root.exists():
        p = source.check_plan(root)
        if p['source_A28'] != str(old): raise ValueError('Different A28 source')
        for path, sha in p['identity_files'].items():
            if source.digest(path) != sha: raise ValueError('Frozen source changed: '+path)
        print('A30 frozen plan reused; completed samples reused.', flush=True); return
    p28 = source.check_plan(old); points, identities = audit.audit(old)
    identities.update(p28['identity_files'])
    for spec in p28['cases']:
        P.verify_case_sources(p28, spec)
        if root.is_relative_to(Path(spec['source']).resolve()): raise ValueError('Output overlaps source')
    if root.is_relative_to(old) or old.is_relative_to(root): raise ValueError('Output overlaps A28')
    own = [Path(__file__), Path(rules.__file__), Path(audit.__file__), Path(source.__file__),
           CARD, PROJECTS/'utils/test_final_allocation_value.py', PROJECTS/'run_a30_20261021.sh']
    workers = dict(p28['worker_identity_files'])
    for path in own:
        identities[str(path)] = source.digest(path); workers[str(path)] = identities[str(path)]
    for path, sha in identities.items():
        if source.digest(path) != sha: raise ValueError('Input/implementation changed: '+path)
    fixed_F = {s['case']['key']: source.read(Path(p28['source_A22'])/s['case']['key']/'run.json')['branches']['T319_G2']
               for s in p28['cases']}
    plan = dict(experiment='A30_final_allocation_value', source_A28=str(old), cases=p28['cases'], fixed_F_metrics=fixed_F,
        config=p28['config'], grid=p28['grid'], runtime=p28['runtime'], native_policy=p28['native_policy'],
        checkpoint_sha256=p28['checkpoint_sha256'], identity_files=identities, worker_identity_files=workers,
        steps=list(rules.STEPS), planned_method_nfe=0, planned_diagnostic_nfe=12*rules.CALLS_PER_CASE,
        counterfactual_trajectory_nfe=301, baseline_reference_nfe=300, online_sensor_budget_included=False,
        deployment_state_policy_nfe=None, original_points=points)
    root.parent.mkdir(parents=True, exist_ok=True); P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir(exist_ok=False); source.signed(root/'plan.json', plan)
    (root/'experiment_card.md').write_text(CARD.read_text())
    print('A30 PREPARE PASS: 72 final outcomes; 864 new diagnostic calls; all candidate trajectories=301 calls. No deployed state policy.', flush=True)


def refine_and_continue(z, y, cache, positions, noises, step, auxiliary, evaluate):
    """Reuse current field; evaluate one midpoint, then every remaining ORIGINAL step."""
    import torch
    from utils.diag_zaps_trace_audit import relative_error
    device = y.device; t, s = positions.nominal[step:step+2]
    x = cache['latent_inputs'][step:step+1].to(device)
    clean = cache['clean'][step:step+1].to(device); q = cache['guidance'][step:step+1].to(device)
    coarse = cache['latent_inputs'][step+1:step+2].to(device)
    spec = P.coupled.interval_spec(z.dm.alphas_cumprod.cpu().numpy(), t, s)
    mid, frozen, _, _ = P.coupled.frozen_bridge(x, clean, q, noises[step], auxiliary, spec, z.dm.alphas_cumprod)
    error = relative_error(coarse, frozen)
    if not math.isfinite(error) or error > 1e-6: raise RuntimeError('Frozen-field/cached endpoint mismatch')
    middle = evaluate(mid, spec['m'])
    middle_q = P.coupled.frozen_guidance(z, y, middle['clean'], t, 29-step)
    x = P.coupled.refined_endpoint(coarse, clean, q, middle['clean'], middle_q, spec)
    for j in range(step+1, 30):
        time_ = positions.nominal[j]; f = evaluate(x, time_)
        if j == 29:
            x = P.terminal.terminal_output(z, x, f)
        else:
            x = P.coupled.posterior(x, f['clean'], time_, positions.nominal[j+1], z.dm.alphas_cumprod, noises[j])+f['q']
        if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite final continuation')
    return x.cpu(), dict(t=t, m=spec['m'], s=s, frozen_error=error, dose=spec['local_guidance_factor'])


def saved_point(folder, root, case, step, require_complete=True):
    records = []
    for path in sorted(folder.glob('attempt_*/run.json')):
        r = source.read(path)
        if r['status'] not in ('sampled', 'complete'): continue
        r = source.load_signed(path)
        if (r['plan_sha256'] != source.digest(root/'plan.json') or r['case'] != case or r['step'] != step or
                r['diagnostic_nfe'] != 30-step or len(r['calls']) != 30-step or
                [v['t'] for v in r['calls']] != r['expected_call_times'] or
                source.digest(path.parent/'output.pt') != r['output_sha256'] or
                not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('Completed sampling contract differs: '+str(path))
        records.append((path, r))
    if len(records) > 1: raise ValueError('Duplicate sampled output; no implicit best-output choice')
    if not records or (require_complete and records[0][1]['status'] != 'complete'): return None
    return records[0]


def run_point(root, plan, spec, step, z, positions, cache, noises, y, baseline_metrics, model, device):
    import torch
    from utils.diag_state_weight_scan import preserved_rng
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    folder = root/spec['case']['key']/f'k{step:02d}'; folder.mkdir(parents=True, exist_ok=True)
    item = saved_point(folder, root, spec['case'], step, require_complete=False)
    if item is not None and item[1]['status'] == 'complete': return
    with preserved_rng():
        if item is None:
            attempt = folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}'; attempt.mkdir(exist_ok=False)
            path = attempt/'run.json'; started = time.perf_counter(); context = {}
            t, s = positions.nominal[step:step+2]
            bridge = P.coupled.interval_spec(z.dm.alphas_cumprod.cpu().numpy(), t, s)
            expected = [bridge['m']]+positions.nominal[step+1:]
            r = dict(status='running', case=spec['case'], step=step, t=t, m=bridge['m'], s=s,
                     plan_sha256=source.digest(root/'plan.json'), method_nfe=0, diagnostic_nfe=0,
                     counterfactual_trajectory_nfe=301, online_sensor_budget_included=False,
                     calls=[], expected_call_times=expected, gates={})
            def save():
                r['sampling_seconds'] = time.perf_counter()-started; source.WRITE(path, r)
            def count(_module, inputs):
                r['diagnostic_nfe'] += 1; actual = int(inputs[1].flatten()[0])
                r['calls'].append(dict(t=actual, expected_t=context['t'])); save()
                if (r['diagnostic_nfe'] > 30-step or actual != context['t'] or
                        actual != expected[r['diagnostic_nfe']-1]): raise RuntimeError('Actual query sequence/budget mismatch')
            def evaluate(value, time_):
                context['t'] = int(time_)
                return source.fields.field(z, y, value, int(time_), positions)
            save(); hook = model.model.register_forward_pre_hook(count)
            try:
                before = rng_state(device)
                with torch.no_grad(), model_rng_guard(model.model, device):
                    seed = 28261021+10000*int(spec['case']['image_id'])+10*step+spec['case']['sampling_seed']
                    gen = torch.Generator(device=device).manual_seed(seed)
                    auxiliary = torch.randn(cache['latent_inputs'][step:step+1].shape, device=device, generator=gen)
                    output, info = refine_and_continue(z, y, cache, positions, noises, step, auxiliary, evaluate)
                if (r['diagnostic_nfe'] != 30-step or not same_rng(before, rng_state(device)) or
                        P.precision.precision_state(torch) != plan['native_policy']):
                    raise RuntimeError('Budget, sampling RNG or native precision changed')
                temporary = attempt/'output.pt.tmp'; torch.save(output, temporary); temporary.replace(attempt/'output.pt')
                r.update(status='sampled', **info, output_sha256=source.digest(attempt/'output.pt'),
                         gates=dict(original_bridge=dict(passed=True), sampling_RNG_precision_budget=dict(passed=True),
                                    output_before_GT=dict(passed=True)))
                source.signed(path, r)
            except BaseException as error:
                r.update(status='failed', error=repr(error)); save(); raise
            finally:
                hook.remove()
        else:
            path, r = item
            output = torch.load(path.parent/'output.pt', map_location='cpu', weights_only=True)
        try:
            # Final output is immutable before GT evaluation, including retries.
            gt = load_image_as_tensor(spec['case']['image']).cpu()
            loss = float((y-z.A.H(output.to(device))).double().square().mean()/(2*.05**2))
            r['metrics'] = dict(loss=loss, **compute_all_metrics(output, gt, lpips_net='alex'))
            r['baseline'] = baseline_metrics
            r['fixed_F_reference'] = plan['fixed_F_metrics'][spec['case']['key']]
            if not all(math.isfinite(v) for v in r['metrics'].values()): raise RuntimeError('Nonfinite final metrics')
            r['status'] = 'complete'; r.pop('metric_error', None); source.signed(path, r)
        except BaseException as error:
            r.update(status='sampled', metric_error=repr(error)); source.signed(path, r); raise
    print('A30 FINAL:', spec['case']['key'], 't=', r['t'], 'PSNR delta=', r['metrics']['psnr']-r['baseline']['psnr'],
          'diagnostic calls=', r['diagnostic_nfe'], flush=True)


def worker(args, root, plan):
    import fcntl
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_state_weight_scan import preserved_rng
    from utils.metrics import compute_all_metrics
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = [s for i, s in enumerate(sorted(plan['cases'], key=lambda s: s['case']['key'])) if (i//2)%2 == args.shard]
        if (all(saved_point(root/s['case']['key']/f'k{k:02d}', root, s['case'], k) is not None for s in specs for k in rules.STEPS)
                and all((root/s['case']['key']/'case_gate.json').exists() for s in specs)):
            print('A30 shard complete; new UNet calls=0.', flush=True); return
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != plan['runtime'] or P.precision.precision_state(torch) != plan['native_policy']:
            raise ValueError('Native runtime/precision differs; no UNet calls')
        torch.set_num_threads(2); P.coupled.archive.recovery.require_space(root)
        with preserved_rng():
            model = P.coupled.archive.recovery.load_model(plan, args.device)
            for spec in specs:
                P.verify_case_sources(plan, spec); src = Path(spec['source']); native = source.read(src/'run.json')
                state = torch.load(src/'pre_last_state.pt', map_location='cpu', weights_only=True)
                cache = torch.load(src/'trajectory.pt', map_location='cpu', weights_only=True)
                original = torch.load(src/'last_opt.pt', map_location='cpu', weights_only=True)
                if state['tau'].tolist() != plan['grid'] or any(cache[n].shape != (30, 3, 256, 256) or
                        cache[n].dtype != torch.float32 or not torch.isfinite(cache[n]).all() for n in ('latent_inputs', 'clean', 'guidance')):
                    raise ValueError('Cache tensor/grid identity differs')
                operator = unit_operator(TransposeMode(get_operator('super_resolution', device=args.device,
                    scale_factor=4, noise_sigma=0.), 'legacy_bicubic').to(args.device)).to(args.device)
                z = ZAPS(model, operator, img_size=256, **plan['config']); z.tau = state['tau'].to(args.device)
                with torch.no_grad():
                    z.zeta.copy_(state['zeta'].to(args.device)); z.D.copy_(state['D'].to(args.device))
                y = torch.load(spec['measurement']['path'], map_location='cpu', weights_only=True)['y'].to(args.device)
                positions = source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()), list(reversed(plan['grid'])))
                noises, _ = P.coupled.replay_noises(state, cache, dict(rows=native['rows']), z, y, args.device, original)
                gt = load_image_as_tensor(spec['case']['image']).cpu()
                baseline = dict(loss=float((y-operator.H(original.to(args.device))).double().square().mean()/(2*.05**2)),
                                **compute_all_metrics(original, gt, lpips_net='alex'))
                reference = native.get('final_metrics', native.get('baseline'))
                if reference is None or any(abs(baseline[m]-reference[m]) > tol for m, tol in (('psnr', .01), ('ssim', .001), ('lpips', .001))):
                    raise RuntimeError('Original final metric protocol differs')
                for step in rules.STEPS:
                    run_point(root, plan, spec, step, z, positions, cache, noises, y, baseline, model, args.device)
                if not all(torch.equal(a.cpu(), state[n]) for a, n in ((z.tau, 'tau'), (z.zeta, 'zeta'), (z.D, 'D'))):
                    raise RuntimeError('Frozen learned parameters changed')
                P.verify_case_sources(plan, spec)
                source.signed(root/spec['case']['key']/'case_gate.json', dict(passed=True, plan_sha256=source.digest(root/'plan.json'),
                    point_records={str(k): source.digest(saved_point(root/spec['case']['key']/f'k{k:02d}', root, spec['case'], k)[0]) for k in rules.STEPS}))
        source.check_plan(root)


def summarize(root, plan):
    rows = []; cost = 0; partial = 0; cases = 0
    for spec in plan['cases']:
        records = []
        for k in rules.STEPS:
            folder = root/spec['case']['key']/f'k{k:02d}'
            item = saved_point(folder, root, spec['case'], k)
            if item is not None: records.append(item[1])
            for path in folder.glob('attempt_*/run.json'):
                r = source.read(path); cost += r['diagnostic_nfe']; partial += r['status'] not in ('complete', 'sampled')
        gate_path = root/spec['case']['key']/'case_gate.json'
        if len(records) == 6 and gate_path.exists():
            gate = source.load_signed(gate_path)
            if not gate['passed'] or gate['plan_sha256'] != source.digest(root/'plan.json'): raise ValueError('Case gate differs')
            for k in rules.STEPS:
                path, _ = saved_point(root/spec['case']['key']/f'k{k:02d}', root, spec['case'], k)
                if source.digest(path) != gate['point_records'][str(k)]: raise ValueError('Case point changed')
            P.verify_case_sources(plan, spec); cases += 1
        rows.extend(records)
    out = dict(experiment='A30_final_allocation_value', completed_cases=cases, planned_cases=12,
        completed_states=len(rows), planned_states=72, method_nfe=0, diagnostic_nfe=cost, planned_diagnostic_nfe=864,
        partial_attempts=partial, cost_is_lower_bound_if_abruptly_interrupted=bool(partial),
        counterfactual_trajectory_nfe=301, baseline_reference_nfe=300, online_sensor_budget_included=False,
        classification='INCOMPLETE_FINAL_ALLOCATION_DIAGNOSTIC')
    if cases == 12:
        result = rules.analyze(rows, plan['original_points']); out.update(classification=result['classification'], analysis=result)
    source.signed(root/'summary.json', out); return out


def compact(out):
    result = {k: v for k, v in out.items() if k != 'analysis'}
    if 'analysis' in out:
        a = out['analysis']; result['analysis'] = {k: v for k, v in a.items() if k != 'targets'}
        result['analysis']['targets'] = {t: dict(mae={n: r['mae'] for n, r in v['models'].items()}, comparisons=v['comparisons'])
                                          for t, v in a['targets'].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--a28-dir', default=source.ROOT)
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--shard', type=int, choices=(0, 1))
    mode.add_argument('--archive', action='store_true'); mode.add_argument('--summary', action='store_true')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.prepare: prepare(args); return
    plan = source.check_plan(root)
    if args.shard is not None: worker(args, root, plan); return
    out = summarize(root, plan); print('A30 Summary:', json.dumps(compact(out), ensure_ascii=False), flush=True)
    print('Records:', root, flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle, commit = export_and_commit({'A30': str(root)})
        print('Archive:', bundle, 'Results commit:', commit, flush=True)


if __name__ == '__main__':
    main()
