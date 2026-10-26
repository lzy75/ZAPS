"""A55r2: isolate current pullback direction, same fixed-grid target-norm law."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import sys
import time
from statistics import mean

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))
from utils import diag_norm_gain_value as previous
from utils import current_direction_control as rules
from utils import diag_current_direction_value as original
from utils.linear_adjoint_autograd_guard import operator_only_autograd

engine, parent, source = previous.engine, previous.parent, previous.source
write, signed = previous.write, previous.signed
ROOT = '/data0/lzy/ZAPS-results/diag_current_direction_value_r2/A55r2_20261026'
CARD = PROJECTS/'experiment_cards/A55r2_current_direction_value_20261026.md'
BRANCH = 'results/ffhq-current-direction-value-r2-20261026'


def require(test, message):
    if not test:
        raise ValueError(message)


def new_arms(p, c):
    return ('Q_ORIGINAL',) if c['key'] in p['physical_sources'] else rules.ARMS


def check(root):
    p = source.check_plan(root)
    require(p['experiment'] == 'A55r2_current_direction_value', 'Wrong plan')
    parent.a42.verify(p)
    return p


def transfer_inventory(p):
    """Read-only next-stage input availability; no claim these images are unused."""
    roots = dict(ffhq=Path(p['cases'][0]['image']).parent,
                 imagenet=Path('/home/lzy/imagenet/256x256'))
    value = dict(scope='Availability only; neither held-out selection nor model validation. No image tensors copied.', datasets={})
    for dataset, root in roots.items():
        files = sorted(f.name for f in root.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')) if root.is_dir() else []
        value['datasets'][dataset] = dict(root=str(root), exists=root.is_dir(), image_count=len(files), filenames=files)
    model_root = PROJECTS/'modules/models'
    value['model_files'] = [dict(path=str(f), bytes=f.stat().st_size, checkpoint_validated=False)
                            for f in sorted(model_root.glob('*.pt')) if f.is_file()]
    return value


def prepare(args):
    from contextlib import ExitStack
    from utils.a54_session import exclusive
    root, prior = Path(args.root).resolve(), Path(args.a55_dir).resolve()
    require(not (root.is_relative_to(prior) or prior.is_relative_to(root)), 'Source/output overlap')
    if root.exists():
        p = check(root)
        require(p['source_A55'] == str(prior), 'Frozen source differs')
        print('A55r2 frozen plan reused; complete work retained.', flush=True)
        return
    # An SSH failure is not a liveness test. Do not inspect changing old attempts.
    with ExitStack() as locks:
        locks.enter_context(exclusive(prior.parent/'launcher.lock'))
        for shard in (0, 1):
            locks.enter_context(exclusive(prior/f'control_shard{shard}.lock'))
        before = original.check(prior)
        identities = dict(before['identity_files'])
        identities[str(prior/'plan.json')] = source.digest(prior/'plan.json')
        reused, legacy = {}, []
        for c in before['cases']:
            found = original.item(prior, before, c, 'control')
            paths = sorted((prior/c['key']).glob('control_*/run.json'))
            require(len(paths) <= 1, 'Multiple old attempts; no output selection')
            if found is not None:
                f, r, _ = found
                reused[c['key']] = str(f)
                for name, sha in r['artifacts'].items():
                    identities[str(f.parent/name)] = sha
            for f in paths:
                r = source.read(f)
                require(r['case'] == c and r['plan_sha256'] == source.digest(prior/'plan.json'), 'Old attempt identity differs')
                require(r['status'] in ('running', 'complete') or
                        (r['status'] == 'failed' and 'A55 controls forbid any input/parameter backward' in r.get('error', '')),
                        'Old failure is not the identified adjoint-guard bug: '+str(f))
                require(r.get('backward_attempts', 0) == r.get('backward_completed', 0) == 0 and
                        0 <= r['nfe'] <= 30*len(new_arms(before, c)), 'Invalid old cost counters')
                identities[str(f)] = source.digest(f)
                legacy.append(dict(case=c['key'], path=str(f), sha256=source.digest(f), status=r['status'],
                                   recorded_forward_attempts=r['nfe'], reused_complete=found is not None,
                                   error=r.get('error')))
        own = [Path(__file__), Path(rules.__file__), CARD, PROJECTS/'run_a55r2_20261026.sh',
               PROJECTS/'utils/a55r2_session.py', PROJECTS/'utils/test_current_direction_control_r2.py',
               PROJECTS/'utils/linear_adjoint_autograd_guard.py', PROJECTS/'utils/test_linear_adjoint_autograd_guard.py',
               PROJECTS/'modules/degradations.py']
        p = dict(before)
        remaining = sum(30*len(new_arms(before, c)) for c in before['cases'] if c['key'] not in reused)
        p.update(experiment='A55r2_current_direction_value', source_A55=str(prior),
            reused_control_sources=reused, legacy_A55_attempts=legacy,
            legacy_A55_recorded_forward_attempts=sum(r['recorded_forward_attempts'] for r in legacy),
            identity_files=identities,
            worker_identity_files={**before['worker_identity_files'], **{str(f): source.digest(f) for f in own}},
            planned_diagnostic_nfe=remaining,
            limits=before['limits']+' R2 only repairs the autograd guard: the original linear-resizer probe adjoint '
                  'is allowed and counted as operator autograd, while all other grad calls and denoiser VJP are forbidden. '
                  'Old A55 bytes remain untouched; completed controls reused; failed attempted forwards separately charged.')
        root.parent.mkdir(parents=True, exist_ok=True)
        parent.P.coupled.archive.recovery.require_space(root.parent)
        root.mkdir(); signed(root/'plan.json', p); shutil.copyfile(CARD, root/'experiment_card.md')
        snapshot = root/'source_snapshot'; snapshot.mkdir()
        shutil.copyfile(prior/'plan.json', snapshot/'A55_plan.json')
        for r in legacy:
            shutil.copyfile(r['path'], snapshot/(r['case']+'_A55_run.json'))
        if (prior/'transfer_inventory.json').exists():
            shutil.copyfile(prior/'transfer_inventory.json', root/'transfer_inventory.json')
        else:
            signed(root/'transfer_inventory.json', transfer_inventory(p))
        signed(root/'legacy_execution_cost.json', dict(attempts=legacy,
            recorded_forward_attempts=p['legacy_A55_recorded_forward_attempts'],
            limits='Old complete and failed/running attempts counted once in addition to new R2 work; '
                   'counters are recorded attempted forwards, not an all-history or internal-checkpoint cost guarantee.'))
    print('A55r2 PREPARE PASS: guard corrected; old complete control cases reused=', len(reused),
          '; old recorded forward attempts=', p['legacy_A55_recorded_forward_attempts'],
          '; NEW forwards=', remaining, '; NEW denoiser input backwards=0. Operator adjoint autograd counted separately.', flush=True)


def item(root, p, c, phase):
    require(phase == 'control', 'A55 only has controls')
    if c['key'] in p['reused_control_sources']:
        prior = Path(p['source_A55']); prior_plan = source.load_signed(prior/'plan.json')
        found = original.item(prior, prior_plan, c, phase)
        require(found is not None and str(found[0]) == p['reused_control_sources'][c['key']], 'Reused control differs')
        return found
    paths = sorted((root/c['key']).glob('control_*/run.json'))
    if not paths:
        return None
    require(len(paths) == 1, 'Multiple attempts; no output selection')
    f = paths[0]; r = source.read(f)
    if r['status'] != 'complete':
        return None
    r = source.load_signed(f); names = new_arms(p, c)
    require(r['case'] == c and r['plan_sha256'] == source.digest(root/'plan.json') and
            r['calls'] == list(reversed(p['grid']))*len(names) and r['nfe'] == 30*len(names), 'Call/identity contract differs')
    require(r['backward_attempts'] == r['backward_completed'] == 0 and set(r['traces']) == set(names), 'Unexpected VJP/arm')
    require(r['gates'] and all(g['passed'] for g in r['gates'].values()), 'Failed gate')
    counts = r['autograd_accounting']
    require(counts['forbidden_grad_attempts'] == 0 and counts['operator_grad_attempts'] == counts['operator_grad_completed'], 'Operator autograd gate failed')
    for name in names:
        require(len(r['traces'][name]) == 30, 'Missing trace')
        for row in r['traces'][name]:
            z = row['readout']
            require(z['backward_calls'] == 0 and z['current_Jacobian_used'] is False, 'Unexpected Jacobian')
            if z['resolved']:
                require(z['norm_rule_relative_error'] <= 2e-6 and z['direction'] == name, 'Target norm/direction gate failed')
    require(all(source.digest(f.parent/name) == sha for name, sha in r['artifacts'].items()), 'Output changed')
    return f, r, 'U_CURRENT'


def sample(root, p, c, phase, model, device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    if item(root, p, c, phase):
        return
    operator, mask, opgate, state, original, y, z, positions, noises = parent.setup(p, c, model, device)
    folder, r = engine.old.previous.new_attempt(root, p, c, phase)
    path = folder/'run.json'; external = rng_state(device); outputs = {}; names = new_arms(p, c)
    with engine.old.previous.counted(model, path, r, 30*len(names)), \
         engine.old.previous.runtime.input_vjp_parameters(model.model), model_rng_guard(model.model, device), \
         patch('torch.randn_like', side_effect=RuntimeError('Unexpected random draw')), \
         operator_only_autograd(r.setdefault('autograd_accounting', {})):
        for name in names:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
            x = state['init_noise'].to(device); r['traces'][name] = []
            with parent.P.precision.ieee(torch):
                for k, t in enumerate(positions.nominal):
                    s = positions.nominal[k+1] if k < 29 else -1
                    field, gamma, reading = rules.sensed_field(z, y, x, t, s, positions, c['sigma'], name, mask)
                    r['traces'][name].append(dict(step=k, t=t, next_t=s, readout=reading))
                    with torch.no_grad():
                        x = parent.old.parent.rules.risk.advance(z, x, field, positions, k, noises, gamma)
                    require(bool(torch.isfinite(x).all()), 'Nonfinite trajectory:'+name)
                outputs[name] = x.detach().cpu()
            torch.cuda.synchronize(device)
            r['arm_seconds'][name] = time.perf_counter()-started
            r['arm_peak_memory_bytes'][name] = torch.cuda.max_memory_allocated(device)
        parent.invariant(model, z, state, device, external, p)
        require(r['calls'] == list(reversed(p['grid']))*len(names) and
                r['backward_attempts'] == r['backward_completed'] == 0, 'Paid call contract failed')
        parent.old.parent.base.save_tensor(outputs, folder/'outputs.pt')
        signed(folder/'decisions_before_labels.json', dict(case=c, traces=r['traces'], GT_used=False, plan_sha256=r['plan_sha256']))
        r['artifacts'] = {name: source.digest(folder/name) for name in ('outputs.pt', 'decisions_before_labels.json')}
        r['gates'] = dict(operator=opgate, calls_rng_parameters=dict(passed=True),
                         no_current_GT_used=dict(passed=True), zero_denoiser_backward=dict(passed=True),
                         norm_rule=dict(passed=True, tolerance=2e-6,
                             scope='Resolved steps follow K||v(current x)||; unresolved steps explicitly use original q.'))
    r['status'] = 'complete'; signed(path, r)
    print('A55 CONTROL:', c['key'], 'new_forwards=', r['nfe'], 'new_backwards=0', flush=True)


@contextmanager
def executor_bindings():
    from unittest.mock import patch
    with patch.object(engine, 'check', check), patch.object(engine, 'item', item), patch.object(engine, 'sample', sample):
        yield


def evaluate(root, p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    pairs = [(c, item(root, p, c, 'control')) for c in p['cases']]
    require(all(found for c, found in pairs), 'Need all10 new control cases before GT')
    hashes = {str(v[0]): source.digest(v[0]) for c, v in pairs}
    path = root/'evaluation.json'
    if path.exists():
        old = source.load_signed(path); require(old['inputs'] == hashes, 'Evaluation sources changed'); return old
    cached = {r['case']['key']: r for r in source.load_signed(Path(p['reference_evaluation']))['rows']}
    physical = {r['case']['key']: r for r in source.load_signed(Path(p['physical_evaluation']))['rows']}
    rows = []
    with parent.scan.preserved_rng(), torch.no_grad():
        for c, (f, r, _) in pairs:
            prior = cached[c['key']]; require(prior['case'] == c, 'Cached case differs')
            outputs = torch.load(f.parent/'outputs.pt', map_location='cpu', weights_only=True)
            gt = load_image_as_tensor(c['image']).cpu()
            metrics = {name: compute_all_metrics(x, gt, lpips_net='alex') for name, x in outputs.items()}
            resolved = {name: sum(row['readout']['resolved'] for row in r['traces'][name]) for name in outputs}
            if c['key'] in p['physical_sources']:
                metrics['V_PHYSICAL'] = dict(physical[c['key']]['metrics']['FULL_PROX'])
                resolved['V_PHYSICAL'] = 30
            metrics['U_CURRENT'] = dict(prior['metrics']['STATE_GAIN']); resolved['U_CURRENT'] = 30
            for name in ('C_NATIVE', 'C_IEEE', 'A44_TASK_FIXED', 'TIME_GAIN', 'SHUFFLE_GAIN'):
                metrics[name] = dict(prior['metrics'][name])
            require(all(math.isfinite(float(v)) for m in metrics.values() for v in m.values()), 'Nonfinite metric')
            rows.append(dict(case=c, metrics=metrics, resolved=resolved))
    value = dict(inputs=hashes, rows=rows, GT_used_only_after_all_new_outputs_frozen=True, limits=p['limits'])
    signed(path, value); return value


def summary(root, p):
    value = evaluate(root, p); groups = []
    attempts = [source.read(f) for f in root.glob('*/control_*/run.json')]
    for task in previous.rules.TASKS:
        rows = [r for r in value['rows'] if r['case']['task'] == task]
        effects = {arm: {ref: {m: mean(r['metrics'][arm][m]-r['metrics'][ref][m] for r in rows)
                                  for m in ('psnr', 'ssim', 'lpips')}
                        for ref in ('C_IEEE', 'U_CURRENT', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')}
                   for arm in ('U_CURRENT', *rules.ARMS)}
        wins = {ref: sum(r['metrics']['U_CURRENT']['psnr'] > r['metrics'][ref]['psnr']+1e-9 for r in rows)
                for ref in ('V_PHYSICAL', 'Q_ORIGINAL')}
        full = all(r['resolved'][arm] == 30 for r in rows for arm in ('U_CURRENT', *rules.ARMS))
        passes = {ref: bool(full and effects['U_CURRENT'][ref]['psnr'] >= .05 and
                          effects['U_CURRENT'][ref]['ssim'] >= -.002 and effects['U_CURRENT'][ref]['lpips'] <= .005 and wins[ref] >= 4)
                  for ref in wins}
        groups.append(dict(task=task, mean_delta=effects, U_positive_images=wins,
            fully_resolved=bool(full), pullback_increment_by_reference=passes,
            current_pullback_direction_candidate=bool(all(passes.values())),
            resolved_states={arm: sum(r['resolved'][arm] for r in rows) for arm in ('U_CURRENT', *rules.ARMS)},
            per_image=rows))
    result = dict(experiment=p['experiment'], classification='DEVELOPMENT_PULLBACK_DIRECTION_CANDIDATE'
        if any(g['current_pullback_direction_candidate'] for g in groups) else 'NO_CONFIRMED_PULLBACK_DIRECTION_INCREMENT',
        completed_cases=10, planned_cases=10, image_clusters=5, reused_U_cases=10, reused_V_cases=4,
        method_nfe=0, diagnostic_nfe=sum(r['nfe'] for r in attempts), planned_diagnostic_nfe=p['planned_diagnostic_nfe'],
        backward_attempts=sum(r['backward_attempts'] for r in attempts), backward_completed=sum(r['backward_completed'] for r in attempts),
        partial_attempts=sum(r['status'] != 'complete' for r in attempts), scientific_forward_nfe=300,
        new_arm_seconds={name: sum(r['arm_seconds'].get(name, 0.) for r in attempts) for name in rules.ARMS},
        operator_accounting={key: sum(row['readout'].get('cg', {}).get(key, 0) for r in attempts
            for trace in r['traces'].values() for row in trace) for key in ('linear_calls', 'adjoint_calls')},
        operator_autograd_accounting={key: sum(r['autograd_accounting'][key] for r in attempts)
            for key in ('operator_grad_attempts','operator_grad_completed','forbidden_grad_attempts')},
        reused_old_control_cases=len(p['reused_control_sources']),
        legacy_A55_recorded_forward_attempts=p['legacy_A55_recorded_forward_attempts'],
        total_recorded_A55_and_R2_forward_attempts=p['legacy_A55_recorded_forward_attempts']+sum(r['nfe'] for r in attempts),
        groups=groups, limits=p['limits']+' U timings are cached, V has four cached trajectories; no controlled end-to-end speed claim.')
    signed(root/'summary.json', result)
    print('A55r2 Summary:', json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    print('Records:', root, flush=True)
    print('Transfer inventory:', root/'transfer_inventory.json', '; availability only, new-image cohort not selected.', flush=True)
    return value


def render(root, p, value):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from utils.diag_cached_guidance_final_value import font
    for row in value['rows']:
        c = row['case']; f, r, _ = item(root, p, c, 'control')
        outputs = torch.load(f.parent/'outputs.pt', map_location='cpu', weights_only=True)
        if c['key'] in p['physical_sources']:
            pf = Path(p['physical_sources'][c['key']])
            outputs['V_PHYSICAL'] = torch.load(pf.parent/'outputs.pt', map_location='cpu', weights_only=True)['FULL_PROX']
        sf = Path(p['norm_sources'][c['key']])
        outputs['U_CURRENT'] = torch.load(sf.parent/'outputs.pt', map_location='cpu', weights_only=True)['NORM_ONLY']
        bf = Path(p['a50_baselines'][c['key']])
        outputs['C_IEEE'] = torch.load(bf.parent/'outputs.pt', map_location='cpu', weights_only=True)['C_IEEE']
        sheet = Image.new('RGB', (1392, 390), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), c['key']+' | same CURRENT-state norm law K||v||; direction differs | OLD development images', fill='black', font=font(13))
        for i, name in enumerate(('GT', 'C_IEEE', 'U_CURRENT', 'V_PHYSICAL', 'Q_ORIGINAL')):
            left = 12+276*i; draw.text((left, 40), name, fill='black', font=font(14))
            if name == 'GT':
                im = Image.open(c['image']).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
            else:
                arr = outputs[name][0].numpy().transpose(1, 2, 0)
                im = Image.fromarray(((np.clip(arr, -1, 1)+1)*127.5).astype(np.uint8))
                m = row['metrics'][name]
                draw.text((left, 66), f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}', fill='black', font=font(12))
            sheet.paste(im, (left, 96))
        (root/c['key']).mkdir(exist_ok=True)
        sheet.save(root/c['key']/'direction_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--a55-dir', default=original.ROOT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--phase', choices=('control',), default='control')
    parser.add_argument('--shard', type=int, choices=(0, 1)); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--archive', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    if args.shard is not None:
        with executor_bindings():
            engine.worker(args)
        return
    root = Path(args.root).resolve(); p = check(root); value = summary(root, p)
    if args.archive:
        from unittest.mock import patch
        render(root, p, value)
        with patch.object(engine.old.previous, 'BRANCH', BRANCH):
            engine.old.previous.archive(root)


if __name__ == '__main__':
    main()
