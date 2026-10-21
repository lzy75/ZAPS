"""A33: continuous identity feedback vs matched continuous time profiles; 2160 diagnostic NFE."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_fixed_grid_response_feedback as old
from utils import continuous_response_feedback as rules

source, P = old.source, old.P
ROOT = '/data0/lzy/ZAPS-results/diag_continuous_response_feedback/A33_20261021'
CARD = PROJECTS/'experiment_cards/A33_continuous_response_feedback_20261021.md'


def prepare(args):
    root, parent = Path(args.root).resolve(), Path(args.a32_dir).resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('A33 must not overlap A32')
    if root.exists():
        plan = source.check_plan(root)
        if plan['source_A32'] != str(parent):
            raise ValueError('Existing A33 parent differs')
        print('A33 frozen plan reused; completed arms will not rerun.', flush=True)
        return
    previous = source.check_plan(parent)
    summary = source.load_signed(parent/'summary.json')
    expected = {(f'{i:05d}', seed) for i in range(20, 26) for seed in (1000, 2000)}
    if (summary['completed_cases'] != 12 or len(previous['cases']) != 12
            or previous['experiment'] != 'A32_fixed_grid_response_feedback'
            or {(s['case']['image_id'], s['case']['sampling_seed']) for s in previous['cases']} != expected):
        raise ValueError('Need all twelve complete A32 cases')
    identities = dict(previous['identity_files'])
    refs = {}
    for spec in previous['cases']:
        P.verify_case_sources(previous, spec)
        item = old.completed(parent, previous, spec)
        if item is None:
            raise ValueError('A32 incomplete: '+spec['case']['key'])
        path, record = item
        refs[spec['case']['key']] = str(path)
        for f in (path, path.with_suffix('.sha256'), path.parent/'outputs.pt'):
            identities[str(f)] = source.digest(f)
    for name in ('plan.json', 'plan.sha256', 'summary.json', 'summary.sha256'):
        identities[str(parent/name)] = source.digest(parent/name)
    workers = dict(previous['worker_identity_files'])
    own = [Path(__file__), Path(rules.__file__), Path(old.__file__), CARD,
           PROJECTS/'run_a33_20261021.sh', PROJECTS/'utils/test_continuous_response_feedback.py',
           PROJECTS/'utils/archive_refinement_confirmation.py', PROJECTS/'utils/metrics.py']
    for path in own:
        workers[str(path)] = source.digest(path)
    for path, sha in workers.items():
        if source.digest(path) != sha:
            raise ValueError('Parent worker differs: '+path)
    plan = dict(experiment='A33_continuous_response_feedback', source_A32=str(parent),
                source_A22=previous['source_A22'], cases=previous['cases'],
                config=previous['config'], grid=previous['grid'], runtime=previous['runtime'],
                native_policy=previous['native_policy'], checkpoint_sha256=previous['checkpoint_sha256'],
                identity_files=identities, worker_identity_files=workers, parent_records=refs,
                policies=rules.POLICIES, pairs=rules.PAIRS, minimum_predicted_improvement=.01,
                scientific_nfe_per_arm=300, prefix_reused_nfe=270, sensor_extra_unet=0,
                planned_method_nfe=0, planned_diagnostic_nfe=2160, GT_online=False,
                scope='Six reused FFHQ development image clusters, two seeds each, final unroll only. '
                      'Same-time identity response surrogate; no exact next-state Jacobian or curvature. '
                      'Continuous stage profiles exclude both seeds of the held image; no GT reference selection.')
    root.parent.mkdir(parents=True, exist_ok=True)
    P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir()
    source.signed(root/'plan.json', plan)
    (root/'experiment_card.md').write_text(CARD.read_text())
    print('A33 PREPARE PASS: 12 cases x6 new arms x30 =2160 diagnostic NFE; every method=300 NFE. '
          'A32 C/F/I/S/STAGE reused. Continuous limits: 2, 8, no fixed cap.', flush=True)


def task_done(root, plan, spec, arm, profile_sha=None):
    results = []
    for path in sorted((root/spec['case']['key']/arm).glob('attempt_*/run.json')):
        record = source.read(path)
        if record['status'] != 'complete':
            continue
        record = source.load_signed(path)
        if (record['plan_sha256'] != source.digest(root/'plan.json')
                or record['case'] != spec['case'] or record['arm'] != arm
                or record['diagnostic_nfe'] != 30 or record['method_nfe'] != 0
                or [v['t'] for v in record['calls']] != list(reversed(plan['grid']))
                or [v['t'] for v in record['rows']] != list(reversed(plan['grid']))
                or record['profile_sha256'] != profile_sha
                or not record['gates'] or not all(g['passed'] for g in record['gates'].values())
                or source.digest(path.parent/'output.pt') != record['output_sha256']):
            raise ValueError('Invalid complete A33 arm: '+str(path))
        results.append((path, record))
    if len(results) > 1:
        raise ValueError('Multiple complete attempts; no output selection')
    return results[0] if results else None


def read_profiles(root, plan):
    path = root/'stage_profiles.json'
    data = source.load_signed(path)
    if data['plan_sha256'] != source.digest(root/'plan.json') or data['GT_used']:
        raise ValueError('Stage profile identity differs')
    for info in data['sources'].values():
        if source.digest(info['path']) != info['sha256']:
            raise ValueError('Frozen feedback trace changed')
    if any(data['policy_sha256'][arm] != rules.profile_signature(data, arm) for arm in data['ready_policies']):
        raise ValueError('Per-range stage signature differs')
    return data


def build_profiles(args):
    root = Path(args.root).resolve()
    plan = source.check_plan(root)
    profiles, ready, missing, signatures = {}, [], {}, {}
    for arm in rules.POLICIES:
        current = {}
        missing[arm] = []
        for spec in plan['cases']:
            item = task_done(root, plan, spec, arm)
            if item is None:
                missing[arm].append(spec['case']['key'])
                continue
            path, record = item
            # Only trace + image identity enter the time rule. Metrics/GT are not inputs.
            current[spec['case']['key']] = dict(image_id=spec['case']['image_id'],
                                               traces={arm: record['rows']})
            signatures[spec['case']['key']+'/'+arm] = dict(path=str(path), sha256=source.digest(path))
        if not missing[arm]:
            ready.append(arm)
            for spec in plan['cases']:
                key = spec['case']['key']
                profiles.setdefault(key, {})[rules.PAIRS[arm]] = rules.stage_profile(
                    current, spec['case']['image_id'], arm)
    data = dict(plan_sha256=source.digest(root/'plan.json'), GT_used=False,
                aggregation='Per-time median of actual same-policy feedback trajectories; '
                            'both held-image seeds excluded; no quantization.',
                ready_policies=ready, unavailable_policies=missing, values=profiles, sources=signatures)
    data['policy_sha256'] = {arm: rules.profile_signature(data, arm) for arm in ready}
    path = root/'stage_profiles.json'
    if path.exists():
        previous = read_profiles(root, plan)
        if any(data['policy_sha256'].get(a) != previous['policy_sha256'][a] for a in previous['ready_policies']):
            raise ValueError('An already frozen time profile changed; original records preserved')
        if previous != data:
            source.signed(path, data)
    else:
        source.signed(path, data)
    print('A33 STAGE profiles ready:', ready, '; unavailable:',
          {a: v for a, v in missing.items() if v}, '; extra UNet=0.', flush=True)


def run_task(root, plan, spec, arm, context_data, device, model, profile=None, profile_sha=None):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics

    if task_done(root, plan, spec, arm, profile_sha):
        return
    z, y, state, cache, positions, noises, replay = context_data
    folder = root/spec['case']['key']/arm
    folder.mkdir(parents=True, exist_ok=True)
    attempt = folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}'
    attempt.mkdir()
    path = attempt/'run.json'
    started = time.perf_counter()
    record = dict(status='running', case=spec['case'], arm=arm,
                  plan_sha256=source.digest(root/'plan.json'), profile_sha256=profile_sha,
                  diagnostic_nfe=0, method_nfe=0, scientific_nfe=300, rows=[], calls=[], gates={})
    call_context = {}
    def save():
        record['seconds'] = time.perf_counter()-started
        source.WRITE(path, record)
    def count(_module, inputs):
        t = int(inputs[1].flatten()[0])
        record['diagnostic_nfe'] += 1
        record['calls'].append(dict(**call_context, t=t))
        save()
        if t != call_context['expected_t'] or record['diagnostic_nfe'] > 30:
            raise RuntimeError('Call budget or time grid differs')
    save()
    hook = model.model.register_forward_pre_hook(count)
    try:
        external = rng_state(device)
        sensor_seconds = 0.
        with torch.no_grad(), model_rng_guard(model.model, device), patch(
                'torch.randn_like', side_effect=RuntimeError('Unexpected sampling draw')):
            x = state['init_noise'].to(device)
            for k, t in enumerate(positions.nominal):
                call_context.update(step=k, expected_t=t)
                f = source.fields.field(z, y, x, t, positions)
                if k == 0:
                    center_error = max(relative_error(x.cpu(), cache['latent_inputs'][0:1]),
                                       relative_error(f['clean'].cpu(), cache['clean'][0:1]),
                                       relative_error(f['q'].cpu(), cache['guidance'][0:1]))
                    if not math.isfinite(center_error) or center_error > 1e-6:
                        raise RuntimeError('Original first-center parity failed')
                if profile is None:
                    now = time.perf_counter()
                    readout = rules.sensor(z, y, f['clean'], f['q'], t, rules.POLICIES[arm])
                    sensor_seconds += time.perf_counter()-now
                    gamma = readout['gamma']
                else:
                    gamma = profile[k]
                    q_rms = float(f['q'].double().square().mean().sqrt())
                    readout = dict(qRMS=q_rms, guidance_rms=gamma*q_rms)
                record['rows'].append(dict(step=k, t=t, gamma=gamma, R=f['R'],
                                           clip=f['clip_fraction'], **{n: v for n, v in readout.items()
                                                                     if n not in ('gamma', 'R')}))
                base = P.terminal.terminal_output(z, x, f) if k == 29 else P.coupled.posterior(
                    x, f['clean'], t, positions.nominal[k+1], z.dm.alphas_cumprod, noises[k])+f['q']
                x = base if gamma == 1. else base+(gamma-1.)*f['q']
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite state; this arm stops, with no output fallback')
            output = x.cpu()
        if (record['diagnostic_nfe'] != 30 or not same_rng(external, rng_state(device))
                or P.precision.precision_state(torch) != plan['native_policy']):
            raise RuntimeError('RNG, precision or budget differs')
        if not all(torch.equal(a.cpu(), state[n]) for a, n in (
                (z.tau, 'tau'), (z.zeta, 'zeta'), (z.D, 'D'))):
            raise RuntimeError('Grid or original learned parameters mutated')
        tmp = attempt/'output.pt.tmp'
        torch.save(output, tmp)
        tmp.replace(attempt/'output.pt')
        record.update(status='sampled', output_sha256=source.digest(attempt/'output.pt'),
                      sensor_wall_seconds=sensor_seconds)
        save()
        # The output and every action have been fixed before GT is read for this arm.
        gt = load_image_as_tensor(spec['case']['image']).cpu()
        metrics = dict(loss=float((y-z.A.H(output.to(device))).double().square().mean()/(2*.05**2)),
                       **compute_all_metrics(output, gt, lpips_net='alex'))
        if not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError('Nonfinite final metric')
        record.update(status='complete', metrics=metrics,
                      gates=dict(first_center=dict(passed=True, relative_error=center_error),
                                 sampling=replay, grid_parameters=dict(passed=True),
                                 budget_rng_precision=dict(passed=True), GT_after_output=dict(passed=True),
                                 sensor_extra_unet=dict(passed=True, nfe=0)))
        save()
        source.signed(path, record)
        print('A33 ARM:', spec['case']['key'], arm, 'PSNR=', round(metrics['psnr'], 5),
              'gamma[min,max]=', [min(v['gamma'] for v in record['rows']),
                                  max(v['gamma'] for v in record['rows'])], flush=True)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        save()
        raise
    finally:
        hook.remove()


def worker(args):
    import fcntl
    import torch
    from utils.diag_state_weight_scan import preserved_rng

    root = Path(args.root).resolve()
    plan = source.check_plan(root)
    data = read_profiles(root, plan) if args.phase == 'stage' else None
    arms = [rules.PAIRS[a] for a in data['ready_policies']] if data else list(rules.POLICIES)
    hashes = {rules.PAIRS[a]: data['policy_sha256'][a] for a in data['ready_policies']} if data else {}
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = [s for i, s in enumerate(sorted(plan['cases'], key=lambda s: s['case']['key']))
                 if (i//2) % 2 == args.shard]
        if all(task_done(root, plan, s, a, hashes.get(a)) for s in specs for a in arms):
            print('A33 phase/shard already complete; extra UNet=0.', flush=True)
            return
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(args.device),
                       NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != plan['runtime'] or P.precision.precision_state(torch) != plan['native_policy']:
            raise ValueError('Source runtime/precision differs')
        torch.set_num_threads(2)
        failures = []
        with preserved_rng(), torch.no_grad():
            model = P.coupled.archive.recovery.load_model(plan, args.device)
            for spec in specs:
                if all(task_done(root, plan, spec, a, hashes.get(a)) for a in arms):
                    continue
                P.verify_case_sources(plan, spec)
                parent_path = Path(plan['parent_records'][spec['case']['key']])
                if source.digest(parent_path) != plan['identity_files'][str(parent_path)]:
                    raise ValueError('A32 reference changed')
                native32 = source.load_signed(parent_path)
                if source.digest(parent_path.parent/'outputs.pt') != native32['outputs_sha256']:
                    raise ValueError('A32 output changed')
                z, y, state, cache, positions = old.setup(plan, spec, model, args.device)
                original = torch.load(Path(spec['source'])/'last_opt.pt', map_location='cpu', weights_only=True)
                native = source.read(Path(spec['source'])/'run.json')
                noises, replay = P.coupled.replay_noises(state, cache, dict(rows=native['rows']),
                                                       z, y, args.device, original)
                context_data = (z, y, state, cache, positions, noises, replay)
                for arm in arms:
                    try:
                        run_task(root, plan, spec, arm, context_data, args.device, model,
                                 None if data is None else data['values'][spec['case']['key']][arm], hashes.get(arm))
                    except Exception as error:
                        failures.append(dict(case=spec['case']['key'], arm=arm, error=repr(error)))
                        print('A33 ARM FAILED; other arms continue:', failures[-1], flush=True)
                P.verify_case_sources(plan, spec)
        source.check_plan(root)
        if failures:
            print('A33 incomplete arms:', failures, flush=True)
            raise SystemExit(1)


def difference(pairs):
    metrics = ('psnr', 'ssim', 'lpips', 'loss')
    means = {m: statistics.mean(a['metrics'][m]-b['metrics'][m] for a, b in pairs) for m in metrics}
    images = sorted({a['case']['image_id'] for a, b in pairs})
    per_image = {image: statistics.mean(a['metrics']['psnr']-b['metrics']['psnr'] for a, b in pairs
                                       if a['case']['image_id'] == image) for image in images}
    return dict(n=len(pairs), image_clusters=len(images), mean_delta=means,
                per_image_psnr=per_image, images_positive=sum(v > 1e-9 for v in per_image.values()),
                quality_guard=means['ssim'] >= -.002 and means['lpips'] <= .005)


def pictures(root, spec, parent_record, tasks):
    import torch
    from PIL import Image, ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font

    old_outputs = torch.load(parent_record.parent/'outputs.pt', map_location='cpu', weights_only=True)
    native = source.load_signed(parent_record)
    arrays = {'GT': _to_uint8_numpy(load_image_as_tensor(spec['case']['image'])),
              'C': _to_uint8_numpy(old_outputs['C']), 'A32_STAGE': _to_uint8_numpy(old_outputs['STAGE'])}
    metrics = {'C': native['metrics']['C'], 'A32_STAGE': native['metrics']['STAGE']}
    for arm, (path, record) in tasks.items():
        arrays[arm] = _to_uint8_numpy(torch.load(path.parent/'output.pt', map_location='cpu', weights_only=True))
        metrics[arm] = record['metrics']
    names = ['GT', 'C', 'A32_STAGE'] + [a for pair in rules.PAIRS.items() for a in pair if a in arrays]
    sheet = Image.new('RGB', (20+276*len(names), 680), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 8), spec['case']['key']+' | every arm 300 NFE | original fixed grid | real outputs',
              fill='black', font=font(18))
    roi = (64, 48, 192, 176)
    for i, arm in enumerate(names):
        x = 20+276*i
        draw.text((x, 43), arm, fill='black', font=font(17))
        if arm != 'GT':
            m = metrics[arm]
            draw.text((x, 68), f'P {m["psnr"]:.3f}  S {m["ssim"]:.4f}', fill='black', font=font(14))
            draw.text((x, 88), f'LPIPS {m["lpips"]:.4f}', fill='black', font=font(14))
        raw = Image.fromarray(arrays[arm])
        full = raw.copy()
        ImageDraw.Draw(full).rectangle(roi, outline='#e7a827', width=2)
        sheet.paste(full, (x, 115))
        sheet.paste(raw.crop(roi).resize((256, 256), Image.Resampling.NEAREST), (x, 390))
    draw.text((20, 653), 'Same fixed ROI and historical display conversion; nearest-neighbor zoom; no enhancement.',
              fill='black', font=font(15))
    sheet.save(root/spec['case']['key']/'comparison.png')


def summarize(args):
    root = Path(args.root).resolve()
    plan = source.check_plan(root)
    data = read_profiles(root, plan) if (root/'stage_profiles.json').exists() else None
    hashes = {rules.PAIRS[a]: data['policy_sha256'][a] for a in data['ready_policies']} if data else {}
    records, task_paths, references = {}, {}, {}
    for spec in plan['cases']:
        key = spec['case']['key']
        parent_path = Path(plan['parent_records'][key])
        if source.digest(parent_path) != plan['identity_files'][str(parent_path)]:
            raise ValueError('Bound A32 record changed')
        parent_record = source.load_signed(parent_path)
        references[key] = {name: dict(case=spec['case'], metrics=parent_record['metrics'][a])
                           for name, a in [('C', 'C'), ('F', 'F'), ('A32_I', 'I'), ('A32_S', 'S'), ('A32_STAGE', 'STAGE')]}
        for arm in list(rules.POLICIES)+list(rules.PAIRS.values()):
            item = task_done(root, plan, spec, arm, hashes.get(arm))
            if item:
                task_paths[(key, arm)], records[(key, arm)] = item
    cost, partial, active, failures = 0, 0, False, []
    for path in sorted(root.glob('*/*/attempt_*/run.json')):
        r = source.read(path)
        cost += r['diagnostic_nfe']
        partial += r['status'] != 'complete'
        active |= r['status'] in ('running', 'sampled')
        if r['status'] == 'failed':
            failures.append(dict(case=r['case']['key'], arm=r['arm'], error=r.get('error')))
    variants = []
    for arm, stage in rules.PAIRS.items():
        keys = [s['case']['key'] for s in plan['cases']]
        feedback_keys = [k for k in keys if (k, arm) in records]
        time_keys = [k for k in keys if (k, stage) in records]
        matched = [k for k in feedback_keys if k in time_keys]
        comparisons = {b: difference([(records[(k, arm)], references[k][b]) for k in feedback_keys])
                       for b in ('C', 'F', 'A32_I', 'A32_S', 'A32_STAGE') if feedback_keys}
        if matched:
            comparisons['matched_STAGE'] = difference([(records[(k, arm)], records[(k, stage)]) for k in matched])
            comparisons['matched_STAGE_vs_C'] = difference([(records[(k, stage)], references[k]['C']) for k in matched])
        rows = [v for k in feedback_keys for v in records[(k, arm)]['rows']]
        gamma = [v['gamma'] for v in rows]
        stats = dict(decisions=len(rows), gamma_min=min(gamma) if gamma else None,
                     gamma_median=statistics.median(gamma) if gamma else None,
                     gamma_max=max(gamma) if gamma else None,
                     capped=sum(v['cap_hit'] for v in rows), fallback=sum(v['fallback'] for v in rows),
                     changed_vs_one=sum(abs(v['gamma']-1.) > 1e-8 for v in rows),
                     outside_old_discrete_set=sum(min(abs(v['gamma']-g) for g in (0., .5, 1., 2.)) > 1e-6 for v in rows))
        complete = len(matched) == 12
        candidate = complete and all(comparisons[b]['quality_guard'] and comparisons[b]['mean_delta']['psnr'] >= .01
                                     for b in ('C', 'matched_STAGE')) and comparisons['matched_STAGE']['images_positive'] >= 4
        classification = 'DEVELOPMENT_STATE_CANDIDATE_ONLY' if candidate else (
            'NO_CONFIRMED_STATE_INCREMENT' if complete else 'INCOMPLETE_RANGE')
        variants.append(dict(arm=arm, upper=rules.POLICIES[arm], completed_feedback=len(feedback_keys),
                             completed_stage=len(time_keys), classification=classification,
                             comparisons=comparisons, multipliers=stats))
    result = dict(experiment=plan['experiment'], planned_cases=12, image_clusters=6,
                  completed_cases=sum(all((s['case']['key'], a) in records
                                          for a in list(rules.POLICIES)+list(rules.PAIRS.values()))
                                      for s in plan['cases']),
                  completed_tasks=len(records), planned_tasks=72, method_nfe=0,
                  diagnostic_nfe=cost, planned_diagnostic_nfe=2160, deployment_nfe_per_arm=300,
                  sensor_extra_unet=0, partial_attempts=partial, cost_is_lower_bound_if_abruptly_interrupted=active,
                  classification='COMPLETE_DEVELOPMENT_COMPARISON' if len(records) == 72 else 'INCOMPLETE_PILOT',
                  variants=variants, failed_attempts=failures,
                  limits=plan['scope']+' No independent confirmation, universal efficacy, NFE saving or full-ten-epoch '
                  'feedback claim. All three ranges are exploratory and fully reported; any chosen range requires '
                  'frozen independent confirmation. Uncapped quadratic optimum is not a stability guarantee.')
    source.signed(root/'summary.json', result)
    with (root/'state_traces.csv').open('w', newline='') as out:
        names = ['case', 'arm', 'step', 't', 'gamma', 'raw_gamma', 'R', 'cosine', 'relative_response',
                 'qRMS', 'guidance_rms', 'clip', 'cap_hit', 'fallback']
        writer = csv.DictWriter(out, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        for (key, arm), record in sorted(records.items()):
            for row in record['rows']:
                writer.writerow(dict(case=key, arm=arm, **row))
    if args.archive:
        for spec in plan['cases']:
            key = spec['case']['key']
            items = {a: (task_paths[(key, a)], r) for (k, a), r in records.items() if k == key}
            if items:
                pictures(root, spec, Path(plan['parent_records'][key]), items)
    print('A33 Summary:', json.dumps(result, ensure_ascii=False), flush=True)
    print('Records:', root, flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle, commit = export_and_commit({'A33': str(root)})
        print('Archive:', bundle, 'Results commit:', commit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--a32-dir', default=old.ROOT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--build-profiles', action='store_true')
    parser.add_argument('--phase', choices=('feedback', 'stage'))
    parser.add_argument('--shard', type=int, choices=(0, 1))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    elif args.build_profiles:
        build_profiles(args)
    elif args.phase:
        if args.shard is None:
            parser.error('--phase requires --shard')
        worker(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
