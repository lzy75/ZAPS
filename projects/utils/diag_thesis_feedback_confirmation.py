"""A117 fixed-grid mechanism comparisons and A118 fresh FFHQ blur confirmation."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils import diag_shared_prefix_feedback as baseline
from utils import equal_nfe_feedback_runtime as algebra
from utils import shared_prefix_feedback as rules
from utils.a54_session import busy, exclusive
from utils.diag_imagenet_closed_blur import saved_rows as validated_rows
from utils.diag_paper_clock_alignment import gpu_inventory
from utils.diag_sr_inpainting_followup import artifact, source_case
from utils.paper_clock_alignment import CHECKPOINTS, available_gpus, digest, read, require, write

TASKS = ('gaussian_deblur', 'motion_deblur')
PILOT_IDS = ('00001', '00004', '00008')
BASE, SCALAR, DIAG = rules.ARMS
PIXEL = 'PIXEL_STEP_SUFFIX'
FIXED = 'FIXED_TIME_SUFFIX'
NOSHRINK = 'NO_SHRINK_SUFFIX'
PANELS = {
    'pilot': ('A117', (SCALAR, DIAG, PIXEL, FIXED, NOSHRINK)),
    'confirm': ('A118', rules.ARMS),
}
CARDS = {'A117': 'A117_thesis_feedback_ablation.md',
         'A118': 'A118_thesis_ffhq_holdout.md'}
PARENT = Path(rules.ROOT)
METRICS = ('psnr', 'ssim', 'lpips')
BASE_SUFFIX = baseline.runtime.run_suffix
DIAGONAL_BRIDGE = baseline.runtime.repaired.exact_diagonal_with_native_straight_through
SCALAR_SELECT = baseline.runtime.scalar_rule.select


def pixel_step(zaps, operator, clean, q, residual, alpha, sigma, counts):
    """Pixel-space exact line search, with a same-state wavelet reference."""
    import torch

    scale = alpha.sqrt().clamp(min=1e-8)
    with torch.no_grad():
        base = operator.H(clean)
        r = residual.reshape(-1).double()

        def response(delta):
            return (operator.H(clean + delta / scale) - base).reshape(-1).double()

        d = response(q)
        scalar = SCALAR_SELECT(algebra._dot(r, d), algebra._dot(r, r),
                               algebra._dot(d, d), r.numel(), sigma)
        start = scalar['gamma'] * q
        goal = scalar['weight'] * r
        error = response(start) - goal
        adjoint = operator.base.transpose if hasattr(operator, 'base') else operator.transpose
        gradient = adjoint(error.reshape_as(base).to(base.dtype)) / scale
        counts['module_adjoint_actions'] += 1
        numerator = algebra._dot(gradient, gradient)
        eta = 0.
        if numerator > 0:
            change = response(gradient)
            denominator = algebra._dot(change, change)
            require(denominator > 0 and math.isfinite(denominator), 'Pixel line search unresolved')
            eta = numerator / denominator
        delta = start - eta * gradient
        final_error = response(delta) - goal
        first = .5 * algebra._dot(error, error)
        last = .5 * algebra._dot(final_error, final_error)
        require(math.isfinite(last) and last <= first + 128 * torch.finfo(clean.dtype).eps * max(1., first),
                'Pixel update failed proxy descent gate')

        support = zaps.dwt.analysis(q) != 0
        counts['module_dwt_analysis'] += 1
        native, detail, counts = DIAGONAL_BRIDGE(zaps, operator, clean, q, residual, alpha, sigma, counts)
        extra = delta - start
        reference_extra = native - start
        norms = algebra._norm(extra) * algebra._norm(reference_extra)
        relative = algebra._norm(delta - native) / max(algebra._norm(native), algebra._norm(delta), 1e-30)
        angle = None if norms == 0 else math.degrees(math.acos(max(-1., min(
            1., algebra._dot(extra, reference_extra) / norms))))
        result = dict(status='PIXEL_STEP', scalar=scalar, eta=eta, proxy_initial=first,
                      proxy_final=last, wavelet_reference_proxy_final=detail['proxy_final'],
                      nonzero_support_fraction=float(support.float().mean()),
                      same_state_correction_relative_error=relative, increment_angle_degrees=angle)
    return delta + (q - q.detach()), result, counts


def adapted_suffix(model, operator, y, case, arm, prefix):
    if arm in rules.ARMS:
        return BASE_SUFFIX(model, operator, y, case, arm, prefix)
    if arm == PIXEL:
        with patch.object(baseline.runtime.repaired, 'exact_diagonal_with_native_straight_through', pixel_step):
            return BASE_SUFFIX(model, operator, y, case, DIAG, prefix)
    require(arm in (FIXED, NOSHRINK), f'Unknown arm {arm}')
    curve = iter(case.get('fixed_gamma', ()))
    calls = 0

    def control(dot, r2, d2, count, sigma):
        nonlocal calls
        calls += 1
        row = SCALAR_SELECT(dot, r2, d2, count, sigma)
        if arm == FIXED:
            gamma = next(curve)
            row.update(gamma=gamma, reason='A117_LEAVE_ONE_OUT_TIME_TABLE')
        else:
            gamma = max(0., dot) / d2 if row['resolved'] else 0.
            row.update(gamma=gamma, weight=1., signal_variance=r2 / count,
                       noise_energy=0., reason='A117_ONLINE_WITHOUT_SHRINK')
        row['predicted_residual'] = max(0., r2 - 2 * gamma * dot + gamma * gamma * d2)
        return row

    with patch.object(baseline.runtime.scalar_rule, 'select', control):
        output = BASE_SUFFIX(model, operator, y, case, SCALAR, prefix)
    require(calls == 30, 'Control did not cover all 30 positions')
    return output


def pilot_cases(parent, plan):
    cases = []
    for task in TASKS:
        group = [c for c in plan['cases'] if c['task'] == task]
        require(len(group) == 10 and len({c['image_id'] for c in group}) == 10,
                f'Need ten A102 calibration curves for {task}')
        traces = {}
        for item in group:
            folder, row, _ = source_case(parent, item)
            path = artifact(folder, row, SCALAR, SCALAR + '_trace.json')
            trace = read(path)
            require([x['t'] for x in trace] == plan['grid'][::-1], f'Calibration grid differs: {path}')
            curve = [step['scalar']['gamma'] for step in trace]
            require(all(math.isfinite(v) and v >= 0 for v in curve), f'Invalid calibration: {path}')
            traces[item['image_id']] = (curve, digest(path))
        for case in group:
            image_id = case['image_id']
            if image_id not in PILOT_IDS:
                continue
            _, _, y_path = source_case(parent, case)
            donors = {key: value for key, value in traces.items() if key != image_id}
            curve = [median(value[0][k] for value in donors.values()) for k in range(30)]
            cases.append(dict(case, fixed_gamma=curve,
                donor_trace_sha256={key: value[1] for key, value in donors.items()},
                observation_path=str(y_path), observation_sha256=digest(y_path)))
    require(len(cases) == 6, 'A117 pilot panel differs')
    return cases


def holdout_cases(folder, root, plan):
    from utils.a56_selection_r2 import collect_usage
    from utils.frozen_direction_transfer import eligible_paths, select

    names = {case['image_id'] for case in plan['cases']}
    hashes = {case['image_sha256'] for case in plan['cases']}
    audit = []
    history = set()
    for base in (PROJECTS / 'results', PROJECTS / 'server_records', PARENT.parent.parent):
        for pattern in ('plan.json', 'run.json'):
            history.update(p.resolve() for p in base.rglob(pattern) if not p.resolve().is_relative_to(root))
    for path in sorted(history):
        result = collect_usage(read(path), {'ffhq': folder})
        if result['used_names']['ffhq'] or result['used_hashes']:
            audit.append(dict(path=str(path), sha256=digest(path)))
            names.update(result['used_names']['ffhq'])
            hashes.update(result['used_hashes'])
    require(bool(audit), 'No experimental history available for image exclusion')
    candidates = [dict(image=str(path.resolve()), image_id=path.stem, image_sha256=digest(path))
                  for path in eligible_paths(folder)]
    images = select(candidates, names, hashes, 10, 20261120)
    cases = []
    for image in images:
        number = int(image['image_id'])
        for task in TASKS:
            cases.append(dict(image, task=task, sigma=.05, seed=1000 + number,
                measurement_seed=701000 + number, mask_seed=2001000 + number,
                key=f'{task}_{image["image_id"]}_n050_s{1000 + number}'))
    return cases, dict(eligible_pngs=len(candidates), archived_names=len(names),
                       archived_hashes=len(hashes), files=audit, selection_seed=20261120)


def plan_run(args, root):
    label, arms = PANELS[args.stage]
    parent = args.parent.resolve()
    require(root != parent and not root.is_relative_to(parent) and not parent.is_relative_to(root),
            'Result directory must be independent of A102')
    old = read(parent / 'plan.json')
    old_summary = read(parent / 'summary.json')
    require(old['experiment'] == rules.EXPERIMENT, 'Wrong A102 parent')
    require(old_summary['completed_cases'] == old_summary['planned_cases'] == 40,
            'Need completed A102')
    require(old['grid'] == baseline.files.grid(), 'A102 grid differs')
    checkpoint = PROJECTS / 'modules/models/ffhq_10m.pt'
    require(digest(checkpoint) == old['checkpoint']['sha256'] == CHECKPOINTS['ffhq']['sha256'],
            'FFHQ checkpoint differs')
    kernel = parent / 'motion_kernel.npy'
    require(digest(kernel) == old['motion_kernel_sha256'], 'A102 kernel differs')
    configs = {task: rules.configuration(task) for task in TASKS}
    require(all(configs[t] == old['configs'][t] for t in TASKS), 'A102 configuration differs')
    card = PROJECTS / 'experiment_cards' / CARDS[label]
    source_paths = [Path(__file__), card, PROJECTS / 'run_thesis_feedback_20261120.sh',
                    *(Path(m.__file__) for m in (baseline, baseline.runtime, baseline.fixtures, rules,
                                                algebra, baseline.runtime.repaired, baseline.runtime.scalar_rule)),
                    PROJECTS / 'modules/zaps_algorithm.py', PROJECTS / 'modules/wavelet.py',
                    PROJECTS / 'utils/test_thesis_feedback_confirmation.py']
    frozen = dict(experiment=label, arms=list(arms), configs=configs, grid=old['grid'],
        motion_kernel_sha256=old['motion_kernel_sha256'], checkpoint_sha256=digest(checkpoint),
        parent=str(parent), parent_plan_sha256=digest(parent / 'plan.json'),
        ffhq_dir=str(args.ffhq_dir.resolve()), sources={str(p.relative_to(PROJECTS)): digest(p) for p in source_paths})
    plan_path = root / 'plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        require(all(plan.get(k) == v for k, v in frozen.items()) and
                digest(root / 'motion_kernel.npy') == old['motion_kernel_sha256'],
                'Frozen plan differs; original result directory retained')
        return plan
    require(not root.exists() or not any(root.iterdir()), 'Result directory is nonempty')
    cases, audit = ((pilot_cases(parent, old), None) if args.stage == 'pilot'
                    else holdout_cases(args.ffhq_dir.resolve(), root, old))
    require(shutil.disk_usage(root.parent).free >= 4 * 1024**3, 'Need 4 GiB available')
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(kernel, root / 'motion_kernel.npy')
    shutil.copyfile(card, root / 'experiment_card.md')
    plan = dict(**frozen, cases=cases, holdout_audit=audit, method_nfe=300,
                joint_forward_cap=len(cases) * (270 + 30 * len(arms)),
                code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                    cwd=PROJECTS, text=True).strip())
    write(plan_path, plan)
    print(f'{label} frozen: {len(cases)} cases, {plan["joint_forward_cap"]} forwards', flush=True)
    return plan


def results(root, plan):
    with patch('utils.diag_imagenet_closed_blur.ARMS', tuple(plan['arms'])):
        return validated_rows(root, plan, baseline.validate)


def summarize(root, plan, status, **extra):
    import numpy as np

    rows = results(root, plan)
    counts = [c for row in rows for c in (row['counts']['prefix'],
                                          *(a['counts'] for a in row['arms'].values()))]
    comparisons = ((PIXEL, DIAG), (SCALAR, FIXED), (SCALAR, NOSHRINK)) if plan['experiment'] == 'A117' else (
        (SCALAR, BASE), (DIAG, BASE), (DIAG, SCALAR))
    groups = []
    for task in TASKS:
        cell = [r for r in rows if r['case']['task'] == task]
        if not cell:
            continue
        quality = {arm: dict(**{k: mean(r['arms'][arm]['metrics'][k] for r in cell) for k in METRICS},
                             seconds=mean(r['arms'][arm]['method_equivalent_seconds'] for r in cell))
                   for arm in plan['arms']}
        pairs = []
        bootstrap = (np.random.default_rng(20261120).integers(0, len(cell), size=(20000, len(cell)))
                     if plan['experiment'] == 'A118' and len(cell) == 10 else None)
        for left, right in comparisons:
            deltas = {k: [r['arms'][left]['metrics'][k] - r['arms'][right]['metrics'][k] for r in cell] for k in METRICS}
            differences = {k: mean(v) for k, v in deltas.items()}
            wins = sum(value > 0 for value in deltas['psnr'])
            passed = (len(cell) == 10 and wins >= 7 and differences['psnr'] >= .1
                      and differences['ssim'] >= 0 and differences['lpips'] <= 0)
            intervals = (
                {k: np.quantile(np.asarray(v)[bootstrap].mean(axis=1), [.025, .975]).tolist()
                 for k, v in deltas.items()} if bootstrap is not None else None)
            pairs.append(dict(candidate=left, reference=right, delta=differences, psnr_wins=wins,
                              per_image={r['case']['image_id']: {k: v[i] for k, v in deltas.items()}
                                         for i, r in enumerate(cell)},
                              paired_bootstrap_95=intervals,
                              meets_confirm_criterion=passed if plan['experiment'] == 'A118' else None))
        group = dict(task=task, images=len(cell), arms=quality, paired=pairs)
        if PIXEL in plan['arms']:
            trace = [step for r in cell for step in read(root / 'cases' / r['case']['key'] / (PIXEL + '_trace.json'))]
            group['pixel_audit'] = dict(positions=len(trace),
                full_support_positions=sum(s['nonzero_support_fraction'] == 1. for s in trace),
                max_update_relative_error=max(s['same_state_correction_relative_error'] for s in trace),
                max_increment_angle_degrees=max((s['increment_angle_degrees'] for s in trace
                                                 if s['increment_angle_degrees'] is not None), default=None))
        groups.append(group)
    report = dict(experiment=plan['experiment'], status=status, completed_cases=len(rows),
        planned_cases=len(plan['cases']), method_nfe=300, joint_forward_cap=plan['joint_forward_cap'],
        actual_completed_cost={k: sum(c.get(k, 0) for c in counts) for k in
                               ('forwards','optimizer_backward_attempts','input_vjp_attempts','input_jvp_attempts')},
        cost_is_lower_bound=status != 'complete', groups=groups, **extra)
    write(root / 'summary.json', report)
    return report


def execute(args, root):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a67_model_freeze import frozen_network

    plan = plan_run(args, root)
    done = {r['case']['key'] for r in results(root, plan)}
    if len(done) == len(plan['cases']):
        return summarize(root, plan, 'complete')
    for case in plan['cases']:
        require(case['key'] in done or not (root / 'cases' / case['key']).exists(),
                f'Existing incomplete case retained: {case["key"]}')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible and ',' not in visible and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Expose one allocated GPU with CUDA_VISIBLE_DEVICES')
    gpu = available_gpus(gpu_inventory(), visible, visible)[0]
    require(torch.cuda.mem_get_info(0)[0] >= 6 * 1024**3, 'Need an allocated GPU with at least 6 GiB free')
    summarize(root, plan, 'running', device_uuid=gpu['uuid'])

    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        with ExitStack() as stack:
            stack.enter_context(frozen_network(model.model))
            stack.enter_context(patch.object(baseline, 'rules', SimpleNamespace(ARMS=tuple(plan['arms']))))
            original_observation = baseline.fixtures.observation
            if args.stage == 'pilot':
                stack.enter_context(patch.object(baseline.runtime, 'run_suffix', adapted_suffix))

            def observation(run_root, case, device):
                if args.stage != 'pilot':
                    return original_observation(run_root, case, device)
                source = Path(case['observation_path'])
                require(digest(source) == case['observation_sha256'], 'Original observation changed')
                dest = run_root / 'observations' / case['key']
                dest.mkdir(parents=True, exist_ok=True)
                for name in ('observation.png','ground_truth.png'):
                    shutil.copyfile(source.parent / name, dest / name)
                row = dict(path=str(source), sha256=case['observation_sha256'], sigma=case['sigma'])
                write(dest / 'observation.json', row)
                return row

            stack.enter_context(patch.object(baseline.fixtures, 'observation', observation))
            for case in plan['cases']:
                require(digest(case['image']) == case['image_sha256'], f'Input image changed: {case["key"]}')
                if case['key'] in done:
                    continue
                baseline.run_case(root, plan, case, model, gpu['uuid'])
                done.add(case['key'])
                summarize(root, plan, 'running', device_uuid=gpu['uuid'])
                print(f'{plan["experiment"]} progress {len(done)}/{len(plan["cases"])}', flush=True)
    except BaseException as error:
        summarize(root, plan, 'failed', error=repr(error), device_uuid=gpu['uuid'])
        raise
    return summarize(root, plan, 'complete', device_uuid=gpu['uuid'])


def archive(root, label):
    from utils import archive_refinement_confirmation as exporter

    report = read(root / 'summary.json')
    require(report['experiment'] == label and report['status'] == 'complete'
            and report['completed_cases'] == report['planned_cases'], 'Complete report required')
    exporter.BRANCH = f'results/{label.lower()}-thesis-feedback-20261120'
    bundle, commit = exporter.export_and_commit({label: root}, projects=PROJECTS)
    subprocess.run(['git', 'push', 'origin', exporter.BRANCH], cwd=PROJECTS, check=True)
    print(f'{label} RESULTS_PUSH_PASS: {commit} {bundle}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=PANELS)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--ffhq-dir', type=Path, default=Path('/home/lzy/FFHQ/00000/00000'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    label = PANELS[args.stage][0]
    root = (args.root or PARENT.with_name(f'{label}_20261120')).resolve()
    lock = root.with_name(root.name + '.lock')
    if args.status:
        report = read(root / 'summary.json') if (root / 'summary.json').is_file() else None
        print(json.dumps(dict(active=lock.exists() and busy(lock), summary=report), ensure_ascii=False, indent=2))
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(lock):
        if args.archive:
            archive(root, label)
        else:
            report = execute(args, root)
            print(f'{label} SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
