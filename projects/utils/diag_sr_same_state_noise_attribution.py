"""A112 measures observation noise alignment on the original scalar trajectory."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback
from unittest.mock import patch

import numpy as np

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils.diag_300nfe_feedback_confirmation import ARMS, ROOTS, TOL, select_cases
from utils.diag_sr_inpainting_followup import A103, source_case
from utils.paper_clock_alignment import digest, grid, read, require, write

ROOT = A103.with_name('A112_20261111')
CARD = PROJECTS/'experiment_cards/A112_sr_same_state_noise_attribution.md'
BRANCH = 'results/zaps-sr-noise-attribution-20261111'
SIGN_MARGIN = 1e-10  # model-domain squared-error per observed entry
DELTAS = ('proxy', 'clean', 'noisy', 'noise_credit', 'anchor_credit', 'proxy_without_noise')


def measurement_deltas(base, scalar, diagonal, noisy, truth, omega):
    """Return diagonal-minus-scalar MSE and two exact target-offset terms."""
    base, scalar, diagonal, noisy, truth = (
        np.asarray(x, dtype=np.float64) for x in (base, scalar, diagonal, noisy, truth))
    require(len({x.shape for x in (base, scalar, diagonal, noisy, truth)}) == 1 and base.size > 0
            and all(np.isfinite(x).all() for x in (base, scalar, diagonal, noisy, truth))
            and 0 <= omega <= 1, 'Invalid measurement diagnostic input')
    offset = diagonal-scalar
    noise = noisy-truth
    target = base+omega*(noisy-base)

    def delta(reference):
        return float(np.mean(np.square(diagonal-reference)-np.square(scalar-reference)))

    clean, proxy, observed = delta(truth), delta(target), delta(noisy)
    noise_credit = float(2*omega*np.mean(noise*offset))
    anchor_credit = float(2*(1-omega)*np.mean((base-truth)*offset))
    no_noise = delta(truth+(1-omega)*(base-truth))
    identity = max(abs(proxy-clean+noise_credit+anchor_credit), abs(no_noise-proxy-noise_credit),
                   abs(observed-clean+2*float(np.mean(noise*offset))))
    require(identity <= 1e-12*max(1., abs(clean), abs(proxy), abs(observed)),
            'Measurement-loss decomposition failed')
    return dict(proxy=proxy, clean=clean, noisy=observed, noise_credit=noise_credit,
                anchor_credit=anchor_credit, proxy_without_noise=no_noise, identity_error=identity)


class ScalarTrajectoryProbe:
    def __init__(self, diagonal_update):
        self.diagonal_update = diagonal_update
        self.measurements = []

    def update(self, zaps, operator, clean, q, residual, alpha, sigma, counts):
        import torch

        diagonal, detail, counts = self.diagonal_update(
            zaps, operator, clean, q, residual, alpha, sigma, counts)
        gamma = detail['scalar']['gamma']
        scalar = gamma*q
        with torch.no_grad():
            scale = alpha.sqrt().clamp(min=1e-8)
            measures = (operator.H(clean), operator.H(clean+scalar/scale),
                        operator.H(clean+diagonal/scale))
            self.measurements.append(tuple(x.detach().cpu().numpy() for x in measures))
        return scalar, detail, counts


def read_steps(probe, trace, noisy, truth):
    require(len(probe.measurements) == len(trace) == 30, 'Expected thirty same-state pairs')
    points = []
    for (base, scalar, diagonal), step in zip(probe.measurements, trace):
        w = step['scalar']['weight']
        losses = measurement_deltas(base, scalar, diagonal, noisy, truth, w)
        initial, final = step['proxy_initial'], step['proxy_final']
        proxy_error = abs(losses['proxy'] * base.size / 2 - (final-initial))
        require(proxy_error <= TOL * max(1., initial, final), 'Probe and solver disagree on proxy change')
        descends = losses['proxy'] < -SIGN_MARGIN
        conflict = descends and losses['clean'] > SIGN_MARGIN
        points.append(dict(step=step['step']+1, t=step['t'], next_t=step['next_t'], omega=w,
                           gamma=step['scalar']['gamma'], deltas=losses, solver_proxy_error=proxy_error,
                           proxy_descends=descends, proxy_clean_conflict=conflict,
                           conflict_loses_descent_without_noise=conflict and
                           losses['proxy_without_noise'] >= -SIGN_MARGIN))
    return points


def summarize_points(points):
    return dict(positions=len(points), proxy_descents=sum(p['proxy_descends'] for p in points),
                proxy_clean_conflicts=sum(p['proxy_clean_conflict'] for p in points),
                conflicts_losing_descent_without_noise=sum(
                    p['conflict_loses_descent_without_noise'] for p in points),
                mean_delta={k:mean(p['deltas'][k] for p in points) for k in DELTAS})


def summarize(rows):
    groups = []
    for sigma in sorted({row['case']['sigma'] for row in rows}):
        cell = [row for row in rows if row['case']['sigma'] == sigma]
        all_points = [p for row in cell for p in row['steps']]
        declining = [p for p in all_points if p['proxy_descends']]
        images_with_clean_increase = 0
        for row in cell:
            clean_changes = [p['deltas']['clean'] for p in row['steps'] if p['proxy_descends']]
            if clean_changes and mean(clean_changes) > SIGN_MARGIN:
                images_with_clean_increase += 1
        groups.append(dict(sigma=sigma, images=len(cell), all_steps=summarize_points(all_points),
            proxy_descending_steps=summarize_points(declining) if declining else None,
            images_with_mean_clean_increase_on_proxy_descents=images_with_clean_increase,
            fixed_state_clean_error_increase=len(cell) == 10 and images_with_clean_increase >= 7
            and bool(declining) and mean(p['deltas']['clean'] for p in declining) > SIGN_MARGIN,
            by_stage={label:summarize_points([p for p in all_points if start <= p['step'] <= stop])
                      for label, start, stop in (('first10', 1, 10), ('middle10', 11, 20), ('last10', 21, 30))},
            by_step=[dict(step=k, **summarize_points([p for p in all_points if p['step'] == k]))
                     for k in range(1, 31)],
            max_replay_relative_l2=max(row['replay_relative_l2'] for row in cell)))
    return groups


def execute(parent, reference, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import equal_nfe_feedback_runtime_r3 as bridge, historical_ffhq_runtime as precision
    from utils import paper_clock_runtime as clock
    from utils.a67_model_freeze import frozen_network
    from utils.diag_shared_prefix_feedback import validate
    from utils.diag_shared_prefix_feedback_multinoise import check
    from utils.shared_prefix_feedback_runtime_multinoise import make_operator, run_suffix, train_prefix

    plan = check(parent)
    reference_plan = read(reference/'plan.json')
    reference_summary = read(reference/'summary.json')
    cases = select_cases('sr', plan)
    require(reference_plan['experiment'] == 'A110' and reference_summary['status'] == 'complete'
            and reference_summary['completed_cases'] == len(cases)
            and reference_plan['parent_sha256'] == digest(parent/'plan.json')
            and [entry['case'] for entry in reference_plan['cases']] == cases, 'A110 provenance differs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 6 * 1024**3, 'Need 6 GiB free')
    require(plan['grid'] == grid() and digest(PROJECTS/'modules/models/ffhq_10m.pt') ==
            plan['checkpoint']['sha256'], 'Grid or checkpoint differs')
    require(all(not root.resolve().is_relative_to(p.resolve()) for p in (parent, reference)),
            'Use an independent A112 result directory')
    root.mkdir(parents=True)
    write(root/'plan.json', dict(experiment='A112', parent=str(parent), reference=str(reference),
          cases=reference_plan['cases'], grid=plan['grid'], checkpoint=plan['checkpoint'],
          a103_plan_sha256=digest(parent/'plan.json'), a110_plan_sha256=digest(reference/'plan.json'),
          a110_summary_sha256=digest(reference/'summary.json'), sign_margin=SIGN_MARGIN,
          code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
          source_sha256=digest(__file__), card_sha256=digest(CARD), joint_forward_cap=6000))
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(experiment='A112', status='running', planned_cases=len(cases), completed_cases=0)
    rows, active = [], None
    write(root/'summary.json', report)
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        descending = plan['grid'][::-1]
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            for entry in reference_plan['cases']:
                case = entry['case']
                dest = root/'cases'/case['key']/'run.json'
                active = dict(status='running', case=case, counts={})
                write(dest, active)
                folder, _, y_path = source_case(parent, case)
                require(digest(folder/'run.json') == entry['parent_record_sha256'] and
                        digest(y_path) == entry['observation_sha256'], 'A110 observation or case changed')
                y_cpu = torch.load(y_path, map_location='cpu', weights_only=True)['y']
                y = y_cpu.to(model.device)
                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    op = make_operator(case, model.device, count=count)
                    prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, op, y, case))
                validate(count, 270, 9)
                require(count['calls'] == descending*9, 'Prefix grid differs')
                probe = ScalarTrajectoryProbe(bridge.exact_diagonal_with_native_straight_through)
                with patch.object(bridge, 'exact_diagonal_with_native_straight_through', probe.update), \
                        clock.counters(model.model) as count:
                    active['counts']['suffix'] = count
                    op = make_operator(case, model.device, count=count)
                    (output, detail), suffix_seconds = clock.clocks(torch.cuda.synchronize,
                        lambda: run_suffix(model, op, y, case, ARMS[2], prefix))
                validate(count, 30, 1)
                require(count['calls'] == descending, 'Suffix grid differs')

                # Offline only, after the full scalar trajectory has finished.
                with torch.no_grad():
                    evaluator = make_operator(case, model.device)
                    truth = evaluator.H(load_image_as_tensor(case['image']).to(model.device)).cpu().numpy()
                active['steps'] = read_steps(probe, detail['trace'], y_cpu.numpy(), truth)
                reference_case = reference/'cases'/case['key']
                frozen_output = torch.load(reference_case/(ARMS[1]+'.pt'), map_location='cpu', weights_only=True)
                require(frozen_output.shape == output.shape and bool(torch.isfinite(frozen_output).all()),
                        'Invalid A110 scalar reference')
                error = float((output.double()-frozen_output.double()).norm()/
                              frozen_output.double().norm().clamp(min=1e-30))
                active.update(status='complete', replay_relative_l2=error, module_counts=detail['module_counts'],
                              prefix_seconds=prefix_seconds, audit_suffix_seconds=suffix_seconds)
                write(dest, active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), groups=summarize(rows))
                write(root/'summary.json', report)
                print(f'A112 COMPLETE: {len(rows)}/{len(cases)} {case["key"]}', flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if active is not None:
            active.update(status='failed', error=repr(error))
            write(root/'cases'/active['case']['key']/'run.json', active)
        raise
    finally:
        attempted = rows + ([active] if active is not None else [])
        counts = [count for row in attempted for count in row['counts'].values()]
        report['actual_cost'] = {k:sum(c.get(k, 0) for c in counts) for k in
            ('forward_attempts', 'forwards', 'optimizer_backward_attempts', 'input_vjp_attempts', 'input_jvp_attempts')}
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(root/'summary.json', report)
    print('A112 SUMMARY:', json.dumps(dict(report, groups=[{k:v for k,v in g.items() if k != 'by_step'}
                                       for g in report['groups']]), ensure_ascii=False), flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter

    require(read(root/'summary.json')['status'] == 'complete', 'A112 must be complete for archive')
    related = {'A109': A103.with_name('A109_20261111'), 'A110': ROOTS['sr'], 'A111': ROOTS['blur']}
    sources = {'A112': root}
    for label, source in related.items():
        if (source/'summary.json').is_file() and read(source/'summary.json').get('status') == 'complete':
            sources[label] = source
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit(sources, projects=PROJECTS)
    print(f'Archive: {bundle} Results commit: {commit} Branch: {BRANCH}', flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--a110-root', type=Path, default=ROOTS['sr'])
    parser.add_argument('--root', type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.root/'summary.json'), ensure_ascii=False, indent=2))
    elif args.archive:
        archive(args.root)
    else:
        execute(args.parent, args.a110_root, args.root)


if __name__ == '__main__':
    main()
