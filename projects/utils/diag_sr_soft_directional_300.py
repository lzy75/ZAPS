"""A106 paired SR-only test of soft noise shrinkage on the extra diagonal step."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import traceback
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils.diag_sr_inpainting_followup import A103, IMAGES, METRICS, source_case
from utils.paper_clock_alignment import digest, read, require, write

EXPERIMENT = 'A106_sr_soft_directional_300'
ROOT = A103.with_name('A106_20261111')
A102 = A103.with_name('A102_20261110')
CARD = PROJECTS / 'experiment_cards/A106_sr_soft_directional_300.md'
BASE, SCALAR, DIAG, SOFT = ('ZAPS_SUFFIX', 'CLOSED_SCALAR_SUFFIX',
                           'CLOSED_DIAG_SUFFIX', 'SOFT_DIAG_SUFFIX')
ARMS = (BASE, SCALAR, DIAG, SOFT)


def reliability(scalar, size, sigma):
    """Nominal noise energy after the positive-ray projection."""
    require(size > 1 and math.isfinite(sigma) and sigma > 0, 'Invalid SR measurement noise')
    if not scalar['resolved'] or scalar['weight'] == 0:
        return 0., 0., size * sigma**2
    residual_energy = scalar['residual_energy']
    dot = scalar['residual_response_dot']
    response_energy = scalar['response_energy']
    fitted = dot > 0
    projected_energy = dot**2 / response_energy if fitted else 0.
    remaining = max(0., residual_energy - projected_energy)
    noise = (size - int(fitted)) * sigma**2
    return remaining / (remaining + noise), remaining, noise


def soft_update(zaps, operator, clean, q, residual, alpha, sigma, counts):
    import torch
    from utils.equal_nfe_feedback_runtime_r1 import exact_adjoint_diagonal

    full, detail, counts = exact_adjoint_diagonal(
        zaps, operator, clean, q, residual, alpha, sigma, counts)
    scalar = detail['scalar']
    beta, remaining, noise = reliability(scalar, residual.numel(), sigma)
    # Follow the same gradient bridge as the original diagonal arm.
    with torch.no_grad():
        start = scalar['gamma'] * q.detach()
        extra = full - start
        candidate = start + beta * extra
        a = alpha.sqrt().clamp(min=1e-8)
        y0 = operator.H(clean)
        target = scalar['weight'] * residual.double()
        errors = [(operator.H(clean + delta/a) - y0).double() - target
                  for delta in (start, candidate)]
        old_loss, new_loss = [float(.5 * e.square().sum()) for e in errors]
        slack = 128 * torch.finfo(clean.dtype).eps * max(1., old_loss)
        require(math.isfinite(new_loss) and new_loss <= old_loss + slack,
                'Damped update failed same-state proxy descent')
        relative_extra = float((candidate-start).double().norm() /
                               start.double().norm().clamp_min(1e-30))
    detail.update(beta=beta, orthogonal_residual_energy=remaining,
                  nominal_remaining_noise_energy=noise,
                  soft_proxy_initial=old_loss, soft_proxy_final=new_loss,
                  relative_applied_extra=relative_extra)
    return candidate + (q - q.detach()), detail, counts


def summarize(rows):
    result = []
    for sigma in (.05, .10):
        group = [row for row in rows if row['case']['sigma'] == sigma]
        if not group:
            continue
        comparisons = []
        for candidate, reference in ((SCALAR, BASE), (DIAG, SCALAR), (SOFT, DIAG), (SOFT, SCALAR)):
            delta = {metric: mean(row['arms'][candidate]['metrics'][metric] -
                                  row['arms'][reference]['metrics'][metric] for row in group)
                     for metric in METRICS}
            errors = {name: mean(row['arms'][candidate]['errors'][name] -
                                  row['arms'][reference]['errors'][name] for row in group)
                      for name in ('clean_H_mse', 'noisy_y_mse')}
            wins = sum(row['arms'][candidate]['metrics']['psnr'] >
                       row['arms'][reference]['metrics']['psnr'] for row in group)
            comparisons.append(dict(candidate=candidate, reference=reference, delta=delta,
                                    errors=errors, psnr_wins=wins,
                                    positive=len(group) == 3 and wins >= 2 and delta['psnr'] >= .10
                                    and delta['ssim'] >= 0 and delta['lpips'] <= 0
                                    and (candidate != SOFT or errors['clean_H_mse'] < 0)))
        result.append(dict(sigma=sigma, n=len(group), comparisons=comparisons,
                           median_case_beta=median(row['arms'][SOFT]['median_beta'] for row in group),
                           steps_with_extra=sum(row['arms'][SOFT]['effective_steps'] for row in group)))
    return result


def execute(normal, high, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import diag_shared_prefix_feedback as base, paper_clock_runtime as clock
    from utils import historical_ffhq_runtime as precision, equal_nfe_feedback_runtime_r3 as repaired
    from utils.a67_model_freeze import frozen_network
    from utils.diag_noise_feasible_zaps import metrics
    from utils.shared_prefix_feedback_runtime_multinoise import make_operator, run_suffix, train_prefix

    normal_plan = base.check(normal)
    # The legacy A103 driver binds A102's module globals at import time.
    from utils.diag_shared_prefix_feedback_multinoise import check as check_multinoise
    plans = (normal_plan, check_multinoise(high))
    require(plans[0]['grid'] == plans[1]['grid'] and plans[0]['checkpoint'] == plans[1]['checkpoint']
            and plans[0]['configs'] == plans[1]['configs'], 'A102/A103 protocol differs')
    selected = []
    for parent, plan, sigma in zip((normal, high), plans, (.05, .10)):
        cases = [case for case in plan['cases'] if case['task'] == 'super_resolution'
                 and case['image_id'] in IMAGES and case['sigma'] == sigma]
        require(len(cases) == 3 and {case['image_id'] for case in cases} == set(IMAGES),
                'Expected all three registered images per noise')
        for case in cases:
            folder, _, y_path = source_case(parent, case)
            selected.append(dict(case=case, parent=str(parent), y_path=str(y_path),
                                 y_sha256=digest(y_path), record_sha256=digest(folder/'run.json')))
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Expose one assigned CUDA GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 6 * 1024**3, 'Need at least 6 GiB free')
    require(digest(PROJECTS/'modules/models/ffhq_10m.pt') == plans[0]['checkpoint']['sha256'],
            'Frozen FFHQ checkpoint differs')
    require(all(not root.resolve().is_relative_to(p.resolve()) for p in (normal, high)),
            'Use an independent A106 output directory')
    root.mkdir(parents=True)
    write(root/'plan.json', dict(experiment=EXPERIMENT, cases=selected, arms=ARMS,
          parent_plan_sha256={str(p): digest(p/'plan.json') for p in (normal, high)},
          code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
          source_sha256=digest(__file__), card_sha256=digest(CARD), grid=plans[0]['grid'],
          checkpoint=plans[0]['checkpoint'], method_nfe=300, total_new_nfe_cap=2340,
          torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0)))
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(experiment=EXPERIMENT, status='running', planned_cases=6, completed_cases=0)
    write(root/'summary.json', report)
    rows, active = [], None
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        grid = plans[0]['grid'][::-1]
        with precision.precision('LEGACY_IEEE'), frozen_network(model.model):
            for entry in selected:
                case = entry['case']
                folder, _, y_path = source_case(Path(entry['parent']), case)
                require(str(y_path) == entry['y_path'] and digest(y_path) == entry['y_sha256'] and
                        digest(folder/'run.json') == entry['record_sha256'], 'Frozen inputs changed')
                dest = root/'cases'/case['key']
                active = dict(case=case, status='running', arms={}, counts={})
                write(dest/'run.json', active)
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    op = make_operator(case, model.device, count=count)
                    prefix, prefix_time = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, op, y, case))
                base.validate(count, 270, 9)
                require(count['calls'] == grid*9, 'Prefix grid differs')
                final_rng, outputs = None, {}
                for arm in ARMS:
                    suffix_arm = DIAG if arm == SOFT else arm
                    update = soft_update if arm == SOFT else repaired.exact_diagonal_with_native_straight_through
                    with patch.object(repaired, 'exact_diagonal_with_native_straight_through', update), \
                            clock.counters(model.model) as count:
                        active['counts'][arm] = count
                        op = make_operator(case, model.device, count=count)
                        (output, detail), seconds = clock.clocks(torch.cuda.synchronize,
                            lambda: run_suffix(model, op, y, case, suffix_arm, prefix))
                    base.validate(count, 30, 1)
                    require(count['calls'] == grid, 'Suffix grid differs')
                    if final_rng is None:
                        final_rng = detail['final_rng']
                    else:
                        require(base.rng_equal(final_rng, detail['final_rng']), 'Final RNG differs')
                    outputs[arm] = output
                    row = dict(method_seconds=prefix_time+seconds, suffix_seconds=seconds,
                               module_counts=detail['module_counts'])
                    if arm == SOFT:
                        row.update(median_beta=median(step['beta'] for step in detail['trace']),
                                   effective_steps=sum(step['relative_applied_extra'] > 1e-6
                                                       for step in detail['trace']))
                    active['arms'][arm] = row
                    write(dest/(arm+'_trace.json'), detail['trace'])
                # Ground truth is only available to the evaluator.
                gt = load_image_as_tensor(case['image']).cpu()
                op = make_operator(case, model.device)
                truth = op.H(gt.to(model.device)).double()
                for arm, output in outputs.items():
                    prediction = op.H(output.to(model.device)).double()
                    active['arms'][arm].update(metrics=metrics(output, gt),
                        errors=dict(clean_H_mse=float((prediction-truth).square().mean()),
                                    noisy_y_mse=float((prediction-y.double()).square().mean())))
                    torch.save(output, dest/(arm+'.pt'))
                active['status'] = 'complete'
                write(dest/'run.json', active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), groups=summarize(rows))
                write(root/'summary.json', report)
                print('A106 COMPLETE:', case['key'], flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if active is not None:
            active.update(status='failed', error=repr(exc))
            write(root/'cases'/active['case']['key']/'run.json', active)
        raise
    finally:
        attempts = rows + ([active] if active is not None else [])
        report['recorded_new_nfe'] = sum(count['forward_attempts'] for row in attempts
                                         for count in row['counts'].values())
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(root/'summary.json', report)
    print('A106 SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a102-root', type=Path, default=A102)
    parser.add_argument('--a103-root', type=Path, default=A103)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.root/'summary.json'), ensure_ascii=False, indent=2))
    else:
        execute(args.a102_root, args.a103_root, args.root)


if __name__ == '__main__':
    main()
