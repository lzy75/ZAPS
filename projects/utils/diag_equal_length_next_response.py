"""A107 compares equal-length directions against a common next-denoiser objective."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils.diag_sr_inpainting_followup import A103, IMAGES, TASKS, source_case
from utils.diag_sr_soft_directional_300 import ROOT as A106, DIAG, SOFT
from utils.paper_clock_alignment import digest, read, require, write

ROOT = A103.with_name('A107_20261111')
CARD = PROJECTS / 'experiment_cards/A107_equal_length_next_response.md'
POSITIONS = (9, 19, 28)  # zero-based; all three have a subsequent network call
ARMS = ('SCALAR_ANCHOR', 'PROXY_DIRECTION', 'NEXT_RESPONSE_DIRECTION')
RELATIVE_LENGTH = .10


def proxy_audit(root):
    summary = read(root/'summary.json')
    require(summary['status'] == 'complete', 'Expected completed A106')
    for arm, initial, final in ((DIAG, 'proxy_initial', 'proxy_final'),
                                (SOFT, 'soft_proxy_initial', 'soft_proxy_final')):
        paths = sorted((root/'cases').glob(f'*/{arm}_trace.json'))
        require(len(paths) == 6, 'Expected six completed trace files')
        steps = [step for path in paths for step in read(path)]
        require(len(steps) == 180, 'Expected 30 positions per trace')
        increases = [step[final]-step[initial] for step in steps]
        tolerance = 128 * 2**-23
        print('A106 SAME-STATE PROXY:', arm, dict(positions=len(steps),
              strict_decreases=sum(v < 0 for v in increases), positive_roundoff=sum(v > 0 for v in increases),
              tolerance_violations=sum(v > tolerance*max(1., step[initial])
                                       for v, step in zip(increases, steps))), flush=True)


def tangent(zaps, value, native):
    """Project into native nonzero wavelet support and remove the scalar ray."""
    import torch
    c = zaps.dwt.analysis(native)
    u = zaps.dwt.analysis(value) * (c != 0)
    c64, u64 = c.double(), u.double()
    energy = c64.square().sum()
    ratio = (u64*c64).sum() / energy.clamp_min(1e-30)
    return zaps.dwt.synthesis((u64-ratio*c64).to(value.dtype))


def readout(operator, y, gt, mask, outputs):
    import torch
    from utils.shrink_multitask_runtime import informative
    from utils.terminal_mask_counterfactual import region_errors

    rows = {}
    gt = gt.to(y.device)
    with torch.no_grad():
        clean_y = informative(operator.H(gt), operator).double()
        noisy_y = informative(y, operator).double()
        for name, output in outputs.items():
            measured = informative(operator.H(output), operator).double()
            full = float((output.double()-gt.double()).square().mean())
            row = dict(full_mse=full, psnr=10*math.log10(4/max(full, 1e-30)),
                       clean_H_mse=float((measured-clean_y).square().mean()),
                       noisy_y_mse=float((measured-noisy_y).square().mean()))
            if mask is not None:
                errors = region_errors(output, gt, mask)
                row.update(observed_mse=errors['mse_observed'], missing_mse=errors['mse_missing'])
            rows[name] = row
    return rows


def probe(model, zaps, operator, y, q, residual, clean, anchor, target_step, alpha, sigma, count):
    import torch
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.equal_nfe_feedback_runtime_r1 import exact_adjoint_diagonal
    from utils.response_target_runtime import predict_clean
    from utils.shrink_multitask_runtime import informative

    module_counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
    full, diag, _ = exact_adjoint_diagonal(zaps, operator, clean, q, residual, alpha, sigma, module_counts)
    scalar = diag['scalar']
    with torch.no_grad():
        r = informative(residual, operator)
        b = informative(operator.H(clean), operator) + scalar['weight']*r
        q_norm = float(q.double().norm())
        proxy = tangent(zaps, full - scalar['gamma']*q, q)
        proxy_norm = float(proxy.double().norm())
        length = RELATIVE_LENGTH * scalar['gamma'] * q_norm
    require(all(math.isfinite(v) for v in (q_norm, proxy_norm, length)), 'Nonfinite tangent')
    if length <= 1e-12 or proxy_norm <= max(1e-12, 1e-6*q_norm):
        return dict(active=False, reason='zero_anchor_or_proxy_tangent',
                    proxy_tangent_norm=proxy_norm), None
    with torch.enable_grad(), input_vjp_parameters(model.model):
        inp = anchor.detach().requires_grad_(True)
        m0 = predict_clean(model, inp, target_step)
        error = informative(operator.H(m0), operator) - b
        count['input_vjp_attempts'] += 1
        gradient = -torch.autograd.grad(.5*error.double().square().sum(), inp)[0].detach()
        m0 = m0.detach()
    with torch.no_grad():
        direction = tangent(zaps, gradient, q)
        size = float(direction.double().norm())
        require(math.isfinite(size) and math.isfinite(length), 'Nonfinite tangent')
        if size <= 1e-12:
            return dict(active=False, reason='zero_true_tangent'), None
        d_proxy = proxy * (length / proxy_norm)
        d_next = direction * (length / size)
        relative_length_error = abs(float(d_proxy.double().norm()-d_next.double().norm()))/length
        require(relative_length_error <= 1e-5, 'Candidate lengths differ')
        outputs = dict(zip(ARMS, (m0, predict_clean(model, anchor+d_proxy, target_step),
                                  predict_clean(model, anchor+d_next, target_step))))
        target_losses = {k:float(.5*(informative(operator.H(v), operator)-b).double().square().sum())
                         for k, v in outputs.items()}
        cosine = float((d_proxy.double()*d_next.double()).sum()) / length**2
    return dict(active=True, common_length=length, direction_cosine=cosine,
                relative_length_error=relative_length_error, actual_target_losses=target_losses,
                proxy_loss_at_scalar=diag['proxy_initial'], proxy_loss_at_full_diagonal=diag['proxy_final']), outputs


def summarize(rows):
    groups = []
    for task in TASKS:
        images = [r for r in rows if r['case']['task'] == task]
        if not images:
            continue
        active = [[p for p in r['positions'] if p['active']] for r in images]
        pairs = []
        for reference in ARMS[:2]:
            per_image = []
            for image, points in zip(images, active):
                if not points:
                    continue
                keys = points[0]['quality'][reference]
                deltas = {k: mean(p['quality'][ARMS[2]][k]-p['quality'][reference][k] for p in points)
                          for k in keys}
                deltas['actual_target_loss'] = mean(
                    p['actual_target_losses'][ARMS[2]]-p['actual_target_losses'][reference] for p in points)
                per_image.append(dict(image=image['case']['image_id'], n=len(points), deltas=deltas))
            avg = {k:mean(r['deltas'][k] for r in per_image)
                   for k in per_image[0]['deltas']} if per_image else {}
            both = sum(r['deltas']['full_mse'] < 0 and r['deltas']['clean_H_mse'] < 0 for r in per_image)
            missing_safe = (task != 'inpainting' or (avg.get('observed_mse', 1) <= 0
                                                       and avg.get('missing_mse', 1) < 0))
            pairs.append(dict(reference=reference, per_image=per_image, mean_delta=avg,
                both_mse_improved_images=both, positive=len(per_image) == 3 and both >= 2
                and avg['full_mse'] < 0 and avg['clean_H_mse'] < 0 and missing_safe))
        cosines = [p['direction_cosine'] for points in active for p in points]
        groups.append(dict(task=task, images=len(images), active_positions=sum(map(len, active)),
                           planned_positions=len(images)*len(POSITIONS),
                           mean_direction_cosine=mean(cosines) if cosines else None, comparisons=pairs))
    return groups


def execute(parent, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import historical_ffhq_runtime as precision, paper_clock_runtime as clock
    from utils import overall_direction_audit as native, response_target_runtime as rt
    from utils.noise_signal_shrink import select
    from utils.a67_model_freeze import frozen_network
    from utils.diag_shared_prefix_feedback import validate
    from utils.diag_shared_prefix_feedback_multinoise import check
    from utils.shared_prefix_feedback_runtime import make_operator, train_prefix
    from utils.shrink_multitask_runtime import informative

    plan = check(parent)
    cases = [case for case in plan['cases'] if case['task'] in TASKS
             and case['image_id'] in IMAGES and case['sigma'] == .10]
    require(len(cases) == 6 and {(c['task'], c['image_id']) for c in cases} ==
            {(t, i) for t in TASKS for i in IMAGES}, 'Need exact six cases')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 8 * 1024**3, 'Need 8 GiB free for VJP')
    checkpoint = PROJECTS/'modules/models/ffhq_10m.pt'
    require(digest(checkpoint) == plan['checkpoint']['sha256'], 'FFHQ checkpoint differs')
    require(not root.resolve().is_relative_to(parent.resolve()), 'Use independent results')
    inputs = {}
    for c in cases:
        folder, _, y_path = source_case(parent, c)
        inputs[c['key']] = dict(record=digest(folder/'run.json'), observation=digest(y_path))
        if c['task'] == 'inpainting':
            inputs[c['key']]['mask'] = digest(y_path.parent/'mask.pt')
    root.mkdir(parents=True)
    write(root/'plan.json', dict(experiment='A107_equal_length_next_response', parent=str(parent),
        parent_plan_sha256=digest(parent/'plan.json'), inputs=inputs, cases=cases,
        positions_zero_based=POSITIONS, arms=ARMS, length_relative_to_scalar=RELATIVE_LENGTH,
        card_sha256=digest(CARD),
        source_sha256=digest(__file__), checkpoint=plan['checkpoint'], grid=plan['grid'],
        new_nfe_cap=1854, input_vjp_cap=18))
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(status='running', planned_cases=6, completed_cases=0)
    write(root/'summary.json', report)
    rows, current = [], None
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            for case in cases:
                key = case['key']
                folder, _, y_path = source_case(parent, case)
                require(digest(folder/'run.json') == inputs[key]['record'] and
                        digest(y_path) == inputs[key]['observation'], 'Frozen case changed')
                dest = root/'cases'/key/'run.json'
                current = dict(case=case, status='running', positions=[], counts={})
                write(dest, current)
                load = lambda path:torch.load(path, map_location='cpu', weights_only=True).to(model.device)
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                operator = make_operator(case, model.device)
                mask = getattr(operator, '_mask_cache', None)
                if mask is not None:
                    mask_path = y_path.parent/'mask.pt'
                    require(digest(mask_path) == inputs[key]['mask'] and
                            torch.equal(mask, load(mask_path)), 'Frozen mask differs')
                with clock.counters(model.model) as count:
                    current['counts']['prefix'] = count
                    prefix = train_prefix(model, operator, y, case)
                validate(count, 270, 9)
                require(count['calls'] == plan['grid'][::-1]*9, 'Prefix grid differs')
                z = native.make_z(model, operator, prefix['entry'], case['task'])
                native.restore_rng(prefix['entry'], model.device)
                x = prefix['entry']['initial'].to(model.device)
                saved_predictions = []
                with torch.no_grad(), clock.counters(model.model) as count:
                    current['counts']['trace_and_probes'] = count
                    for k, i in enumerate(range(29, -1, -1)):
                        t, s = int(z.tau[i]), int(z.tau[i-1]) if i else -1
                        uncond, q, clean = native.native_parts(model, z, x, y, i)
                        r = y-operator.H(clean)
                        a = model.alphas_cumprod[t]
                        response = informative(operator.H(clean+q/a.sqrt().clamp(min=1e-8))
                                               - operator.H(clean), operator).double()
                        residual = informative(r, operator).double()
                        scalar = select(float((residual*response).sum()), float(residual.square().sum()),
                                        float(response.square().sum()), residual.numel(), case['sigma'])
                        x = (uncond + scalar['gamma']*q).detach()
                        if k in POSITIONS:
                            before = native.rng_state(model.device)
                            item, outputs = probe(model, z, operator, y, q, r, clean, x, s, a,
                                                  case['sigma'], count)
                            require(rt.rng_equal(before, native.rng_state(model.device)), 'Probe changed sampling RNG')
                            item.update(step=k+1, t=t, next_t=s)
                            current['positions'].append(item)
                            if outputs is not None:
                                saved_predictions.append((item, outputs))
                expected_calls = []
                by_step = {p['step']: p for p in current['positions']}
                for k, t in enumerate(plan['grid'][::-1], 1):
                    expected_calls.append(t)
                    item = by_step.get(k)
                    if item is not None:
                        expected_calls += [item['next_t']] * (3 if item['active'] else
                                                           int(item['reason'] == 'zero_true_tangent'))
                require(count['calls'] == expected_calls and count['forwards'] == count['forward_attempts']
                        and count['optimizer_backward_attempts'] == 0
                        and count['input_vjp_attempts'] == sum(
                            p['active'] or p.get('reason') == 'zero_true_tangent'
                            for p in current['positions']), 'Diagnostic budget differs')
                gt = load_image_as_tensor(case['image'])
                for item, outputs in saved_predictions:
                    item['quality'] = readout(operator, y, gt, mask, outputs)
                current['status'] = 'complete'
                write(dest, current)
                rows.append(current)
                current = None
                report.update(completed_cases=len(rows), groups=summarize(rows))
                write(root/'summary.json', report)
                print('A107 COMPLETE:', key, len(saved_predictions), '/', len(POSITIONS), flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if current is not None:
            current.update(status='failed', error=repr(error))
            write(root/'cases'/current['case']['key']/'run.json', current)
        raise
    finally:
        all_rows = rows + ([current] if current is not None else [])
        counts = [c for row in all_rows for c in row['counts'].values()]
        report['recorded_new_nfe'] = sum(c['forward_attempts'] for c in counts)
        report['input_vjp_attempts'] = sum(c['input_vjp_attempts'] for c in counts)
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(root/'summary.json', report)
    print('A107 SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit-a106', type=Path, nargs='?', const=A106)
    args = parser.parse_args()
    if args.audit_a106 is not None:
        proxy_audit(args.audit_a106)
    else:
        execute(args.parent, args.root)


if __name__ == '__main__':
    main()
