"""A108 applies the A107 directions at three positions and reads terminal quality."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils.diag_equal_length_next_response import (
    ROOT as A107, POSITIONS, RELATIVE_LENGTH, readout, tangent,
)
from utils.diag_sr_inpainting_followup import A103, IMAGES, TASKS, METRICS, source_case
from utils.paper_clock_alignment import digest, read, require, write

ROOT = A103.with_name('A108_20261111')
CARD = PROJECTS/'experiment_cards/A108_online_three_position_response.md'
BASE, SCALAR, PROXY, TRUE_RAY, TRUE_DIAG = (
    'ZAPS300', 'SCALAR300', 'PROXY_DIAG300', 'TRUE_RAY303', 'TRUE_DIAG303')
ARMS = (BASE, SCALAR, PROXY, TRUE_RAY, TRUE_DIAG)
RESPONSE_ARMS = (TRUE_RAY, TRUE_DIAG)


def apply_length(direction, length):
    norm = float(direction.double().norm())
    require(math.isfinite(norm) and math.isfinite(length), 'Nonfinite direction or step length')
    if length <= 1e-12 or norm <= 1e-12:
        return direction * 0, False
    update = direction * (length/norm)
    require(abs(float(update.double().norm())-length) <= 1e-5*length, 'Step length mismatch')
    return update, True


def response_direction(model, operator, anchor, target, next_t, count):
    import torch
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.response_target_runtime import predict_clean
    from utils.shrink_multitask_runtime import informative

    with torch.enable_grad(), input_vjp_parameters(model.model):
        inp = anchor.detach().requires_grad_(True)
        clean = predict_clean(model, inp, next_t)
        error = informative(operator.H(clean), operator) - target
        loss = .5*error.double().square().sum()
        count['input_vjp_attempts'] += 1
        direction = -torch.autograd.grad(loss, inp)[0].detach()
    require(bool(torch.isfinite(direction).all()), 'Nonfinite true-response direction')
    return direction, float(loss.detach())


def online_suffix(model, operator, y, case, arm, prefix, count, grid):
    import torch
    from utils import overall_direction_audit as native, response_target_runtime as rt
    from utils.equal_nfe_feedback_runtime_r1 import exact_adjoint_diagonal
    from utils.noise_signal_shrink import select
    from utils.shrink_multitask_runtime import informative

    zaps = native.make_z(model, operator, prefix['entry'], case['task'])
    native.restore_rng(prefix['entry'], model.device)
    x = prefix['entry']['initial'].to(model.device)
    trace, expected_calls = [], []
    sigma = case['sigma']
    proxy_counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
    with torch.no_grad():
        for k, t in enumerate(grid):
            next_t = grid[k+1] if k < len(grid)-1 else -1
            uncond, q, clean = native.native_parts(model, zaps, x, y, len(grid)-1-k)
            expected_calls.append(t)
            if arm == BASE:
                x = uncond + q
                continue
            y0 = operator.H(clean)
            residual = y-y0
            r = informative(residual, operator).double()
            alpha = model.alphas_cumprod[t]
            d = informative(operator.H(clean+q/alpha.sqrt().clamp(min=1e-8))-y0, operator).double()
            scalar = select(float((r*d).sum()), float(r.square().sum()),
                            float(d.square().sum()), r.numel(), sigma)
            anchor = uncond + scalar['gamma']*q
            x = anchor
            if k not in POSITIONS or arm == SCALAR:
                continue
            q_norm = float(q.double().norm())
            length = RELATIVE_LENGTH * scalar['gamma'] * q_norm
            item = dict(step=k+1, t=t, next_t=next_t, gamma=scalar['gamma'], target_length=length)
            if arm == PROXY:
                diagonal, _, _ = exact_adjoint_diagonal(
                    zaps, operator, clean, q, residual, alpha, sigma, proxy_counts)
                direction = tangent(zaps, diagonal-scalar['gamma']*q, q)
                if float(direction.double().norm()) <= max(1e-12, 1e-6*q_norm):
                    direction = direction*0
            else:
                target = informative(y0, operator) + scalar['weight']*r.to(y0.dtype)
                before = native.rng_state(model.device)
                gradient, loss = response_direction(model, operator, anchor, target, next_t, count)
                expected_calls.append(next_t)
                require(rt.rng_equal(before, native.rng_state(model.device)), 'VJP consumed sampling RNG')
                if arm == TRUE_RAY:
                    dot = float((gradient.double()*q.double()).sum())
                    direction = (1. if dot > 0 else -1. if dot < 0 else 0.)*q
                else:
                    direction = tangent(zaps, gradient, q)
                item['actual_target_loss_at_anchor'] = loss
            update, applied = apply_length(direction, length)
            x = anchor+update
            item.update(applied=applied, applied_length=float(update.double().norm()),
                        cosine_to_native=float((update.double()*q.double()).sum()) /
                        max(float(update.double().norm())*q_norm, 1e-30))
            trace.append(item)
    required_vjp = len(POSITIONS) if arm in RESPONSE_ARMS else 0
    require(count['calls'] == expected_calls and count['forwards'] == count['forward_attempts']
            == len(grid)+required_vjp and count['input_vjp_attempts'] == required_vjp
            and count['optimizer_backward_attempts'] == 0
            and count.get('input_jvp_attempts', 0) == 0, 'Online suffix budget differs')
    require(bool(torch.isfinite(x).all()), 'Nonfinite reconstruction')
    return x.detach().cpu(), trace, native.rng_state(model.device)


def summarize(rows):
    comparisons = ((SCALAR, BASE), (PROXY, SCALAR), (TRUE_DIAG, SCALAR),
                   (TRUE_DIAG, PROXY), (TRUE_RAY, SCALAR), (TRUE_DIAG, TRUE_RAY))
    groups = []
    for task in TASKS:
        cases = [row for row in rows if row['case']['task'] == task]
        if not cases:
            continue
        results = []
        for cand, ref in comparisons:
            pairs = [(row['arms'][cand], row['arms'][ref]) for row in cases]
            delta = {k:mean(a['quality'][k]-b['quality'][k] for a, b in pairs) for k in METRICS}
            keys = set(pairs[0][0]['errors']) - {'psnr'}
            errors = {k:mean(a['errors'][k]-b['errors'][k] for a, b in pairs) for k in sorted(keys)}
            wins = sum(a['quality']['psnr'] > b['quality']['psnr'] for a, b in pairs)
            same_budget = all(a['nfe'] == b['nfe'] and a['input_vjp'] == b['input_vjp'] for a, b in pairs)
            active_pairs = sum(a['applied_steps'] == b['applied_steps'] == len(POSITIONS) for a, b in pairs)
            region_pass = (task != 'inpainting' or
                           (errors['missing_mse'] < 0 and errors['observed_mse'] <= 0))
            results.append(dict(candidate=cand, reference=ref, delta=delta, errors=errors,
                                psnr_wins=wins, same_nfe_and_vjp=same_budget,
                                all_three_actions_in_both_images=active_pairs,
                                positive=len(cases) == 3 and wins >= 2 and delta['psnr'] >= .1
                                and delta['ssim'] >= 0 and delta['lpips'] <= 0
                                and errors['clean_H_mse'] < 0 and region_pass
                                and (ref != TRUE_RAY or (same_budget and active_pairs == len(cases)))))
        groups.append(dict(task=task, images=len(cases), comparisons=results,
            arms={arm:dict(mean_quality={k:mean(row['arms'][arm]['quality'][k] for row in cases) for k in METRICS},
                           mean_method_seconds=mean(row['arms'][arm]['method_seconds'] for row in cases),
                           nfe=cases[0]['arms'][arm]['nfe'], input_vjp=cases[0]['arms'][arm]['input_vjp'])
                  for arm in ARMS}))
    return groups


def execute(parent, a107_root, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import historical_ffhq_runtime as precision, paper_clock_runtime as clock
    from utils.a67_model_freeze import frozen_network
    from utils.diag_noise_feasible_zaps import metrics
    from utils.diag_shared_prefix_feedback import validate
    from utils.diag_shared_prefix_feedback_multinoise import check
    from utils.response_target_runtime import rng_equal
    from utils.shared_prefix_feedback_runtime import make_operator, train_prefix

    plan = check(parent)
    probe_plan = read(a107_root/'plan.json')
    probe_summary = read(a107_root/'summary.json')
    require(probe_plan['parent_plan_sha256'] == digest(parent/'plan.json')
            and probe_plan['positions_zero_based'] == list(POSITIONS)
            and probe_plan['length_relative_to_scalar'] == RELATIVE_LENGTH, 'A107 protocol differs')
    require(probe_summary['status'] == 'complete' and probe_summary['completed_cases'] == 6
            and {g['task'] for g in probe_summary['groups']} == set(TASKS)
            and all(all(c['positive'] for c in g['comparisons']) for g in probe_summary['groups']),
            'A107 diagnostic did not pass')
    cases = [c for c in plan['cases'] if c['task'] in TASKS and c['sigma'] == .10
             and c['image_id'] in IMAGES]
    require(len(cases) == 6 and cases == probe_plan['cases'], 'A107 panel differs')
    inputs = {}
    for case in cases:
        folder, _, y_path = source_case(parent, case)
        record = dict(record=digest(folder/'run.json'), observation=digest(y_path))
        if case['task'] == 'inpainting':
            record['mask'] = digest(y_path.parent/'mask.pt')
        require(record == probe_plan['inputs'][case['key']], 'A107 input hashes differ')
        inputs[case['key']] = record
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 8*1024**3, 'Need 8 GiB free for VJP')
    require(digest(PROJECTS/'modules/models/ffhq_10m.pt') == plan['checkpoint']['sha256'],
            'FFHQ checkpoint differs')
    require(all(not root.resolve().is_relative_to(p.resolve()) for p in (parent, a107_root)),
            'Use an independent A108 result directory')
    root.mkdir(parents=True)
    write(root/'plan.json', dict(experiment='A108_online_three_position_response',
        parent=str(parent), a107=str(a107_root), inputs=inputs, cases=cases, arms=ARMS,
        a107_plan_sha256=digest(a107_root/'plan.json'), a107_summary_sha256=digest(a107_root/'summary.json'),
        checkpoint=plan['checkpoint'], positions_zero_based=POSITIONS, grid=plan['grid'],
        relative_length=RELATIVE_LENGTH, new_nfe_cap=2556, input_vjp_cap=36,
        source_sha256=digest(__file__), card_sha256=digest(CARD),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip()))
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(status='running', planned_cases=6, completed_cases=0)
    write(root/'summary.json', report)
    rows, active = [], None
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        grid = plan['grid'][::-1]
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            for case in cases:
                key = case['key']
                folder, _, y_path = source_case(parent, case)
                require(digest(folder/'run.json') == inputs[key]['record'] and
                        digest(y_path) == inputs[key]['observation'], 'Frozen A108 case changed')
                dest = root/'cases'/key
                active = dict(case=case, status='running', counts={}, arms={})
                write(dest/'run.json', active)
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    op = make_operator(case, model.device, count=count)
                    (prefix, prefix_seconds) = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, op, y, case))
                validate(count, 270, 9)
                require(count['calls'] == grid*9, 'Prefix grid differs')
                final_rng, outputs = None, {}
                for arm in ARMS:
                    with clock.counters(model.model) as count:
                        active['counts'][arm] = count
                        op = make_operator(case, model.device, count=count)
                        mask = getattr(op, '_mask_cache', None)
                        if case['task'] == 'inpainting':
                            mask_path = y_path.parent/'mask.pt'
                            require(digest(mask_path) == inputs[key]['mask'] and torch.equal(
                                mask.cpu(), torch.load(mask_path, map_location='cpu', weights_only=True)),
                                'Frozen mask differs')
                        (payload, seconds) = clock.clocks(torch.cuda.synchronize,
                            lambda: online_suffix(model, op, y, case, arm, prefix, count, grid))
                    output, trace, rng = payload
                    if final_rng is None:
                        final_rng = rng
                    else:
                        require(rng_equal(final_rng, rng), 'Final RNG differs')
                    active['arms'][arm] = dict(nfe=270+count['forwards'], input_vjp=count['input_vjp_attempts'],
                        method_seconds=prefix_seconds+seconds, applied_steps=sum(p['applied'] for p in trace))
                    outputs[arm] = output
                    write(dest/(arm+'_trace.json'), trace)
                gt = load_image_as_tensor(case['image']).cpu()
                errors = readout(op, y, gt, mask, {arm:x.to(model.device) for arm, x in outputs.items()})
                for arm, output in outputs.items():
                    active['arms'][arm].update(quality=metrics(output, gt), errors=errors[arm])
                    torch.save(output, dest/(arm+'.pt'))
                active['status'] = 'complete'
                write(dest/'run.json', active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), groups=summarize(rows))
                write(root/'summary.json', report)
                print('A108 COMPLETE:', key, flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if active is not None:
            active.update(status='failed', error=repr(error))
            write(root/'cases'/active['case']['key']/'run.json', active)
        raise
    finally:
        attempted = rows + ([active] if active is not None else [])
        counts = [c for row in attempted for c in row['counts'].values()]
        report['recorded_new_nfe'] = sum(c['forward_attempts'] for c in counts)
        report['optimizer_backward_attempts'] = sum(c['optimizer_backward_attempts'] for c in counts)
        report['input_vjp_attempts'] = sum(c['input_vjp_attempts'] for c in counts)
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(root/'summary.json', report)
    print('A108 SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--a107-root', type=Path, default=A107)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.root/'summary.json'), ensure_ascii=False, indent=2))
    else:
        execute(args.parent, args.a107_root, args.root)


if __name__ == '__main__':
    main()
