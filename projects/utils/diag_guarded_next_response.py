"""A109 checks measured next-response descent and its effect on terminal quality."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils import diag_online_three_position_response as a108
from utils.diag_equal_length_next_response import POSITIONS, RELATIVE_LENGTH, readout, tangent
from utils.diag_sr_inpainting_followup import A103, IMAGES, METRICS, TASKS, source_case
from utils.paper_clock_alignment import digest, read, require, write

ROOT = A103.with_name('A109_20261111')
CARD = PROJECTS/'experiment_cards/A109_guarded_next_response.md'
SCALAR, RAY, DIAG = a108.SCALAR, 'GUARDED_RAY306', 'GUARDED_DIAG306'
ARMS = (SCALAR, RAY, DIAG)
RELATIVE_LOSS_TOLERANCE = 64 * 2**-23


def decision(anchor_loss, trial_loss, nonzero_step):
    require(all(math.isfinite(loss) and loss >= 0 for loss in (anchor_loss, trial_loss)),
            'Invalid next-response loss')
    if not nonzero_step:
        return 'zero_step'
    margin = RELATIVE_LOSS_TOLERANCE * max(anchor_loss, trial_loss, 1e-30)
    decrease = anchor_loss - trial_loss
    if decrease > margin:
        return 'accepted'
    if decrease < -margin:
        return 'increased'
    return 'within_tolerance'


def guarded_suffix(model, operator, y, case, arm, prefix, count, grid, trace):
    import torch
    from utils import overall_direction_audit as native, response_target_runtime as rt
    from utils.noise_signal_shrink import select
    from utils.shrink_multitask_runtime import informative

    require(arm in (RAY, DIAG), 'Unknown guarded arm')
    zaps = native.make_z(model, operator, prefix['entry'], case['task'])
    native.restore_rng(prefix['entry'], model.device)
    x = prefix['entry']['initial'].to(model.device)
    expected_calls = []
    count.update(module_dwt_analysis=0, module_dwt_synthesis=0)
    with torch.no_grad():
        for k, t in enumerate(grid):
            uncond, q, clean = native.native_parts(model, zaps, x, y, len(grid)-1-k)
            expected_calls.append(t)
            y0 = operator.H(clean)
            residual = informative(y-y0, operator).double()
            scale = model.alphas_cumprod[t].sqrt().clamp(min=1e-8)
            response = informative(operator.H(clean+q/scale)-y0, operator).double()
            scalar = select(float((residual*response).sum()), float(residual.square().sum()),
                            float(response.square().sum()), residual.numel(), case['sigma'])
            anchor = uncond + scalar['gamma']*q
            x = anchor
            if k not in POSITIONS:
                continue

            s = grid[k+1]
            target = informative(y0, operator) + scalar['weight']*residual.to(y0.dtype)
            before = native.rng_state(model.device)
            gradient, anchor_loss = a108.response_direction(model, operator, anchor, target, s, count)
            if arm == RAY:
                dot = float((gradient.double()*q.double()).sum())
                direction = (1. if dot > 0 else -1. if dot < 0 else 0.)*q
            else:
                direction = tangent(zaps, gradient, q)
                count['module_dwt_analysis'] += 2
                count['module_dwt_synthesis'] += 1
            length = RELATIVE_LENGTH * scalar['gamma'] * float(q.double().norm())
            update, nonzero_step = a108.apply_length(direction, length)
            candidate = anchor + update
            error = informative(operator.H(rt.predict_clean(model, candidate, s)), operator) - target
            trial_loss = float(.5 * error.double().square().sum())
            expected_calls.extend((s, s))
            require(rt.rng_equal(before, native.rng_state(model.device)),
                    'Response evaluation changed sampling RNG')
            outcome = decision(anchor_loss, trial_loss, nonzero_step)
            accepted = outcome == 'accepted'
            if accepted:
                x = candidate
            trace.append(dict(step=k+1, t=t, next_t=s, outcome=outcome, gamma=scalar['gamma'],
                              target_length=length, trial_length=float(update.double().norm()),
                              anchor_loss=anchor_loss, trial_loss=trial_loss,
                              selected_loss=trial_loss if accepted else anchor_loss))

    require(count['calls'] == expected_calls and len(trace) == len(POSITIONS)
            and count['forwards'] == count['forward_attempts'] == len(grid)+2*len(POSITIONS)
            and count['input_vjp_attempts'] == len(POSITIONS)
            and count['optimizer_backward_attempts'] == count.get('input_jvp_attempts', 0) == 0,
            'Guarded suffix budget differs')
    require(bool(torch.isfinite(x).all()), 'Nonfinite reconstruction')
    return x.detach().cpu(), native.rng_state(model.device)


def summarize(rows):
    groups = []
    for task in TASKS:
        cases = [row for row in rows if row['case']['task'] == task]
        if not cases:
            continue
        arms = {}
        for arm in ARMS:
            results = [row['arms'][arm] for row in cases]
            stats = dict(quality={k:mean(r['quality'][k] for r in results) for k in METRICS},
                         nfe=results[0]['nfe'], input_vjp=results[0]['input_vjp'],
                         mean_method_seconds=mean(r['method_seconds'] for r in results))
            if arm != SCALAR:
                trials = [row['traces'][arm] for row in cases]
                points = [p for trace in trials for p in trace]
                stats.update(trials=len(points), outcomes=dict(Counter(p['outcome'] for p in points)),
                             images_with_accepted_step=sum(any(p['outcome'] == 'accepted' for p in trace)
                                                           for trace in trials),
                             selected_loss_increases=sum(p['selected_loss'] > p['anchor_loss'] for p in points),
                             mean_selected_relative_drop=mean(
                                 (p['anchor_loss']-p['selected_loss'])/max(p['anchor_loss'], 1e-30)
                                 for p in points))
            arms[arm] = stats
        comparisons = []
        for candidate, reference in ((DIAG, RAY), (DIAG, SCALAR), (RAY, SCALAR)):
            pairs = [(row['arms'][candidate], row['arms'][reference]) for row in cases]
            delta = {k:mean(a['quality'][k]-b['quality'][k] for a, b in pairs) for k in METRICS}
            errors = {k:mean(a['errors'][k]-b['errors'][k] for a, b in pairs)
                      for k in sorted(set(pairs[0][0]['errors']) - {'psnr'})}
            wins = sum(a['quality']['psnr'] > b['quality']['psnr'] for a, b in pairs)
            equal_budget = all((a['nfe'], a['input_vjp']) == (b['nfe'], b['input_vjp'])
                               for a, b in pairs)
            region_pass = task != 'inpainting' or (errors['missing_mse'] < 0 and errors['observed_mse'] <= 0)
            comparisons.append(dict(candidate=candidate, reference=reference, delta=delta, errors=errors,
                psnr_wins=wins, same_nfe_and_vjp=equal_budget,
                positive=len(cases) == 3 and wins >= 2 and delta['psnr'] >= .1 and delta['ssim'] >= 0
                and delta['lpips'] <= 0 and errors['clean_H_mse'] < 0 and region_pass
                and (reference != RAY or equal_budget)))
        groups.append(dict(task=task, images=len(cases), arms=arms, comparisons=comparisons))
    return groups


def execute(parent, previous, root):
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
    old = read(previous/'plan.json')
    report108 = read(previous/'summary.json')
    require(old['experiment'] == 'A108_online_three_position_response' and report108['status'] == 'complete'
            and report108['completed_cases'] == 6 and Path(old['parent']).resolve() == parent.resolve()
            and old['grid'] == plan['grid'] and old['checkpoint'] == plan['checkpoint']
            and old['positions_zero_based'] == list(POSITIONS) and old['relative_length'] == RELATIVE_LENGTH,
            'Completed matching A108 required')
    cases = old['cases']
    require(len(cases) == 6 and all(case in plan['cases'] and case['sigma'] == .10 for case in cases)
            and {(c['task'], c['image_id']) for c in cases} == {(t, i) for t in TASKS for i in IMAGES},
            'A108 panel differs from A103')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 8*1024**3, 'Need 8 GiB free for VJP')
    require(digest(PROJECTS/'modules/models/ffhq_10m.pt') == plan['checkpoint']['sha256'],
            'Frozen FFHQ checkpoint differs')
    require(all(not root.resolve().is_relative_to(p.resolve()) for p in (parent, previous)),
            'Use an independent A109 result directory')
    root.mkdir(parents=True)
    write(root/'plan.json', dict(experiment='A109_guarded_next_response', parent=str(parent),
          a103_plan_sha256=digest(parent/'plan.json'), a108_plan_sha256=digest(previous/'plan.json'),
          a108_summary_sha256=digest(previous/'summary.json'), cases=cases, inputs=old['inputs'],
          arms=ARMS, positions_zero_based=POSITIONS, relative_length=RELATIVE_LENGTH, grid=plan['grid'],
          relative_loss_tolerance=RELATIVE_LOSS_TOLERANCE, checkpoint=plan['checkpoint'],
          new_nfe_cap=2232, input_vjp_cap=36, optimizer_backward_cap=54,
          source_sha256=digest(__file__), card_sha256=digest(CARD),
          code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip()))
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(status='running', planned_cases=len(cases), completed_cases=0)
    write(root/'summary.json', report)
    rows, active = [], None
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        grid = plan['grid'][::-1]
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            for case in cases:
                key = case['key']
                dest = root/'cases'/key
                active = dict(case=case, status='running', counts={}, arms={}, traces={})
                write(dest/'run.json', active)
                folder, _, y_path = source_case(parent, case)
                hashes = old['inputs'][key]
                require(digest(folder/'run.json') == hashes['record']
                        and digest(y_path) == hashes['observation'], 'A108 observation or case record differs')
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                expected_mask = None
                if case['task'] == 'inpainting':
                    mask_path = y_path.parent/'mask.pt'
                    require(digest(mask_path) == hashes['mask'], 'A108 mask differs')
                    expected_mask = torch.load(mask_path, map_location='cpu', weights_only=True)

                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    op = make_operator(case, model.device, count=count)
                    (prefix, prefix_seconds) = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, op, y, case))
                validate(count, 270, 9)
                require(count['calls'] == grid*9, 'Prefix grid differs')

                outputs, final_rng = {}, None
                for arm in ARMS:
                    trace = active['traces'][arm] = []
                    with clock.counters(model.model) as count:
                        active['counts'][arm] = count
                        op = make_operator(case, model.device, count=count)
                        mask = getattr(op, '_mask_cache', None)
                        if expected_mask is not None:
                            require(mask is not None and torch.equal(mask.cpu(), expected_mask),
                                    'Rebuilt mask differs')
                        if arm == SCALAR:
                            (output, _, rng), seconds = clock.clocks(torch.cuda.synchronize,
                                lambda: a108.online_suffix(model, op, y, case, arm, prefix, count, grid))
                        else:
                            (output, rng), seconds = clock.clocks(torch.cuda.synchronize,
                                lambda: guarded_suffix(model, op, y, case, arm, prefix, count, grid, trace))
                    if final_rng is None:
                        final_rng = rng
                    else:
                        require(rng_equal(final_rng, rng), 'Final RNG differs')
                    outputs[arm] = output
                    active['arms'][arm] = dict(nfe=270+count['forwards'], input_vjp=count['input_vjp_attempts'],
                        method_seconds=prefix_seconds+seconds, suffix_seconds=seconds)

                gt = load_image_as_tensor(case['image']).cpu()
                errors = readout(op, y, gt, mask, {arm:output.to(model.device) for arm, output in outputs.items()})
                for arm, output in outputs.items():
                    active['arms'][arm].update(quality=metrics(output, gt), errors=errors[arm])
                    torch.save(output, dest/(arm+'.pt'))
                active['status'] = 'complete'
                write(dest/'run.json', active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), groups=summarize(rows))
                write(root/'summary.json', report)
                print('A109 COMPLETE:', key, {arm:dict(Counter(p['outcome'] for p in trace))
                      for arm, trace in rows[-1]['traces'].items() if trace}, flush=True)
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
        report.update(recorded_new_nfe=sum(c['forward_attempts'] for c in counts),
                      input_vjp_attempts=sum(c['input_vjp_attempts'] for c in counts),
                      optimizer_backward_attempts=sum(c['optimizer_backward_attempts'] for c in counts),
                      cost_is_lower_bound=report['status'] != 'complete')
        write(root/'summary.json', report)
    print('A109 SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--a108-root', type=Path, default=a108.ROOT)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.root/'summary.json'), ensure_ascii=False, indent=2))
    else:
        execute(args.parent, args.a108_root, args.root)


if __name__ == '__main__':
    main()
