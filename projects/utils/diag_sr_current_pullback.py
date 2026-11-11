"""A116 compares a reused current denoiser VJP with a true next response."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils.diag_equal_length_next_response import ROOT as A107, POSITIONS, RELATIVE_LENGTH, readout, tangent
from utils.diag_online_three_position_response import apply_length
from utils.diag_sr_inpainting_followup import A103, IMAGES, source_case
from utils.paper_clock_alignment import digest, read, require, write

ROOT = A103.with_name('A116_20261111')
CARD = PROJECTS/'experiment_cards/A116_sr_current_pullback.md'
BRANCH = 'results/zaps-sr-current-pullback-20261111'
ANCHOR, PROXY, CURRENT, NEXT = 'SCALAR_ANCHOR', 'PROXY_DIRECTION', 'CURRENT_PULLBACK', 'NEXT_RESPONSE_DIRECTION'
ARMS = (ANCHOR, PROXY, CURRENT, NEXT)
LENGTH_MARGIN = 1e-6


def pullback(output, inp, error, count, label):
    import torch

    count[label] = count.get(label, 0)+1
    count['input_vjp_attempts'] += 1
    update, = torch.autograd.grad(output, inp, grad_outputs=-error.detach().to(output.dtype))
    require(bool(torch.isfinite(update).all()), 'Nonfinite input pullback')
    return update.detach()


def scalar_rule(operator, clean, q, y, a, sigma):
    from utils.noise_signal_shrink import select

    measured = operator.H(clean)
    residual = y-measured
    d = (operator.H(clean+q/a)-measured).double()
    r = residual.double()
    scalar = select(float((r*d).sum()), float(r.square().sum()), float(d.square().sum()), r.numel(), sigma)
    return scalar, measured, residual


def probe_position(model, zaps, operator, x, y, i, sigma, count, module):
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    from utils import overall_direction_audit as native, response_target_runtime as rt
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.equal_nfe_feedback_runtime_r1 import exact_adjoint_diagonal

    t, s = int(zaps.tau[i]), int(zaps.tau[i-1])
    alpha = model.alphas_cumprod[t]
    a = alpha.sqrt().clamp(min=1e-8)
    with torch.enable_grad(), input_vjp_parameters(model.model):
        inp = x.detach().requires_grad_(True)
        times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model._predict_eps(inp, times)
        clean_graph = ((inp-(1-alpha).sqrt()*eps)/a).clamp(-1, 1)
        with torch.no_grad():
            clean = clean_graph.detach()
            require(torch.equal(clean, zaps._tweedie_estimate(x, eps.detach(), alpha, i)),
                    'Current forward value differs from native Tweedie estimate')
            uncond = ddpm_posterior_step(x, clean, t, s, model.alphas_cumprod,
                                        eta=1., learned_log_var=None, mode='ddpm')
            residual = y-operator.H(clean)
            v = operator.transpose(residual)
            hv = zaps.dwt.synthesis(zaps.D[i]*zaps.dwt.analysis(v))
            q = zaps.zeta[i]*((v+(1-alpha)*hv)/a)
            scalar, measured, residual = scalar_rule(operator, clean, q, y, a, sigma)
            full, detail, _ = exact_adjoint_diagonal(zaps, operator, clean, q, residual, alpha, sigma, module)
            require(detail['scalar'] == scalar, 'Scalar anchor differs from the proxy solve')
            correction = scalar['gamma']*q
            anchor = uncond+correction
            target = measured+scalar['weight']*residual
            # Freeze the unexplained target residual at the scalar anchor.
            e_virtual = operator.H(clean+correction/a)-target
            before = native.rng_state(model.device)
        current = pullback(operator.H(clean_graph), inp, e_virtual, count, 'current_input_vjp')
        del clean_graph, eps

    with torch.enable_grad(), input_vjp_parameters(model.model):
        future = anchor.detach().requires_grad_(True)
        predicted = rt.predict_clean(model, future, s)
        projected = operator.H(predicted)
        e_next = projected.detach()-target
        true = pullback(projected, future, e_next, count, 'next_input_vjp')
        anchor_prediction = predicted.detach()
        del predicted, projected
    with torch.no_grad():
        q_norm = float(q.double().norm())
        length = RELATIVE_LENGTH*scalar['gamma']*q_norm
        directions = {PROXY:tangent(zaps, full-correction, q), CURRENT:tangent(zaps, current, q),
                      NEXT:tangent(zaps, true, q)}
        module['tangent_dwt_analysis'] = module.get('tangent_dwt_analysis', 0)+6
        module['tangent_dwt_synthesis'] = module.get('tangent_dwt_synthesis', 0)+3
        changes, active = {}, {}
        for arm, direction in directions.items():
            changes[arm], active[arm] = apply_length(direction, length)
        if float(directions[PROXY].double().norm()) <= max(1e-12, LENGTH_MARGIN*q_norm):
            active[PROXY] = False
        outputs = {ANCHOR:anchor_prediction}
        virtual = {ANCHOR:operator.H(clean+correction/a)}
        for arm, change in changes.items():
            outputs[arm] = rt.predict_clean(model, anchor+change, s)
            virtual[arm] = operator.H(clean+(correction+change)/a)
        objective = {arm:float(.5*(operator.H(prediction)-target).double().square().sum())
                     for arm, prediction in outputs.items()}
        cosine = {arm:float((changes[arm].double()*changes[NEXT].double()).sum()) / length**2
                  if active[arm] and active[NEXT] else None for arm in (PROXY, CURRENT)}
        require(rt.rng_equal(before, native.rng_state(model.device)), 'Direction probes changed sampling RNG')
    return anchor, dict(t=t, next_t=s, common_length=length, active=active, cosine_to_next=cosine,
                        actual_target_losses=objective), outputs, virtual


def summarize(rows):
    comparisons = []
    for candidate, reference in ((CURRENT, ANCHOR), (CURRENT, PROXY), (NEXT, CURRENT), (NEXT, ANCHOR)):
        per_image = []
        for row in rows:
            points = [p for p in row['positions'] if all(p['active'].values())]
            if not points:
                continue
            metrics = {k:mean(p['quality'][candidate][k]-p['quality'][reference][k] for p in points)
                       for k in ('psnr', 'full_mse', 'clean_H_mse', 'noisy_y_mse')}
            metrics['virtual_clean_H_mse'] = mean(p['virtual_clean_H_mse'][candidate]-p['virtual_clean_H_mse'][reference]
                                                for p in points)
            metrics['actual_target_loss'] = mean(p['actual_target_losses'][candidate]-p['actual_target_losses'][reference]
                                                for p in points)
            per_image.append(dict(image_id=row['case']['image_id'], positions=len(points), delta=metrics))
        avg = {k:mean(r['delta'][k] for r in per_image) for k in per_image[0]['delta']} if per_image else {}
        wins = sum(r['delta']['full_mse'] < 0 and r['delta']['clean_H_mse'] < 0 for r in per_image)
        comparisons.append(dict(candidate=candidate, reference=reference, mean_delta=avg, per_image=per_image,
            both_mse_improved_images=wins, positive=len(per_image) == 3
            and all(r['positions'] == len(POSITIONS) for r in per_image)
            and wins >= 2 and avg['full_mse'] < 0 and avg['clean_H_mse'] < 0))
    points = [p for row in rows for p in row['positions']]
    valid = [p for p in points if all(p['active'].values())]
    return dict(images=len(rows), observed_positions=len(points), common_active_positions=len(valid),
                mean_cosine_to_next={arm:mean(p['cosine_to_next'][arm] for p in valid) if valid else None
                                     for arm in (PROXY, CURRENT)}, comparisons=comparisons)


def execute(parent, reference, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import historical_ffhq_runtime as precision, paper_clock_runtime as clock
    from utils import overall_direction_audit as native
    from utils.a67_model_freeze import frozen_network
    from utils.diag_shared_prefix_feedback import validate
    from utils.diag_shared_prefix_feedback_multinoise import check
    from utils.shared_prefix_feedback_runtime import make_operator, train_prefix

    plan = check(parent)
    old, old_report = read(reference/'plan.json'), read(reference/'summary.json')
    cases = [c for c in old['cases'] if c['task'] == 'super_resolution']
    require(old['experiment'] == 'A107_equal_length_next_response' and old_report['status'] == 'complete'
            and old['parent_plan_sha256'] == digest(parent/'plan.json') and old['grid'] == plan['grid']
            and old['positions_zero_based'] == list(POSITIONS) and old['length_relative_to_scalar'] == RELATIVE_LENGTH
            and len(cases) == 3 and {c['image_id'] for c in cases} == set(IMAGES)
            and all(c['sigma'] == .10 and c in plan['cases'] for c in cases), 'Completed matching A107 SR panel required')
    summary_path = root/'summary.json'
    if summary_path.is_file():
        result = read(summary_path)
        require(result['experiment'] == 'A116' and result['status'] == 'complete', 'Prior A116 attempt retained')
        print('A116 SUMMARY:', json.dumps(result, ensure_ascii=False), flush=True)
        return
    require(not root.exists() or root.is_dir() and not any(root.iterdir()), 'A116 result directory already contains files')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 8*1024**3, 'Need 8 GiB free for input VJP')
    require(digest(PROJECTS/'modules/models/ffhq_10m.pt') == plan['checkpoint']['sha256'], 'Checkpoint differs')
    root.mkdir(parents=True, exist_ok=True)
    write(root/'plan.json', dict(experiment='A116', cases=cases, arms=ARMS, checkpoint=plan['checkpoint'],
          a103_plan_sha256=digest(parent/'plan.json'), a107_plan_sha256=digest(reference/'plan.json'),
          positions_zero_based=POSITIONS, grid=plan['grid'], common_relative_length=RELATIVE_LENGTH,
          forward_cap=936, current_input_vjp_cap=9, next_input_vjp_cap=9, source_sha256=digest(__file__),
          card_sha256=digest(CARD), code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                                    cwd=PROJECTS, text=True).strip()))
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(experiment='A116', status='running', completed_cases=0, planned_cases=3)
    write(summary_path, report)
    rows, active = [], None
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        grid = plan['grid'][::-1]
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            for case in cases:
                key = case['key']
                folder, _, y_path = source_case(parent, case)
                require(old['inputs'][key] == dict(record=digest(folder/'run.json'), observation=digest(y_path)),
                        'Frozen A107 input changed')
                record = root/'cases'/key/'run.json'
                active = dict(case=case, status='running', counts={}, positions=[])
                write(record, active)
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    op = make_operator(case, model.device, count=count)
                    prefix = train_prefix(model, op, y, case)
                validate(count, 270, 9)
                require(count['calls'] == grid*9, 'Prefix grid differs')
                predictions, expected = [], []
                module = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
                with torch.no_grad(), clock.counters(model.model) as count:
                    active['counts']['suffix_and_probes'] = count
                    op = make_operator(case, model.device, count=count)
                    zaps = native.make_z(model, op, prefix['entry'], case['task'])
                    native.restore_rng(prefix['entry'], model.device)
                    x = prefix['entry']['initial'].to(model.device)
                    for k, t in enumerate(grid):
                        i = len(grid)-1-k
                        expected.append(t)
                        if k in POSITIONS:
                            x, point, outputs, virtual = probe_position(
                                model, zaps, op, x, y, i, case['sigma'], count, module)
                            expected.extend([grid[k+1]]*4)
                            point['step'] = k+1
                            active['positions'].append(point)
                            predictions.append((outputs, virtual))
                        else:
                            uncond, q, clean = native.native_parts(model, zaps, x, y, i)
                            a = model.alphas_cumprod[t].sqrt().clamp(min=1e-8)
                            scalar, _, _ = scalar_rule(op, clean, q, y, a, case['sigma'])
                            x = uncond+scalar['gamma']*q
                require(count['calls'] == expected and count['forward_attempts'] == count['forwards'] == 42
                        and count['current_input_vjp'] == count['next_input_vjp'] == 3
                        and count['input_vjp_attempts'] == 6 and count['optimizer_backward_attempts'] == 0,
                        'Probe budget differs')
                gt = load_image_as_tensor(case['image']).to(model.device)
                truth = op.H(gt).double()
                for point, (outputs, virtual) in zip(active['positions'], predictions):
                    point['quality'] = readout(op, y, gt, None, outputs)
                    point['virtual_clean_H_mse'] = {arm:float((pred.double()-truth).square().mean())
                                                   for arm, pred in virtual.items()}
                active.update(status='complete', module_counts=module)
                write(record, active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), group=summarize(rows))
                write(summary_path, report)
                print(f'A116 COMPLETE: {len(rows)}/3 {key}', flush=True)
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
        report['actual_cost'] = {k:sum(c.get(k, 0) for c in counts) for k in
            ('forward_attempts', 'forwards', 'current_input_vjp', 'next_input_vjp', 'optimizer_backward_attempts')}
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(summary_path, report)
    print('A116 SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    from utils.a54_session import exclusive

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--reference', type=Path, default=A107)
    parser.add_argument('--root', type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.status:
        print(json.dumps(read(root/'summary.json'), ensure_ascii=False, indent=2))
        return
    if args.archive:
        from utils import archive_refinement_confirmation as exporter
        result = read(root/'summary.json')
        require(result['experiment'] == 'A116' and result['status'] == 'complete', 'A116 must be complete')
        exporter.BRANCH = BRANCH
        bundle, commit = exporter.export_and_commit({'A116': root}, projects=PROJECTS)
        print(f'Archive: {bundle} Results commit: {commit} Branch: {BRANCH}', flush=True)
        subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
        return
    require(all(not root.is_relative_to(p.resolve()) for p in (args.parent, args.reference)), 'Use independent A116 results')
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(root.with_name(root.name+'.lock')):
        execute(args.parent, args.reference, root)


if __name__ == '__main__':
    main()
