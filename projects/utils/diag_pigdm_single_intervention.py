"""A128: one extra scalar or diagonal update on a frozen A126 PiGDM protocol."""
import argparse
import time
from pathlib import Path
from unittest.mock import patch

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a126_full_response as a126
from utils import diag_transfer_objective_audit as audit
from utils import full_coefficient_response as solver
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

TARGETS = (920, 90)
MODES = ('ray', 'diagonal')
ROOT = a126.ROOT/'audit_a128_pigdm_single_intervention'
PARENT = a126.ROOT/'A126_pigdm_ffhq'


def controller(model, wavelet, count, times, trace, position, mode, saved):
    original_solve = solver.solve
    base = solver.controller(model, wavelet, 'RESPONSE_RAY', count, times, trace)
    if position is None:
        return base

    def apply(op, origin, q, y, sigma, old, t, s, predict):
        if t != position:
            return base(op, origin, q, y, sigma, old, t, s, predict)

        def refine(provider, transform, direction, gamma, arm, counters):
            anchor, detail = original_solve(provider, transform, direction, gamma, arm, counters)
            previous = {key: counters.get(key, 0) for key in
                        ('forwards', 'input_vjp_attempts', 'input_jvp_attempts')}
            delta, extra = solver.minimize(provider, transform, direction, anchor, mode, 1, counters)
            extra_cost = {key: counters.get(key, 0)-value for key, value in previous.items()}
            saved.update(t=t, next_t=s, origin=origin.detach(), anchor=anchor.detach(),
                         q=q.detach(), update=delta.detach(), extra=extra, extra_cost=extra_cost)
            return delta, detail

        with patch.object(solver, 'solve', refine):
            return base(op, origin, q, y, sigma, old, t, s, predict)

    return apply


def sample(plan, model, wavelet, case, y, kernel, position, mode):
    import torch
    from utils import pigdm_noisy_r1_runtime as runtime
    from utils.diag_pigdm_noisy_r1 import fingerprint

    saved = {}
    trace = []
    with clock.counters(model.model) as counts:
        native.initialize_counts(counts)

        def operation():
            operator = transfer.make_operator(plan, case, model.device, kernel, counts)
            apply = controller(model, wavelet, counts, plan['active_times'], trace, position, mode, saved)
            run = solver.instrument(runtime, 'pigdm', 'RESPONSE_RAY', plan['active_times'], apply)
            return run('A128', model, operator, y, case, 'SHRINK', counts)

        (output, detail), seconds = clock.clocks(torch.cuda.synchronize, operation)
        pairing = transfer.history.fingerprint(detail, shared.rng(), counts, 'pigdm')
    pairing.pop('calls')
    files.require([step['t'] for step in trace] == sorted(plan['active_times'], reverse=True),
                  'Native intervention grid changed')
    files.require(counts['forwards'] == counts['forward_attempts'] ==
                  plan['nfe'] + counts['response_network_forwards'], 'Forward accounting changed')
    files.require(counts['input_vjp_attempts'] == plan['nfe'] + counts['response_vjp_attempts'] and
                  counts['input_jvp_attempts'] == counts.get('response_jvp_forwards', 0) and
                  counts['optimizer_backward_attempts'] == 0, 'Derivative accounting changed')
    row = dict(position=position, mode=mode, pairing=pairing, seconds=seconds, counts=counts)
    if position is not None:
        files.require(saved.get('t') == position and saved['next_t'] >= 0, 'Single intervention missing')
        cost = saved['extra_cost']
        files.require(cost['input_vjp_attempts'] == 1 and cost['input_jvp_attempts'] <= 1 and cost['forwards'] <= 5,
                      'Extra update exceeded the common budget')
        row.update(extra=saved['extra'], extra_cost=cost,
                   anchor_identity={key:fingerprint(saved[key]) for key in ('origin', 'q', 'anchor')})
    return output, row, saved


def local_metrics(plan, case, model, kernel, saved, truth):
    import torch
    from utils import pigdm_noisy_r1_runtime as runtime

    diffusion = runtime.original_diffusion()
    op = transfer.make_operator(plan, case, model.device, kernel)
    gt_observation = op.H(truth)
    rates = {}
    with clock.counters(model.model) as counts, torch.no_grad():
        for name in ('anchor', 'update'):
            value = solver.response.pigdm_predict(model, diffusion, saved['origin']+saved[name], saved['next_t'])
            rates[name] = dict(image_mse=audit.mse(value-truth),
                               clean_observation_mse=audit.mse(op.H(value)-gt_observation))
    files.require(counts['forwards'] == counts['forward_attempts'] == 2, 'Local evaluation cost differs')
    return rates, counts['forwards']


def execute(parent, root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.diag_noise_feasible_zaps import metrics
    from utils.historical_ffhq_runtime import precision

    plan = files.read(parent/'plan.json')
    files.require(plan['experiment'] == 'A126' and plan['host'] == 'pigdm' and
                  files.read(parent/'summary.json')['status'] == 'complete', 'Completed A126 PiGDM required')
    cases = [case for case in plan['cases'] if case['image_id'] == audit.IMAGE_ID]
    files.require(len(cases) == 2 and {case['task'] for case in cases} == set(transfer.TASKS) and
                  set(TARGETS) <= set(plan['active_times']), 'Fixed tasks or time indices missing')
    if (root/'summary.json').exists():
        report = files.read(root/'summary.json')
        files.require(report['status'] == 'complete', f'Existing partial audit retained at {root}')
        print('A128 SUMMARY', report['comparisons'], flush=True)
        return
    admission.check(torch, root)
    checkpoint = plan['checkpoint']
    original_plan = files.read(Path(plan['source'])/'plan.json')
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'] and
                  files.digest(plan['motion_kernel']) == original_plan['motion_kernel_sha256'],
                  'Archived model or motion kernel changed')
    kernel = np.load(plan['motion_kernel'], allow_pickle=False)
    root.mkdir(parents=True)
    report = dict(status='running', parent_plan_sha256=files.digest(parent/'plan.json'),
                  image=audit.IMAGE_ID, targets=TARGETS, cases=[], comparisons=[])
    path = root/'summary.json'
    files.write(path, report)
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)
        model.model.eval()
        wavelet = OrthogonalDWT2D(wave='db4', level=3).to(model.device)
        for case in cases:
            folder = parent/'cases'/case['key']/'RESPONSE_RAY'
            reference = files.read(folder/'run.json')
            files.require(reference['status'] == 'complete' and reference['case'] == case and
                          files.digest(case['image']) == case['image_sha256'] and
                          files.digest(case['y_path']) == case['y_sha256'], 'Archived case identity differs')
            archived = audit.checked_tensor(audit.qualified_artifact(folder, 'output.pt', reference), 'cpu')
            y = audit.checked_tensor(case['y_path'], model.device)
            truth = load_image_as_tensor(case['image']).to(model.device)
            data = dict(case=case['key'], task=case['task'], runs=[])
            report['cases'].append(data)
            for position, mode in [(None, 'baseline')] + [(t, m) for t in TARGETS for m in MODES]:
                with precision('LEGACY_IEEE'), frozen_network(model.model), input_vjp_parameters(model.model):
                    output, row, saved = sample(plan, model, wavelet, case, y, kernel, position, mode)
                    files.require(row['pairing'] == reference['pairing'], 'Native RNG or sampling grid differs')
                    if position is not None:
                        row['local'], row['evaluation_nfe'] = local_metrics(plan, case, model, kernel, saved, truth)
                if position is None:
                    error = audit.norm(output-archived)/max(audit.norm(archived), 1e-30)
                    files.require(error <= 1e-4, f'A126 true-response scalar replay differs: {error:.3g}')
                    row['replay_relative_l2'] = error
                row['quality'] = metrics(output, truth.cpu())
                row['terminal_mse'] = audit.mse(output-truth.cpu())
                data['runs'].append(row)
                rows = [run for case_data in report['cases'] for run in case_data['runs']]
                report['actual_nfe'] = sum(run['counts']['forwards'] + run.get('evaluation_nfe', 0) for run in rows)
                files.require(report['actual_nfe'] <= 2796, 'Diagnostic NFE cap exceeded')
                files.write(path, report)
                print('A128 COMPLETE', case['key'], position, mode, row['quality'], flush=True)
            comparisons = compare(data)
            report['comparisons'].extend(comparisons)
            for result in comparisons:
                print('A128 PAIRED', result, flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        files.write(path, report)


def compare(data):
    base = data['runs'][0]
    results = []
    for position in TARGETS:
        ray, diag = [row for row in data['runs'] if row['position'] == position]
        files.require(ray['mode'] == 'ray' and diag['mode'] == 'diagonal' and
                      ray['anchor_identity'] == diag['anchor_identity'],
                      'The two updates started at different states')
        local = {metric: diag['local']['update'][metric]-ray['local']['update'][metric]
                 for metric in ('image_mse', 'clean_observation_mse')}
        ray_loss = ray['extra']['iterations'][-1]['after']
        diag_loss = diag['extra']['iterations'][-1]['after']
        results.append(dict(task=data['task'], t=position, diagonal_minus_extra_ray_local=local,
            diagonal_minus_extra_ray_objective=diag_loss-ray_loss,
            diagonal_minus_extra_ray_terminal={k:diag['quality'][k]-ray['quality'][k] for k in transfer.METRICS},
            diagonal_minus_extra_ray_terminal_mse=diag['terminal_mse']-ray['terminal_mse'],
            extra_ray_minus_baseline_psnr=ray['quality']['psnr']-base['quality']['psnr'],
            diagonal_minus_baseline_psnr=diag['quality']['psnr']-base['quality']['psnr'],
            extra_ray_cost=ray['extra_cost'], diagonal_cost=diag['extra_cost']))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    files.require(args.parent.resolve() != args.output.resolve() and
                  not args.output.resolve().is_relative_to(args.parent.resolve()), 'Use a separate output directory')
    started = time.perf_counter()
    execute(args.parent, args.output)
    print('A128 REPORT', args.output/'summary.json', f'{time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
