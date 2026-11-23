"""A129: paired propagation of one diagonal correction through native sampling."""
import argparse
import math
from pathlib import Path
from unittest.mock import patch

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_pigdm_single_intervention as a128
from utils import diag_transfer_objective_audit as audit
from utils import full_coefficient_response as solver
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import shared_prefix_feedback_runtime as zaps_runtime
from utils import transfer_diagonal_runtime as native

FOLLOW_UP = 5
ROOT = a128.ROOT.with_name('audit_a129_diagonal_propagation')


def window(grid, position):
    index = grid.index(position)
    points = grid[index:index + FOLLOW_UP + 1]
    files.require(len(points) >= 2, f'No native step follows {position}')
    return points


def capture(positions, states):
    def record(t, s, clean, origin, guidance):
        if t in positions:
            states[t] = dict(next_t=s, **{name: value.detach().cpu().clone() for name, value in
                             [('clean', clean), ('origin', origin), ('guidance', guidance)]})
    return record


def instrument_pigdm(runtime, times, apply, record):
    return audit.insert_probe(runtime.run, [
        ("if arm=='SHRINK':", "if arm=='SHRINK' and t in active_times:"),
        ("   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q",
         "   origin=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()\n"
         "   delta=apply(op,origin,q,y,case['sigma'],row,t,s,lambda xx:predict(model,diffusion,xx,s)) if t in active_times else None\n"
         "   guidance=row['gamma']*q if delta is None else delta\n"
         "   x=origin+guidance\n"
         "   record(t,s,m,origin,guidance)")],
        active_times=set(times), apply=apply, predict=solver.response.pigdm_predict, record=record)


def instrument_zaps(position, record):
    return audit.insert_probe(zaps_runtime._custom_suffix, [
        ('        x = uncond+delta',
         '        if t == intervention:\n'
         '            delta, detail, module = repaired.exact_diagonal_with_native_straight_through(\n'
         '                zaps, zaps.A, clean, q, residual, ab[t], rules.SIGMA, module)\n'
         '        x = uncond+delta\n'
         '        record(t,s,clean,uncond,delta)')],
        intervention=position, record=record)


def log_snr(alpha):
    return math.log(alpha) - math.log1p(-alpha)


def match_times(pi_log_snr, zaps_log_snr):
    return {t:min(zaps_log_snr, key=lambda candidate: (abs(zaps_log_snr[candidate]-pi_log_snr[t]), candidate))
            for t in a128.TARGETS}


def scalar_replay_check(out, reference):
    error = audit.norm(out-reference)/max(audit.norm(reference), 1e-30)
    files.require(error <= 1e-4, f'Archived ZAPS scalar replay relative L2 differs: {error:.3g}')
    return error


def pigdm_run(plan, model, wavelet, case, y, kernel, position, mode, grid, reference):
    import torch
    from utils import pigdm_noisy_r1_runtime as runtime
    from utils.diag_pigdm_noisy_r1 import fingerprint

    trace, saved, states = [], {}, {}
    with clock.counters(model.model) as count:
        native.initialize_counts(count)
        op = transfer.make_operator(plan, case, model.device, kernel, count)
        apply = a128.controller(model, wavelet, count, plan['active_times'], trace, position, mode, saved)
        run = instrument_pigdm(runtime, plan['active_times'], apply, capture(set(window(grid, position)), states))
        (out, detail), seconds = clock.clocks(torch.cuda.synchronize,
            lambda:run('A129', model, op, y, case, 'SHRINK', count))
        pairing = transfer.history.fingerprint(detail, shared.rng(), count, 'pigdm')
    pairing.pop('calls')
    files.require(pairing == reference['pairing'] and list(states) == window(grid, position) and
                  [row['t'] for row in trace] == sorted(plan['active_times'], reverse=True),
                  'PiGDM native trajectory or RNG check failed')
    files.require(count['forwards'] == count['forward_attempts'] == plan['nfe']+count['response_network_forwards'] and
                  count['input_vjp_attempts'] == plan['nfe']+count['response_vjp_attempts'] and
                  count['input_jvp_attempts'] == count['response_jvp_forwards'] and
                  count['optimizer_backward_attempts'] == 0 and saved.get('t') == position,
                  'PiGDM call or intervention accounting changed')
    return dict(output=out, states=states, counts=count, seconds=seconds, extra_cost=saved['extra_cost'],
                anchor={key:fingerprint(saved[key]) for key in ('origin', 'q', 'anchor')})


def zaps_run(model, kernel, case, y, prefix, position, observed_positions, expected_grid):
    import torch

    states = {}
    suffix = instrument_zaps(position, capture(observed_positions, states))
    with clock.counters(model.model) as count, patch.object(zaps_runtime, '_custom_suffix', suffix):
        op = zaps_runtime.make_operator(case, model.device, count=count, kernel=kernel)
        (out, detail), seconds = clock.clocks(torch.cuda.synchronize,
            lambda:zaps_runtime.run_suffix(model, op, y, case, audit.SCALAR, prefix))
    files.require(count['forwards'] == count['forward_attempts'] == 30 and
                  count['optimizer_backward_completed'] == count['optimizer_backward_attempts'] == 1 and
                  count['calls'] == expected_grid and set(states) == observed_positions,
                  'ZAPS suffix budget or sampling grid changed')
    return dict(output=out, states=states, counts=count, seconds=seconds, rng=detail['final_rng'])


def error(clean, op, truth, observed_truth):
    return dict(image_mse=audit.mse(clean-truth), clean_observation_mse=audit.mse(op.H(clean)-observed_truth))


def difference(diagonal, scalar):
    return {name: diagonal[name]-scalar[name] for name in scalar}


def transition_geometry(diagonal, scalar):
    import torch

    unconditioned = diagonal['origin'].double()-scalar['origin'].double()
    guidance = diagonal['guidance'].double()-scalar['guidance'].double()
    after = unconditioned+guidance
    before_norm = audit.norm(unconditioned)
    return dict(unconditioned_rms=audit.mse(unconditioned)**.5, guidance_rms=audit.mse(guidance)**.5,
                after_guidance_rms=audit.mse(after)**.5,
                after_over_unconditioned=audit.norm(after)/before_norm if before_norm > 1e-14 else None,
                signed_guidance_projection=float(torch.sum(unconditioned*guidance))/before_norm**2
                if before_norm > 1e-14 else None)


def first_decay(rows):
    """Predeclare a 50% loss of the initial local benefit for the offline probe."""
    initial = rows[0]['error_delta']['image_mse']
    if initial >= 0:
        return None
    return next((index for index in range(1, len(rows))
                 if rows[index]['error_delta']['image_mse'] >= initial/2), None)


def diagnose(model, op, truth, predict, grid, position, snr, ray, diagonal):
    import torch
    from utils.diag_noise_feasible_zaps import metrics

    points = window(grid, position)
    scalar, diag = ray['states'], diagonal['states']
    files.require(torch.equal(scalar[position]['origin'], diag[position]['origin']) and
                  torch.equal(scalar[position]['clean'], diag[position]['clean']),
                  'Branches diverged before the intervention')
    if 'anchor' in ray:
        files.require(ray['anchor'] == diagonal['anchor'], 'A128 ray anchor changed between branches')
    truth = truth.to(model.device)
    observed_truth = op.H(truth)
    rows = []
    with torch.no_grad():
        for previous, current in zip(points, points[1:]):
            pair = [error(states[current]['clean'].to(model.device), op, truth, observed_truth)
                    for states in (scalar, diag)]
            rows.append(dict(t=current, preceding_t=previous, log_snr=snr[current],
                cumulative_log_snr=snr[current]-snr[position], error_delta=difference(pair[1], pair[0]),
                preceding_update=transition_geometry(diag[previous], scalar[previous])))
        index = first_decay(rows)
        shadow = None
        if index is not None:
            row = rows[index]
            previous, current = row['preceding_t'], row['t']
            files.require(scalar[previous]['next_t'] == diag[previous]['next_t'] == current,
                          'Shadow comparison needs an adjacent transition')
            common = scalar[previous]['guidance'].to(model.device)
            with clock.counters(model.model) as count:
                predictions = [predict((states[previous]['origin'].to(model.device)+common), current)
                               for states in (scalar, diag)]
            files.require(count['forwards'] == count['forward_attempts'] == 2, 'Shadow cost differs')
            reference = error(scalar[current]['clean'].to(model.device), op, truth, observed_truth)
            replay_error = audit.norm(predictions[0]-scalar[current]['clean'].to(model.device))/max(
                audit.norm(scalar[current]['clean']), 1e-30)
            files.require(replay_error <= 1e-4, f'Shadow scalar predictor differs: {replay_error:.3g}')
            shadow = dict(preceding_t=previous, measured_at=current, scalar_replay_relative_l2=replay_error,
                          actual_error_delta=row['error_delta'],
                          common_guidance_error_delta=difference(error(predictions[1], op, truth, observed_truth), reference),
                          nfe=count['forwards'])
    target = truth.cpu()
    ray_quality, diag_quality = (metrics(run['output'], target) for run in (ray, diagonal))
    return dict(t=position, log_snr=snr[position], steps=rows, common_guidance_shadow=shadow,
                terminal=dict(ray=ray_quality, diagonal=diag_quality, delta=difference(diag_quality, ray_quality)))


def verify_a128(out, truth, row):
    mse = audit.mse(out-truth.cpu())
    files.require(math.isclose(mse, row['terminal_mse'], rel_tol=1e-4, abs_tol=1e-9),
                  'A128 terminal raw MSE replay differs')


def zaps_predict(model, value, t):
    import torch

    alpha = model.alphas_cumprod[t]
    index = torch.full((value.shape[0],), t, device=value.device, dtype=torch.long)
    epsilon = model._predict_eps(value, index)
    return ((value-(1-alpha).sqrt()*epsilon)/alpha.sqrt().clamp(min=1e-8)).clamp(-1, 1)


def report_pair(report, path, host, case, pair, ray, diagonal):
    pair.update(host=host, case=case['key'], task=case['task'],
                ray_counts=ray['counts'], diagonal_counts=diagonal['counts'],
                ray_seconds=ray['seconds'], diagonal_seconds=diagonal['seconds'])
    if 'extra_cost' in ray:
        pair.update(extra_ray_cost=ray['extra_cost'], extra_diagonal_cost=diagonal['extra_cost'])
    report['pairs'].append(pair)
    shadow = pair['common_guidance_shadow']
    report['actual_nfe'] += (ray['counts']['forwards'] if host == 'pigdm' else 0) + diagonal['counts']['forwards']
    report['actual_nfe'] += shadow['nfe'] if shadow else 0
    files.require(report['actual_nfe'] <= 2968, 'A129 diagnostic NFE cap exceeded')
    files.write(path, report)
    print('A129 PAIRED', host, case['task'], pair['t'], dict(
        image_mse_deltas=[(step['t'], step['error_delta']['image_mse']) for step in pair['steps']],
        state_gap_ratios=[(step['preceding_t'], step['preceding_update']['after_over_unconditioned'])
                          for step in pair['steps']],
        common_guidance_shadow=shadow, terminal_delta=pair['terminal']['delta']), flush=True)


def execute(zroot, proot, a128_root, root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.diag_shared_prefix_feedback import rng_equal
    from utils.historical_ffhq_runtime import precision
    from utils import pigdm_noisy_r1_runtime as runtime

    zplan, pplan = files.read(zroot/'plan.json'), files.read(proot/'plan.json')
    history = files.read(a128_root/'summary.json')
    files.require(zplan['experiment'] == 'A118' and pplan['experiment'] == 'A126' and
                  pplan['host'] == 'pigdm' and pplan['nfe'] == 100 and
                  all(files.read(p/'summary.json')['status'] == 'complete' for p in (zroot, proot)) and
                  history['status'] == 'complete' and history['parent_plan_sha256'] == files.digest(proot/'plan.json'),
                  'Completed and compatible A118, A126 and A128 results required')
    cases = [case for case in pplan['cases'] if case['image_id'] == audit.IMAGE_ID]
    files.require(len(cases) == 2 and {case['task'] for case in cases} == set(transfer.TASKS) and
                  set(a128.TARGETS) <= set(pplan['active_times']), 'Frozen cases or intervention positions differ')
    checkpoint = pplan['checkpoint']
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'] == zplan['checkpoint_sha256'],
                  'FFHQ checkpoint identity differs')
    files.require(files.digest(pplan['motion_kernel']) == files.digest(zroot/'motion_kernel.npy') ==
                  zplan['motion_kernel_sha256'], 'Fixed motion kernel differs')
    if (root/'summary.json').exists():
        report = files.read(root/'summary.json')
        files.require(report['status'] == 'complete', f'Partial run retained at {root}')
        print('A129 REPORT', root/'summary.json', flush=True)
        return
    admission.check(torch, root)
    root.mkdir(parents=True)
    path = root/'summary.json'
    report = dict(status='running', image=audit.IMAGE_ID, pairs=[], zaps_prefixes=[], zaps_scalar_replays=[],
                  actual_nfe=0, sources={str(p):files.digest(p) for p in
                      (zroot/'plan.json', proot/'plan.json', a128_root/'summary.json', Path(__file__))})
    files.write(path, report)
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4', level=3).to(model.device)
        kernel = np.load(pplan['motion_kernel'], allow_pickle=False)
        diffusion = runtime.original_diffusion()
        pi_grid = list(range(990, -1, -10))
        zaps_grid = sorted(zplan['grid'], reverse=True)
        grid_tensor = torch.tensor(pi_grid, device=model.device, dtype=torch.long)
        pi_snr = {t:log_snr(alpha) for t, alpha in zip(pi_grid, diffusion.alpha(grid_tensor).reshape(-1).tolist())}
        zaps_snr = {t:log_snr(float(model.alphas_cumprod[t])) for t in zaps_grid}
        matched = match_times(pi_snr, zaps_snr)
        report['matching'] = [dict(pigdm_t=t, pigdm_log_snr=pi_snr[t],
                                   zaps_t=s, zaps_log_snr=zaps_snr[s]) for t, s in matched.items()]
        files.write(path, report)
        for case in cases:
            key = case['key']
            past = next(entry for entry in history['cases'] if entry['case'] == key)
            pi_ref = files.read(proot/'cases'/key/'RESPONSE_RAY'/'run.json')
            zfolder = zroot/'cases'/key
            zref = files.read(zfolder/'run.json')
            zcase = zref['case']
            observation = files.read(zroot/'observations'/key/'observation.json')
            files.require(pi_ref['status'] == zref['status'] == 'complete' and pi_ref['case'] == case and
                          files.digest(case['image']) == case['image_sha256'] == zcase['image_sha256'] and
                          files.digest(case['y_path']) == case['y_sha256'] and
                          files.digest(observation['path']) == observation['sha256'], 'A129 case identity differs')
            truth = load_image_as_tensor(case['image'])
            pi_y = audit.checked_tensor(case['y_path'], model.device)
            zaps_y = audit.checked_tensor(observation['path'], model.device)
            pi_op = transfer.make_operator(pplan, case, model.device, kernel)
            zaps_op = zaps_runtime.make_operator(zcase, model.device, kernel=kernel)
            with precision('LEGACY_IEEE'), frozen_network(model.model):
                with input_vjp_parameters(model.model):
                    for t in a128.TARGETS:
                        runs = []
                        for mode in a128.MODES:
                            result = pigdm_run(pplan, model, wave, case, pi_y, kernel, t, mode, pi_grid, pi_ref)
                            old = next(row for row in past['runs'] if row['position'] == t and row['mode'] == mode)
                            verify_a128(result['output'], truth, old)
                            runs.append(result)
                        pair = diagnose(model, pi_op, truth,
                                        lambda value, time:solver.response.pigdm_predict(model, diffusion, value, time),
                                        pi_grid, t, pi_snr, *runs)
                        report_pair(report, path, 'pigdm', case, pair, *runs)
                with clock.counters(model.model) as count:
                    prefix_op = zaps_runtime.make_operator(zcase, model.device, count=count, kernel=kernel)
                    prefix = zaps_runtime.train_prefix(model, prefix_op, zaps_y, zcase)
                files.require(count['forwards'] == count['forward_attempts'] == 270 and
                              count['optimizer_backward_completed'] == count['optimizer_backward_attempts'] == 9 and
                              count['calls'] == zaps_grid*9, 'A118 prefix budget or grid changed')
                report['zaps_prefixes'].append(dict(case=key, counts=count))
                report['actual_nfe'] += count['forwards']
                positions = set().union(*(window(zaps_grid, s) for s in matched.values()))
                ray = zaps_run(model, kernel, zcase, zaps_y, prefix, None, positions, zaps_grid)
                archived = audit.checked_tensor(audit.qualified_artifact(zfolder, audit.SCALAR+'.pt',
                                                zref['arms'][audit.SCALAR]), 'cpu')
                relative_error = scalar_replay_check(ray['output'], archived)
                report['zaps_scalar_replays'].append(dict(case=key, counts=ray['counts'], relative_l2=relative_error))
                report['actual_nfe'] += ray['counts']['forwards']
                for s in matched.values():
                    diagonal = zaps_run(model, kernel, zcase, zaps_y, prefix, s, set(window(zaps_grid, s)), zaps_grid)
                    files.require(rng_equal(ray['rng'], diagonal['rng']), 'A118 final RNG changed')
                    pair = diagnose(model, zaps_op, truth, lambda value, time:zaps_predict(model, value, time),
                                    zaps_grid, s, zaps_snr, ray, diagonal)
                    report_pair(report, path, 'zaps', zcase, pair, ray, diagonal)
        files.require(len(report['pairs']) == 8, 'Incomplete A129 pair panel')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), actual_nfe_is_lower_bound=True)
        raise
    finally:
        files.write(path, report)
    print('A129 REPORT', path, 'actual_nfe=', report['actual_nfe'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zaps-root', type=Path, default=transfer.PARENT)
    parser.add_argument('--pigdm-root', type=Path, default=a128.PARENT)
    parser.add_argument('--a128-root', type=Path, default=a128.ROOT)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    parents = [args.zaps_root.resolve(), args.pigdm_root.resolve(), args.a128_root.resolve()]
    files.require(all(output != p and not output.is_relative_to(p) for p in parents),
                  'Use a separate A129 output directory')
    execute(*parents, output)


if __name__ == '__main__':
    main()
