"""Read A118/A126 traces or compare corrections at fixed scalar states."""
import argparse
import inspect
import math
from pathlib import Path
import statistics
from unittest.mock import patch

from utils import diag_a118_blur_transfer as transfer
from utils import diag_a126_full_response as a126
from utils import full_coefficient_response as solver
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import shared_prefix_feedback_runtime as zaps_runtime

SCALAR = 'CLOSED_SCALAR_SUFFIX'
DIAG = 'CLOSED_DIAG_SUFFIX'
IMAGE_ID = '00108'
TIMES = (900, 500, 100)
PROBE_NFE_LIMIT = 980


def qualified_artifact(folder, filename, metadata):
    path = folder / filename
    files.require(files.digest(path) == metadata['artifacts'][filename], f'Artifact changed: {path}')
    return path


def staged_summary(rows):
    gains = [before - after for before, after in rows]
    fractions = [(old - new) / old for old, new in rows if old > 1e-9]
    return dict(positions=len(rows), strict_decrease=sum(gain > 1e-9 for gain in gains),
                objective_start_near_zero=len(rows)-len(fractions),
                median_relative_drop=statistics.median(fractions) if fractions else None)


def trace_audit(zaps_root, pi_root):
    zplan = files.read(zaps_root/'plan.json')
    pplan = files.read(pi_root/'plan.json')
    files.require(files.read(zaps_root/'summary.json')['status'] == 'complete', 'A118 must be complete')
    files.require(files.read(pi_root/'summary.json')['status'] == 'complete', 'A126 PiGDM must be complete')
    files.require(zplan['experiment'] == 'A118' and pplan['experiment'] == 'A126' and
                  pplan['host'] == 'pigdm', 'Wrong analysis roots')
    reports = []
    for task in transfer.TASKS:
        zaps_rows, pi_rows = [], []
        pi_stops = []
        accepted, rejected = 0, 0
        for case in pplan['cases']:
            if case['task'] != task:
                continue
            key = case['key']
            zfolder = zaps_root/'cases'/key
            zrun = files.read(zfolder/'run.json')
            ptrace = pi_root/'cases'/key/'RESPONSE_DIAG'
            prun = files.read(ptrace/'run.json')
            files.require(zrun['status'] == prun['status'] == 'complete', f'Incomplete pair: {key}')
            zsteps = files.read(qualified_artifact(zfolder, DIAG+'_trace.json', zrun['arms'][DIAG]))
            psteps = files.read(qualified_artifact(ptrace, 'trace.json', prun))
            files.require([r['t'] for r in zsteps] == sorted(zplan['grid'], reverse=True) and
                          [r['t'] for r in psteps] == sorted(pplan['active_times'], reverse=True),
                          f'Wrong intervention grid: {key}')
            zaps_rows.extend((r['proxy_initial'], r['proxy_final']) for r in zsteps)
            for step in psteps:
                diag = step['diagonal']
                ray_loss = step['ray']['iterations'][-1]['after']
                pi_rows.append((ray_loss, diag['iterations'][-1]['after']))
                pi_stops.append(diag['termination'])
                for update in diag['iterations']:
                    for trial in update.get('trials', []):
                        accepted += trial['accepted'] and trial['step'] > 0
                        rejected += not trial['accepted']
        files.require(len(zaps_rows) == len(pi_rows) == 90, f'Expected three paired images for {task}')
        reports.append(dict(task=task, zaps_proxy=staged_summary(zaps_rows),
                            pigdm_response=staged_summary(pi_rows), accepted_nonzero_updates=accepted,
                            rejected_trials=rejected, stopping={s:pi_stops.count(s) for s in sorted(set(pi_stops))}))
    return reports


def nearest_positions(grid):
    return {min(grid, key=lambda time: (abs(time-target), time)): target for target in TIMES}


def norm(value):
    return float(value.double().norm())


def mse(value):
    return float(value.double().square().mean())


def probe_state(network, op, wavelet, origin, clean, q, y, sigma, alpha, row, predict, truth, count):
    """All proposed corrections use the same state, noise draw, and closed scalar anchor."""
    import torch

    clean, q, origin = clean.detach(), q.detach(), origin.detach()
    with torch.no_grad():
        h_current = op.H(clean)
        residual = y - h_current
        current_target = h_current + row['weight'] * residual
        anchor = row['gamma'] * q
        proxy = a126.proxy.refine(wavelet, 1, 1., count, [])(op, q, residual, alpha, row, 0)

    provider = solver.response.Provider(network, predict, op, origin, y, sigma, count)
    next_target = provider.target
    actual = {}
    for name, target in (('actual_current_target', current_target.reshape(-1)), ('actual_next_target', next_target)):
        provider.target = target.detach()
        delta, detail = solver.minimize(provider, wavelet, q, anchor, 'diagonal', 1, count)
        actual[name] = (delta, detail)

    proposals = {'scalar': (anchor, None), 'proxy_diagonal': (proxy, None), **actual}
    with torch.no_grad():
        truth_observed = op.H(truth)
        scalar_observed = None
        rows = {}
        for name, (delta, detail) in proposals.items():
            estimate = predict(origin + delta)
            observed = op.H(estimate)
            proxy_observed = h_current + op.H(delta / math.sqrt(alpha))
            if scalar_observed is None:
                scalar_observed = observed
            change = observed - scalar_observed
            increment = delta - anchor
            proxy_change = op.H(increment / math.sqrt(alpha))
            change_norm = norm(change)
            denominator = change_norm * norm(proxy_change)
            angular = None if denominator < 1e-24 else float(solver.response.dot(change, proxy_change))/denominator
            increment_norm = norm(increment)
            q_squared = float(solver.response.dot(q, q))
            parallel = q * (float(solver.response.dot(increment, q))/q_squared) if q_squared > 1e-24 else 0
            rows[name] = dict(proxy_loss=mse(proxy_observed-current_target),
                              actual_current_target_loss=mse(observed-current_target),
                              actual_next_target_loss=mse(observed.reshape(-1)-next_target),
                              clean_observation_mse=mse(observed-truth_observed),
                              clean_image_mse=mse(estimate-truth),
                              noisy_observation_mse=mse(observed-y),
                              increment_over_scalar_norm=increment_norm/max(norm(anchor),1e-30),
                              transverse_fraction=norm(increment-parallel)/increment_norm if increment_norm else None,
                              actual_proxy_response_cosine=angular,
                              actual_proxy_response_relative_error=(norm(change-proxy_change)/change_norm
                                                                    if change_norm>1e-12 else None),
                              solver=detail)
        return dict(current_weight=row['weight'], next_weight=provider.omega,
                    target_shift_mse=mse(current_target.reshape(-1)-next_target), candidates=rows)


def insert_probe(function, edits, **bindings):
    source = inspect.getsource(function)
    for before, after in edits:
        files.require(source.count(before) == 1, 'Scalar replay source changed')
        source = source.replace(before, after, 1)
    namespace = {**function.__globals__, **bindings}
    exec(compile(source, inspect.getfile(function), 'exec'), namespace)
    return namespace[function.__name__]


def replay_zaps(root, model, case, y, callback, count):
    import torch

    def at_state(zaps, origin, clean, q, row, t, s):
        def predict(value):
            if s < 0:
                return value
            time = torch.full((value.shape[0],), s, device=value.device, dtype=torch.long)
            eps = model._predict_eps(value, time)
            alpha = model.alphas_cumprod[s]
            return ((value - (1-alpha).sqrt()*eps) / alpha.sqrt().clamp(min=1e-8)).clamp(-1, 1)
        callback(zaps.A, zaps.dwt, origin, clean, q, float(model.alphas_cumprod[t]), row, t, s, predict)

    suffix = insert_probe(zaps_runtime._custom_suffix, [
        ('        x = uncond+delta',
         '        at_state(zaps, uncond, clean, q, row, t, s)\n        x = uncond+delta')],
        at_state=at_state)
    from utils import diag_shared_prefix_feedback as zaps_driver
    op = zaps_driver.operator(root, case, model.device, count)
    prefix = zaps_runtime.train_prefix(model, op, y, case)
    with patch.object(zaps_runtime, '_custom_suffix', suffix):
        return zaps_runtime.run_suffix(model, op, y, case, SCALAR, prefix)[0]


def replay_pigdm(plan, model, case, y, callback, wavelet, kernel, count):
    from utils import pigdm_noisy_r1_runtime as rt

    def at_state(op, origin, clean, q, alpha, row, t, s, diffusion):
        predict = lambda value: solver.response.pigdm_predict(model, diffusion, value, s)
        callback(op, wavelet, origin, clean, q, alpha, row, t, s, predict)

    run = insert_probe(rt.run, [
        ("if arm=='SHRINK':", "if arm=='SHRINK' and t in active_times:"),
        ("   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q",
         "   origin=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()\n"
         "   if t in active_times: at_state(op,origin,m,q,a,row,t,s,diffusion)\n"
         "   x=origin+row['gamma']*q")],
        at_state=at_state, active_times=set(plan['active_times']))
    op = transfer.make_operator(plan, case, model.device, kernel, count)
    return run('transfer_audit', model, op, y, case, 'SHRINK', count)[0]


def checked_tensor(path, device):
    import torch
    value = torch.load(path, map_location=device, weights_only=True)
    return value['y'] if isinstance(value, dict) else value


def run_probe(zroot, proot, output):
    import time
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.historical_ffhq_runtime import precision

    trace_report = trace_audit(zroot, proot)
    zplan, pplan = files.read(zroot/'plan.json'), files.read(proot/'plan.json')
    files.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    files.require(torch.cuda.mem_get_info()[0] >= 6*1024**3, 'Need 6 GiB free on the selected GPU')
    checkpoint = pplan['checkpoint']
    checkpoint_path = Path(checkpoint['path'])
    files.require(files.digest(checkpoint_path) == checkpoint['sha256'] == zplan['checkpoint_sha256'],
                  'Archived checkpoint identity differs')
    files.require(files.digest(zroot/'motion_kernel.npy') == zplan['motion_kernel_sha256'] and
                  files.digest(pplan['motion_kernel']) == zplan['motion_kernel_sha256'], 'Motion kernel differs')
    kernel = np.load(pplan['motion_kernel'], allow_pickle=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    report_path = output/'summary.json'
    report = dict(status='running', trace_audit=trace_report, image=IMAGE_ID, target_times=TIMES,
                  sources={str(p): files.digest(p/'plan.json') for p in (zroot, proot)}, cases=[])
    files.write(report_path, report)
    started = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
        model.model.eval()
        wavelet = OrthogonalDWT2D(wave='db4', level=3).to(model.device)
        selected = [case for case in pplan['cases'] if case['image_id'] == IMAGE_ID]
        files.require({case['task'] for case in selected} == set(transfer.TASKS) and len(selected) == 2,
                      'Expected one matching image per task')
        with precision('LEGACY_IEEE'), frozen_network(model.model):
            for case in selected:
                for host in ('zaps', 'pigdm'):
                    result = replay_case(zroot, proot, zplan, pplan, model, wavelet, kernel, case, host)
                    report['cases'].append(result)
                    report['actual_nfe'] = sum(r['counts']['forwards'] for r in report['cases'])
                    files.require(report['actual_nfe'] <= PROBE_NFE_LIMIT, 'Diagnostic NFE cap exceeded')
                    files.write(report_path, report)
                    print('TRANSFER PROBE', host, result['case'], result['summary'], flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        files.write(report_path, report)
    print('TRANSFER PROBE REPORT', report_path, flush=True)


def replay_case(zroot, proot, zplan, pplan, model, wavelet, kernel, case, host):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils import diag_shared_prefix_feedback as zaps_driver
    from utils import native_ray_transfer_runtime as shared
    from utils import transfer_diagonal_runtime as native
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from contextlib import nullcontext

    key = case['key']
    if host == 'zaps':
        folder = zroot/'cases'/key
        saved = files.read(folder/'run.json')
        case = saved['case']
        obs = files.read(zroot/'observations'/key/'observation.json')
        y_path, y_hash = obs['path'], obs['sha256']
        reference = qualified_artifact(folder, SCALAR+'.pt', saved['arms'][SCALAR])
        positions = nearest_positions(zplan['grid'])
        native_nfe = 300
    else:
        folder = proot/'cases'/key/'SCALAR'
        saved = files.read(folder/'run.json')
        y_path, y_hash = case['y_path'], case['y_sha256']
        reference = qualified_artifact(folder, 'output.pt', saved)
        positions = nearest_positions(pplan['active_times'])
        native_nfe = pplan['nfe']
    files.require(saved['status'] == 'complete' and files.digest(y_path) == y_hash and
                  files.digest(case['image']) == case['image_sha256'], f'Input identity failed: {key}')
    truth = load_image_as_tensor(case['image']).to(model.device)
    y = checked_tensor(y_path, model.device)
    probes = []
    with clock.counters(model.model) as count:
        native.initialize_counts(count)

        def at_position(op, wave, origin, clean, q, alpha, row, t, s, predict):
            if t not in positions:
                return
            before_rng = shared.rng()
            forwards = count['forwards']
            result = probe_state(model.model, op, wave, origin, clean, q, y, case['sigma'], alpha,
                                 row, predict, truth, count)
            result.update(t=t, next_t=s, requested_t=positions[t], nfe=count['forwards']-forwards)
            files.require(zaps_driver.rng_equal(before_rng, shared.rng()), 'Probe consumed native sampling RNG')
            probes.append(result)

        scope = input_vjp_parameters(model.model) if host == 'pigdm' else nullcontext()
        with scope:
            if host == 'zaps':
                out = replay_zaps(zroot, model, case, y, at_position, count)
            else:
                out = replay_pigdm(pplan, model, case, y, at_position, wavelet, kernel, count)
        final = shared.rng()
    archived = checked_tensor(reference, 'cpu')
    replay_error = float((out.double()-archived.double()).norm()/archived.double().norm().clamp_min(1e-30))
    files.require(replay_error <= 1e-4, f'{host}/{key} scalar replay differs: {replay_error:.3g}')
    files.require(len(probes) == 3 and count['forwards'] == count['forward_attempts'] ==
                  native_nfe + sum(p['nfe'] for p in probes) and all(p['nfe'] <= 15 for p in probes),
                  'Replay coverage or cost differs')
    if host == 'pigdm':
        from utils.diag_pigdm_noisy_r1 import fingerprint
        files.require(saved['pairing']['final_cpu'] == fingerprint(final['cpu']) and
                      saved['pairing']['final_cuda'] == [fingerprint(state) for state in final['cuda']],
                      'PiGDM replay RNG differs')
    return dict(host=host, case=key, replay_relative_l2=replay_error, probes=probes, counts=count,
                summary=brief(probes))


def brief(probes):
    results = {}
    metrics = ('proxy_loss', 'actual_current_target_loss', 'actual_next_target_loss',
               'clean_image_mse', 'clean_observation_mse', 'noisy_observation_mse')
    for name in ('proxy_diagonal', 'actual_current_target', 'actual_next_target'):
        results[name] = []
        for state in probes:
            scalar = state['candidates']['scalar']
            candidate = state['candidates'][name]
            results[name].append(dict(t=state['t'],
                delta={metric: round(candidate[metric]-scalar[metric], 9) for metric in metrics},
                response_cos=round(candidate['actual_proxy_response_cosine'], 4)
                             if candidate['actual_proxy_response_cosine'] is not None else None,
                current_weight=round(state['current_weight'], 4), next_weight=round(state['next_weight'], 4)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('trace', 'probe'))
    parser.add_argument('--zaps-root', type=Path, default=transfer.PARENT)
    parser.add_argument('--pigdm-root', type=Path, default=a126.ROOT/'A126_pigdm_ffhq')
    parser.add_argument('--output', type=Path,
                        default=a126.ROOT/'audit_a118_a126_fixed_states')
    args = parser.parse_args()
    if args.mode == 'trace':
        for row in trace_audit(args.zaps_root, args.pigdm_root):
            print('TRANSFER TRACE', row)
    else:
        run_probe(args.zaps_root, args.pigdm_root, args.output)


if __name__ == '__main__':
    main()
