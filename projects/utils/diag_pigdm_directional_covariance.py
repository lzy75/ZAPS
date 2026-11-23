"""A130: put a frozen directional denoiser-error covariance into native PiGDM."""
import argparse
import math
from pathlib import Path
import statistics
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_pigdm_single_intervention as a128
from utils import diag_transfer_objective_audit as audit
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = a128.ROOT.with_name('diag_a130_pigdm_directional_covariance')
CARD = transfer.PROJECTS/'experiment_cards/A130_pigdm_directional_covariance.md'
CALIBRATION_IDS = ('00001', '00004', '00008')
ANCHORS = (0, 90, 300, 600, 920, 990)
DRAWS = 2
SIZE = 256
ARMS = ('BASE', 'SCALAR', 'COVARIANCE_SCALAR')
GRID = tuple(range(990, -1, -10))


def frequency_groups(size, device):
    import torch

    x = torch.fft.rfftfreq(size, device=device, dtype=torch.float64)[None, :]
    y = torch.fft.fftfreq(size, device=device, dtype=torch.float64)[:, None]
    radius = (x*x+y*y).sqrt()
    radial = torch.bucketize(radius, radius.new_tensor([1/16, 1/8, 1/4]))
    angle = torch.remainder(torch.atan2(y, x)+math.pi/8, math.pi)
    direction = (angle/(math.pi/4)).long().clamp(max=3)
    # The two self-conjugate columns use an axis bin on both sides.
    direction[:, 0] = 0
    direction[:, -1] = 0
    bands = 4*radial + direction
    multiplicity = radius.new_full(radius.shape, 2)
    multiplicity[:, (0, -1)] = 1
    masks = torch.stack([(bands == index)*multiplicity for index in range(16)])
    return bands, masks


def interpolate(values, alphas, grid):
    import torch

    def log_snr(t):
        alpha = alphas[t]
        return math.log(alpha)-math.log1p(-alpha)

    axis = {t:log_snr(t) for t in set(grid).union(ANCHORS)}
    table = {}
    for t in grid:
        if t in values:
            table[t] = values[t]
            continue
        lower, upper = next((l, u) for l, u in zip(ANCHORS, ANCHORS[1:]) if l < t < u)
        weight = (axis[t]-axis[lower])/(axis[upper]-axis[lower])
        table[t] = torch.lerp(values[lower], values[upper], weight)
    return table


def calibrate(model, calibration, alphas):
    import torch
    from modules.main_single import load_image_as_tensor

    bands, masks = frequency_groups(SIZE, model.device)
    mass = masks.sum(dim=(-2, -1))
    files.require(bool((mass > 0).all()) and float(mass.sum()) == SIZE**2,
                  'Frequency groups must cover the full Hermitian spectrum')
    expected = len(calibration)*DRAWS*len(ANCHORS)
    totals = {t:torch.zeros_like(mass) for t in ANCHORS}
    started = time.perf_counter()
    before = shared.rng()
    from utils.diag_shared_prefix_feedback import rng_equal
    with clock.counters(model.model) as count, torch.no_grad():
        for row in calibration:
            clean = load_image_as_tensor(row['path']).to(model.device)
            for draw in range(DRAWS):
                gen = torch.Generator(device=model.device).manual_seed(731300 + 10*int(row['id']) + draw)
                epsilon = torch.randn(clean.shape, generator=gen, dtype=clean.dtype, device=clean.device)
                for t in ANCHORS:
                    alpha = alphas[t]
                    state = math.sqrt(alpha)*clean + math.sqrt(1-alpha)*epsilon
                    time_index = torch.tensor([t], dtype=torch.long, device=model.device)
                    predicted = model.model(state, time_index)[:, :clean.shape[1]]
                    estimate = (state-math.sqrt(1-alpha)*predicted)/math.sqrt(alpha)
                    power = torch.fft.rfft2(estimate-clean, norm='ortho').abs().double().square().mean((0, 1))
                    totals[t] += (masks*power).sum(dim=(-2, -1))
    files.require(rng_equal(before, shared.rng()), 'Calibration advanced sampling RNG')
    files.require(count['forwards'] == count['forward_attempts'] == expected and
                  count['input_vjp_attempts'] == count['optimizer_backward_attempts'] == 0,
                  'Calibration neural cost differs')
    variances = {t:totals[t]/(mass*len(calibration)*DRAWS) for t in ANCHORS}
    values = {t:value/((value*mass).sum()/mass.sum()) for t,value in variances.items()}
    table = interpolate(values, alphas, GRID)
    weights = {}
    for t, row in table.items():
        covariance = row[bands]
        files.require(bool(torch.isfinite(row).all()) and bool((row > 0).all()) and
                      abs(float((row*mass).sum()/mass.sum())-1) < 1e-12 and
                      torch.equal(covariance[:, 0], covariance[(-torch.arange(SIZE, device=model.device)) % SIZE, 0]) and
                      torch.equal(covariance[:, -1], covariance[(-torch.arange(SIZE, device=model.device)) % SIZE, -1]),
                      f'Invalid normalized covariance at t={t}')
        weights[t] = covariance.float()
    torch.cuda.synchronize()
    return weights, dict(nfe=count['forwards'], seconds=time.perf_counter()-started, ordered_times=count['calls'],
                         band_sizes=mass.cpu().tolist(),
                         error_variance={t:v.cpu().tolist() for t,v in variances.items()},
                         normalized_variance={t:v.cpu().tolist() for t,v in table.items()})


def solve_weighted(op, residual, lam, covariance):
    import torch

    files.require(math.isfinite(lam) and lam > 0 and op.spectrum.shape == covariance.shape,
                  'Invalid covariance regularized solve')
    op.bump('regularized_solves')
    op.bump('adjoint_actions')
    op.bump('fft_actions')
    op.bump('ifft_actions')
    return torch.fft.irfft2(op.spectrum.conj()*torch.fft.rfft2(residual) /
                            (op.spectrum.abs().square()*covariance+lam), s=(op.size, op.size))


def instrument(runtime, times, covariance):
    def solve_at(op, residual, lam, t):
        return op.solve(residual, lam) if covariance is None else solve_weighted(op, residual, lam, covariance[t])

    return audit.insert_probe(runtime.run, [
        ("if arm=='SHRINK':", "if arm=='SHRINK' and t in active_times:"),
        ("v=op.solve(r,case['sigma']**2/(1-a))", "v=solve_at(op,r,case['sigma']**2/(1-a),t)")],
        active_times=set(times), solve_at=solve_at)


def profile_inputs(plan, source_plan, calibration_dir):
    panel = source_plan['cases']
    images = {case['image_id']:case['image_sha256'] for case in panel}
    files.require(len(images) == 10, 'Expected frozen A118 ten-image panel')
    folder = calibration_dir if calibration_dir is not None else Path(plan['cases'][0]['image']).parent
    rows = []
    for image_id in CALIBRATION_IDS:
        path = (folder / (image_id+'.png')).resolve()
        file_hash = files.digest(path)
        files.require(image_id not in images and file_hash not in set(images.values()),
                      'Calibration image overlaps the ten evaluation images')
        rows.append(dict(id=image_id, path=str(path), sha256=file_hash))
    files.require(len({row['sha256'] for row in rows}) == len(rows), 'Repeated calibration image')
    return rows


def sampler_run(model, plan, case, kernel, arm, weights, counts, save_progress):
    import torch
    from utils import pigdm_noisy_r1_runtime as runtime

    y = audit.checked_tensor(case['y_path'], model.device)
    op = transfer.make_operator(plan, case, model.device, kernel, counts)
    run = instrument(runtime, plan['active_times'], weights if arm == 'COVARIANCE_SCALAR' else None)
    return run('A130', model, op, y, case, 'BASE' if arm=='BASE' else 'SHRINK', counts, progress=save_progress)


def unit_gate(plan, cases, model, kernel):
    import torch

    generator = torch.Generator(device=model.device).manual_seed(731300)
    residual = torch.randn((1, 3, SIZE, SIZE), generator=generator, device=model.device)
    with torch.no_grad():
        for task in transfer.TASKS:
            case = next(row for row in cases if row['task'] == task)
            op = transfer.make_operator(plan, case, model.device, kernel)
            unit = torch.ones_like(op.spectrum.real)
            for lam in (.0025, .025, 25.):
                expected = op.solve(residual, lam)
                received = solve_weighted(op, residual, lam, unit)
                torch.testing.assert_close(received, expected, rtol=0, atol=0)
    return dict(passed=True, fft_solves=12, denoiser_forwards=0)


def reconstruct(root, model, plan, case, kernel, arm, weights, parent):
    import torch
    from utils.diag_noise_feasible_zaps import metrics

    directory = root/'cases'/case['key']/arm
    directory.mkdir(parents=True)
    record_path = directory/'run.json'
    row = dict(status='running', case=case['key'], task=case['task'], arm=arm)
    files.write(record_path, row)
    counts = {}
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)

            def progress(step, state):
                row.update(native_step=step, counts=dict(state))
                files.write(record_path, row)

            (out, detail), seconds = clock.clocks(torch.cuda.synchronize,
                lambda:sampler_run(model, plan, case, kernel, arm, weights, counts, progress))
            pairing = transfer.history.fingerprint(detail, shared.rng(), counts, 'pigdm')
            pairing.pop('calls')
        expected = [] if arm == 'BASE' else sorted(plan['active_times'], reverse=True)
        files.require([step['t'] for step in detail['trace'] if 'weight' in step] == expected and
                      tuple(detail['grid']) == tuple(counts['calls']) == GRID, 'Native control or denoiser grid changed')
        files.require(counts['forwards'] == counts['forward_attempts'] == counts['input_vjp_attempts'] ==
                      counts['base_regularized_solves'] == 100 and
                      counts.get('input_jvp_attempts', 0) == counts['optimizer_backward_attempts'] == 0,
                      '100 NFE / 100 input VJP budget mismatch')
        scalar_record = parent/'cases'/case['key']/'SCALAR'/'run.json'
        reference = files.read(scalar_record)
        files.require(reference['status'] == 'complete' and reference['case'] == case and
                      pairing == reference['pairing'], 'Frozen A126 sampling identity differs')
        if arm in ('BASE', 'SCALAR'):
            old_path = Path(case['baseline']) if arm == 'BASE' else scalar_record
            if arm == 'BASE':
                files.require(files.digest(old_path) == case['baseline_sha256'], 'A124R1 baseline record changed')
            old = files.read(old_path)
            files.require(old['status'] == 'complete', 'Historical control is incomplete')
            out_path = audit.qualified_artifact(old_path.parent, 'output.pt', old)
            archived = audit.checked_tensor(out_path, 'cpu')
            relative = audit.norm(out-archived)/max(audit.norm(archived), 1e-30)
            files.require(relative <= 1e-4, f'{arm} archived control replay differs: {relative:.3g}')
            row['archived_relative_l2'] = relative
        evaluation_started = time.perf_counter()
        truth = shared.load_image(case['image'])
        quality = metrics(out, truth)
        raw_mse = audit.mse(out-truth)
        evaluation_seconds = time.perf_counter()-evaluation_started
        output_started = time.perf_counter()
        torch.save(out, directory/'output.pt')
        files.write(directory/'trace.json', detail['trace'])
        row.update(status='complete', quality=quality, full_seconds=seconds, counts=counts,
                   raw_mse=raw_mse, pairing=pairing, evaluation_seconds=evaluation_seconds,
                   output_seconds=time.perf_counter()-output_started,
                   artifacts={name:files.digest(directory/name) for name in ('output.pt', 'trace.json')})
        print('A130 COMPLETE', case['key'], arm, quality, f'{seconds:.2f}s', flush=True)
        return row
    except BaseException as exc:
        row.update(status='failed', error=repr(exc), counts=counts)
        raise
    finally:
        files.write(record_path, row)


def summarize(rows):
    result = []
    metrics = transfer.METRICS
    for task in transfer.TASKS:
        by_arm = {arm:{record['case']:record for record in rows
                       if record['task'] == task and record['arm'] == arm} for arm in ARMS}
        if not all(len(items) == 3 for items in by_arm.values()):
            continue
        candidate, scalar = (by_arm[arm] for arm in ('COVARIANCE_SCALAR', 'SCALAR'))
        files.require(candidate.keys() == scalar.keys() == by_arm['BASE'].keys(), 'Incomplete paired task')
        keys = sorted(scalar)
        delta = {k:statistics.mean(candidate[key]['quality'][k]-scalar[key]['quality'][k] for key in keys)
                 for k in metrics}
        wins = sum(candidate[key]['quality']['psnr'] > scalar[key]['quality']['psnr'] for key in keys)
        arms = {}
        for arm, items in by_arm.items():
            arms[arm] = dict(quality={k:statistics.mean(row['quality'][k] for row in items.values()) for k in metrics},
                             seconds=statistics.mean(row['full_seconds'] for row in items.values()),
                             nfe=100, input_vjp=100)
        result.append(dict(task=task, images=len(keys), arms=arms, covariance_minus_scalar=delta, psnr_wins=wins,
                           time_ratio=arms['COVARIANCE_SCALAR']['seconds']/arms['SCALAR']['seconds'],
                           positive=(delta['psnr'] >= .1 and wins >= 2 and delta['ssim'] >= 0 and delta['lpips'] <= 0)))
    return result


def execute(parent, root, calibration_dir):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision
    from utils import pigdm_noisy_r1_runtime as runtime

    plan = files.read(parent/'plan.json')
    files.require(plan['experiment'] == 'A126' and plan['host'] == 'pigdm' and plan['nfe'] == 100 and
                  files.read(parent/'summary.json')['status'] == 'complete',
                  'A130 requires completed A126 PiGDM')
    cases = plan['cases']
    files.require(len(cases) == 6 and len({c['image_id'] for c in cases}) == 3 and
                  {c['task'] for c in cases} == set(transfer.TASKS) and
                  all(c['sigma'] == .05 for c in cases), 'Unexpected frozen evaluation panel')
    history_path = Path(plan['source'])/'plan.json'
    files.require(files.digest(history_path) == plan['source_plan_sha256'], 'A124R1 plan changed')
    history = files.read(history_path)
    calibration = profile_inputs(plan, history, calibration_dir)
    sources = {str(p):files.digest(p) for p in (Path(__file__), CARD, parent/'plan.json', history_path)}
    inputs = dict(sources=sources, calibration=calibration, cases=cases,
                  calibration_draws=DRAWS, calibration_times=ANCHORS, arms=ARMS, native_grid=GRID,
                  weight_rule='power of forward-noised clean prediction errors, normalized over full FFT, log-SNR interpolation')
    if (root/'summary.json').exists():
        report = files.read(root/'summary.json')
        files.require(report['status'] == 'complete' and
                      files.read(root/'resolved_config.json')['sources'] == sources and
                      files.read(root/'resolved_config.json')['calibration'] == calibration,
                      f'Existing partial or different run retained at {root}')
        for group in report['groups']:
            print('A130 SUMMARY', group, flush=True)
        return
    checkpoint = plan['checkpoint']
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'] and
                  files.digest(plan['motion_kernel']) == history['motion_kernel_sha256'],
                  'Archived model or motion kernel changed')
    for case in cases:
        files.require(files.digest(case['image']) == case['image_sha256'] and
                      files.digest(case['y_path']) == case['y_sha256'], 'Frozen input changed')
    admission.check(torch, root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json', inputs)
    path = root/'summary.json'
    report = dict(experiment='A130', status='running', completed_runs=0, planned_runs=18,
                  reconstruction_nfe=0, calibration_nfe=0, groups=[])
    files.write(path, report)
    rows = []
    started = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)
        model.model.eval()
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-started
        kernel = np.load(plan['motion_kernel'], allow_pickle=False)
        diffusion = runtime.original_diffusion()
        indices = torch.tensor(GRID, device=model.device, dtype=torch.long)
        alphas = dict(zip(GRID, diffusion.alpha(indices).tolist()))
        with precision('LEGACY_IEEE'), frozen_network(model.model):
            report['unit_covariance_gate'] = unit_gate(plan, cases, model, kernel)
            weights, profile = calibrate(model, calibration, alphas)
            files.write(root/'covariance_profile.json', profile)
            report.update(calibration_nfe=profile['nfe'], calibration_seconds=profile['seconds'])
            files.write(path, report)
            print('A130 ADMITTED', report['unit_covariance_gate'], 'calibration_nfe=', profile['nfe'], flush=True)
            with input_vjp_parameters(model.model):
                for index, case in enumerate(cases):
                    order = ARMS[index % len(ARMS):]+ARMS[:index % len(ARMS)]
                    for arm in order:
                        rows.append(reconstruct(root, model, plan, case, kernel, arm, weights, parent))
                        report.update(completed_runs=len(rows), reconstruction_nfe=sum(
                            row['counts']['forwards'] for row in rows), groups=summarize(rows))
                        files.write(path, report)
        files.require(report['completed_runs'] == 18 and len(report['groups']) == 2, 'Incomplete A130')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), recorded_cost_may_be_lower_bound=True)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        files.write(path, report)
    for group in report['groups']:
        print('A130 SUMMARY', group, flush=True)
    print('A130 REPORT', path, 'reconstruction_nfe=', report['reconstruction_nfe'],
          'calibration_nfe=', report['calibration_nfe'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=a128.PARENT)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--calibration-dir', type=Path)
    args = parser.parse_args()
    parent, root = args.parent.resolve(), args.output.resolve()
    files.require(root != parent and not root.is_relative_to(parent), 'Use a separate A130 directory')
    execute(parent, root, args.calibration_dir)


if __name__ == '__main__':
    main()
