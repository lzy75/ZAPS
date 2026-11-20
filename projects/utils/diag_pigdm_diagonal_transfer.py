"""A119, equal-budget closed scalar/diagonal transfer on A85R1 motion cases."""
import argparse
import ast
from pathlib import Path
import inspect
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from utils import diag_pigdm_noisy_r1 as parent
from utils import pigdm_noisy_r1_runtime as native
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import native_ray_transfer_runtime as shared
from utils.equal_nfe_feedback_runtime import diagonal_update
from utils.a54_session import exclusive

ROOT = Path('/data0/lzy/ZAPS-results/diag_pigdm_noisy_transfer/A119_ffhq_20261120')
BRANCH = 'results/pigdm-diagonal-transfer-20261120'
ARMS = ('BASE', 'SHRINK', 'DIAGONAL')
METRICS = ('psnr', 'ssim', 'lpips')
NFE = 100
REPLAY_TOLERANCE = 1e-4


def diagonal_sampler(feedback):
    """Keep the A85R1 sampler verbatim apart from the additive correction."""
    source = inspect.getsource(native.run)
    before = "x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
    after = ("delta,row=feedback(op,m.detach(),q,r,a,case['sigma'])\n"
             "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    files.require(source.count(before) == 1, 'A85R1 correction interface changed')
    filename = str(Path(native.__file__))
    module = ast.parse(source.replace(before, after), filename)
    bindings = {**vars(native), 'feedback': feedback}
    exec(compile(module, filename, 'exec'), bindings)
    return bindings['run']


def make_feedback(wavelet, observation_error, count, trace):
    import torch

    def feedback(op, clean, q, residual, alpha, sigma):
        scale = math.sqrt(alpha)
        op.module = True
        try:
            delta, row, _ = diagonal_update(
                wavelet, op, clean, q, residual, q.new_tensor(alpha), sigma, count)
            response = op.H(q / scale).double()
            scalar = native.policy.select(
                float((residual.double() * response).sum()),
                float(residual.double().square().sum()), float(response.square().sum()),
                residual.numel(), sigma)
            scalar_delta = scalar['gamma'] * q
            scalar_response = op.H(scalar_delta / scale).double()
            diag_response = op.H(delta / scale).double()
            target = scalar['weight'] * residual.double()
            initial_error = scalar_response - target
            final_error = diag_response - target
            gain = .5 * float(initial_error.square().mean() - final_error.square().mean())
            tol = 128 * torch.finfo(q.dtype).eps * max(1., float(initial_error.square().mean()))
            anchor_gap = abs(float(initial_error.square().mean()) -
                             2 * row['proxy_initial'] / residual.numel())
            files.require(anchor_gap <= tol and gain >= -tol, 'Fixed-state scalar anchor/descent gate failed')
        finally:
            op.module = False

        # y - H(clean) + H(truth) - y equals the clean target.
        # The truth-derived quantity is used only after the update is chosen.
        clean_target = residual.double() + observation_error
        diff = diag_response - scalar_response
        credit = float(((clean_target - target) * diff).mean())
        clean_gain = .5 * float((scalar_response-clean_target).square().mean() -
                                (diag_response-clean_target).square().mean())
        files.require(math.isclose(clean_gain, gain + credit, rel_tol=1e-8, abs_tol=1e-10),
                      'Clean-observation decomposition differs')
        result = dict(proxy_gain=gain, clean_observation_gain=clean_gain, target_credit=credit,
                      alignment_step=row['eta'], anchor_gap=anchor_gap)
        trace.append(result)
        return delta, dict(gamma=scalar['gamma'], **result)

    return feedback


def pair_fingerprint(detail, count):
    return dict(initial=parent.fingerprint(detail['initial']),
                entry_cpu=parent.fingerprint(detail['entry_rng']['cpu']),
                entry_cuda=[parent.fingerprint(value) for value in detail['entry_rng']['cuda']],
                final_cpu=parent.fingerprint(detail['final_rng']['cpu']),
                final_cuda=[parent.fingerprint(value) for value in detail['final_rng']['cuda']],
                grid=detail['grid'], calls=count['calls'])


def measure(root, source, case, arm, observation, model, wavelet):
    import torch
    from modules.dataset_loader import tensor_to_image

    folder = root / 'cases' / case['key'] / arm
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', case=case, arm=arm, counts={})
    record_path = folder / 'run.json'
    files.write(record_path, record)
    count = {}
    try:
        truth = shared.load_image(case['image'], model.device)
        audit = []
        with parent.precision.precision('LEGACY_IEEE'), shared.counters(model.model) as count:
            count.update(module_dwt_analysis=0, module_dwt_synthesis=0,
                         module_adjoint_actions=0, audit_H_actions=0)
            def tick(step, observed):
                record['counts'] = dict(observed)
                files.write(record_path, record)
                print('A119 PROGRESS:', case['key'], arm, step, '/', NFE, flush=True)

            def operation():
                y = torch.load(observation['path'], map_location='cpu', weights_only=True)['y'].to(model.device)
                op = parent.fresh_operator(source, case, model.device, observation, count)
                sampler = native.run
                mode = arm
                if arm == 'DIAGONAL':
                    op.transpose = op.Ht
                    with torch.no_grad():
                        observation_error = op.H(truth).double() - y.double()
                    count['base_H_actions'] -= 1
                    count['audit_H_actions'] += 1
                    sampler = diagonal_sampler(make_feedback(wavelet, observation_error, count, audit))
                    mode = 'BASE'
                return sampler('A119', model, op, y, case, mode, count, progress=tick)

            (output, detail), elapsed = clock.clocks(torch.cuda.synchronize, operation)
        files.require(count['forward_attempts'] == count['forwards'] == count['input_vjp_attempts'] ==
                      count['corrections'] == NFE and count['optimizer_backward_attempts'] == 0,
                      'A119 network budget differs')
        files.require(len(audit) == (NFE if arm == 'DIAGONAL' else 0), 'A119 correction trace incomplete')
        pair = pair_fingerprint(detail, count)
        relative = None
        if arm in ('BASE', 'SHRINK'):
            original_path = Path(source['root']) / 'cases' / case['key'] / arm / 'output.pt'
            reference = torch.load(original_path, map_location='cpu', weights_only=True)
            relative = float((output.double() - reference.double()).norm() /
                             reference.double().norm().clamp_min(1e-30))
            files.require(relative <= REPLAY_TOLERANCE, f'A85R1 {arm} reproduction differs: {relative:.3g}')
        if arm != 'BASE':
            prior = files.read(folder.parent / 'BASE' / 'run.json')
            files.require(prior['pair'] == pair, 'A119 initialization, RNG or denoiser grid not paired')
        metrics = parent.common.metrics(output, truth.cpu())
        torch.save(output, folder / 'output.pt')
        tensor_to_image(output.squeeze(0), denormalize=True).save(folder / 'output.png')
        files.write(folder / 'trace.json', detail['trace'])
        record.update(status='complete', counts=count, pair=pair, metrics=metrics,
                      historical_relative_l2=relative, full_seconds=elapsed,
                      proxy_descents=sum(row['proxy_gain'] > 0 for row in audit),
                      proxy_clean_conflicts=sum(row['proxy_gain'] > 0 and row['clean_observation_gain'] < 0
                                                for row in audit))
        files.write(record_path, record)
        print('A119 COMPLETE:', case['key'], arm, metrics, round(elapsed, 2), flush=True)
    except BaseException as error:
        record.update(status='failed', counts=count, error=repr(error), traceback=traceback.format_exc())
        files.write(record_path, record)
        raise


def report(root):
    records = [files.read(path) for path in sorted((root / 'cases').glob('*/*/run.json'))]
    complete = {arm: {r['case']['key']: r for r in records if r['arm'] == arm and r['status'] == 'complete'}
                for arm in ARMS}
    groups = {}
    comparisons = []
    for arm, rows in complete.items():
        if rows:
            groups[arm] = dict(images=len(rows), quality={k: statistics.mean(r['metrics'][k] for r in rows.values())
                                                         for k in METRICS},
                               full_seconds=statistics.mean(r['full_seconds'] for r in rows.values()))
    for arm, reference in (('SHRINK', 'BASE'), ('DIAGONAL', 'SHRINK')):
        keys = sorted(complete[arm].keys() & complete[reference].keys())
        if keys:
            values = [{k: complete[arm][key]['metrics'][k] - complete[reference][key]['metrics'][k]
                       for k in METRICS} for key in keys]
            delta = {k: statistics.mean(row[k] for row in values) for k in METRICS}
            wins = sum(row['psnr'] > 0 for row in values)
            comparisons.append(dict(candidate=arm, reference=reference, paired_images=len(keys), delta=delta,
                                    psnr_wins=wins, positive=(len(keys) == 3 and wins >= 2 and
                                    delta['psnr'] >= .1 and delta['ssim'] >= 0 and delta['lpips'] <= 0)))
    result = dict(experiment='A119', completed_runs=sum(map(len, complete.values())), planned_runs=9,
                  groups=groups, comparisons=comparisons,
                  timing='Full 100-step run, including progress and diagonal audit; excludes model load, metrics and output writes.',
                  proxy_descents=sum(r.get('proxy_descents', 0) for r in complete['DIAGONAL'].values()),
                  proxy_clean_conflicts=sum(r.get('proxy_clean_conflicts', 0) for r in complete['DIAGONAL'].values()),
                  recorded_cost={key: sum(r['counts'].get(key, 0) for r in records)
                                 for key in ('forwards', 'forward_attempts', 'input_vjp_attempts', 'module_H_actions',
                                             'module_adjoint_actions', 'module_dwt_analysis', 'module_dwt_synthesis',
                                             'audit_H_actions')},
                  failures=[dict(case=r['case']['key'], arm=r['arm'], error=r['error'])
                            for r in records if r['status'] == 'failed'])
    return result


def execute(root, source_root, device_id):
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters

    source = parent.check(source_root)
    source['root'] = str(source_root)
    cases = [c for c in source['cases'] if c['task'] == 'motion_deblur']
    files.require(source['dataset'] == 'ffhq' and len(cases) == 3, 'Expected completed A85R1 motion panel')
    files.require(str(device_id) and torch.cuda.device_count() == 1, 'Allocate one visible CUDA device')
    row = next((row for row in parent.gpu_inventory() if device_id in (row['index'], row['uuid'])), None)
    files.require(row and row['free_bytes'] >= parent.rules.min_free('ffhq'), 'Selected GPU needs at least 6 GiB free')
    observations = {}
    for case in cases:
        obs_dir = source_root / 'observations' / case['key']
        files.require((obs_dir / 'observation.json').is_file(), 'Original A85R1 observation missing')
        observations[case['key']] = parent.observation(source_root, source, case, 'cpu')
        files.require(all(parent.completed(source_root, case, arm) is not None for arm in ('BASE', 'SHRINK')),
                      'Original A85R1 BASE and SHRINK records must be complete')
        files.require(files.digest(case['image']) == source['inputs'][case['image']], 'Source image changed')
    checkpoint = source['checkpoint']
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'], 'Source model weights changed')
    files.write(root / 'plan.json', dict(experiment='A119', source=str(source_root), cases=cases,
                                       parent_plan=files.digest(source_root / 'plan.json'),
                                       observations=observations, arms=ARMS, device=row,
                                       checkpoint=checkpoint, replay_tolerance=REPLAY_TOLERANCE,
                                       source_hashes={n: files.digest(path) for n, path in
                                                      (('pilot', __file__), ('native_loop', native.__file__),
                                                       ('diagonal_update', inspect.getfile(diagonal_update)))}))
    subprocess.run([sys.executable, '-m', 'unittest', '-v', 'utils.test_pigdm_noisy_r1_runtime'],
                   cwd=parent.PROJECTS, check=True)
    started = time.perf_counter()
    model = load_diffusion_model('ffhq', 'cuda:0')
    model.model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)
    model.model.eval()
    wavelet = SimpleNamespace(dwt=OrthogonalDWT2D(wave='db4', level=3).to(model.device))
    print('A119 model load seconds:', round(time.perf_counter()-started, 2), flush=True)
    with frozen_network(model.model), input_vjp_parameters(model.model):
        for index, case in enumerate(cases):
            obs = observations[case['key']]
            # Compare feedback arms in alternating order, after checking the native arm.
            order = ARMS if index % 2 == 0 else (ARMS[0], ARMS[2], ARMS[1])
            for arm in order:
                measure(root, source, case, arm, obs, model, wavelet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, default=Path(parent.rules.root_for('ffhq')))
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--status', action='store_true')
    action.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(report(args.root))
        return
    if args.archive:
        from utils import archive_refinement_confirmation as archive
        with exclusive(args.root.parent / (args.root.name + '.lock')):
            files.require((args.root / 'summary.json').is_file(), 'No A119 summary to archive')
            archive.BRANCH = BRANCH
            bundle, commit = archive.export_and_commit({'A119': args.root})
            subprocess.run(['git', 'push', 'origin', BRANCH], cwd=parent.PROJECTS.parent, check=True)
            print('A119 Archive:', bundle, 'commit:', commit, 'branch:', BRANCH, flush=True)
        return
    device = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not device or ',' in device:
        parser.error('Set CUDA_VISIBLE_DEVICES to one available GPU index or UUID')
    args.root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(args.root.parent / (args.root.name + '.lock')):
        # Keep prior attempts, including incomplete runs, immutable.
        args.root.mkdir()
        try:
            execute(args.root, args.source, device)
        except BaseException:
            (args.root / 'failure.log').write_text(traceback.format_exc())
            raise
        finally:
            summary = report(args.root)
            files.write(args.root / 'summary.json', summary)
            print('A119 SUMMARY:', summary, flush=True)


if __name__ == '__main__':
    main()
