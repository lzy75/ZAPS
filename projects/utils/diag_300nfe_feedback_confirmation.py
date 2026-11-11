"""A110 removes SR null-space increments; A111 repeats blur with new observation noise."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils import sr_observable_counterfactual as sr
from utils.diag_sr_inpainting_followup import A103, METRICS, source_case
from utils.paper_clock_alignment import digest, grid, read, require, write
from utils.shared_prefix_feedback import ARMS, IMAGES, ROOT as A102_ROOT

A102 = Path(A102_ROOT)
ROW = 'OBSERVABLE_INCREMENT_SUFFIX'
ROOTS = {'sr': A103.with_name('A110_20261111'), 'blur': A103.with_name('A111_20261111')}
CARDS = {'sr': 'A110_sr_observable_increment.md', 'blur': 'A111_blur_observation_seed.md'}
TOL = 128 * 2**-23
OBS_SEED_OFFSET = 20261111


def select_cases(mode, plan):
    tasks = ('super_resolution',) if mode == 'sr' else ('gaussian_deblur', 'motion_deblur')
    sigmas = (.025, .10) if mode == 'sr' else (.05,)
    cases = [c for c in plan['cases'] if c['task'] in tasks and c['sigma'] in sigmas]
    require(len(cases) == 20 and {(c['task'], c['image_id'], c['sigma']) for c in cases} ==
            {(t, i, s) for t in tasks for i in IMAGES for s in sigmas}, 'Expected frozen twenty-case panel')
    return cases


def rebuild_observation_case(case):
    seed = case['measurement_seed'] + OBS_SEED_OFFSET
    return dict(case, measurement_seed=seed, key=f'{case["key"]}_obs{seed}')


class ObservableProjection:
    def __init__(self, operator, device):
        import numpy as np
        import torch
        from utils.equal_nfe_feedback_runtime_r3 import exact_diagonal_with_native_straight_through

        self.diagonal_update = exact_diagonal_with_native_straight_through
        self.audit = sr.Projector(*sr.native_matrices(operator.base.down_sample, (1, 3, 256, 256)))
        self.axes = [torch.as_tensor(axis, device=device) for axis in self.audit.axes]
        sample = np.random.default_rng(20261111).standard_normal((1, 3, 256, 256))
        with torch.no_grad():
            native = operator.H(torch.as_tensor(sample, device=device)).cpu().numpy()
        self.forward_error = sr.relative(native, self.audit.H(sample))
        require(self.forward_error <= 1e-10, 'Separable matrices differ from native H')

    def row(self, extra):
        qy, qx = self.axes
        return (qy @ (qy.T @ extra.double() @ qx) @ qx.T).to(extra.dtype)

    def update(self, zaps, operator, clean, q, residual, alpha, sigma, counts):
        import torch

        diagonal, detail, counts = self.diagonal_update(
            zaps, operator, clean, q, residual, alpha, sigma, counts)
        with torch.no_grad():
            scalar = detail['scalar']
            start = scalar['gamma'] * q.detach()
            extra = diagonal.detach() - start
            row = self.row(extra)
            null = extra - row
            selected = start + row
            a = alpha.sqrt().clamp(min=1e-8)
            base = operator.H(clean)
            target = scalar['weight'] * residual.double()
            response_s = operator.H(clean + start/a) - base
            response_d = operator.H(clean + diagonal.detach()/a) - base
            response_r = operator.H(clean + selected/a) - base
            norm = lambda v: float(v.double().norm())
            losses = [.5 * norm(resp.double()-target)**2 for resp in (response_s, response_d, response_r)]
            response_gap = norm(response_r-response_d)
            response_scale = max(1., norm(base), norm(response_d))
            null_energy = norm(null)**2
            extra_energy = norm(extra)**2
            require(bool(torch.isfinite(selected).all()) and response_gap <= TOL * response_scale
                    and abs(losses[2]-losses[1]) <= TOL * max(1., losses[1], losses[2]),
                    'Observable projection changed the frozen proxy response')
            detail.update(proxy_at_exact_scalar=losses[0], proxy_at_full_diagonal=losses[1],
                          proxy_at_row_increment=losses[2], response_gap=response_gap,
                          response_relative_gap=response_gap/response_scale,
                          extra_energy=extra_energy, removed_null_energy=null_energy,
                          removed_null_fraction=null_energy/max(extra_energy, 1e-30))
        return selected + (q - q.detach()), detail, counts

    def risk(self, output, gt, y):
        import numpy as np

        x, truth = output.numpy(), gt.numpy()
        values = sr.risks(x, truth, self.audit)
        h_x = self.audit.H(x)
        values.update(noisy_y_mse=float(np.mean((h_x-y.cpu().numpy())**2)),
                      clean_H_mse=float(np.mean((h_x-self.audit.H(truth))**2)))
        return values


def summarize(rows, mode):
    groups = []
    for task, sigma in sorted({(r['case']['task'], r['case']['sigma']) for r in rows}):
        cell = [r for r in rows if (r['case']['task'], r['case']['sigma']) == (task, sigma)]
        arms = (*ARMS, ROW) if mode == 'sr' else ARMS
        group = dict(task=task, sigma=sigma, images=len(cell), arms={}, comparisons=[])
        for arm in arms:
            samples = [row['arms'][arm] for row in cell]
            group['arms'][arm] = dict(quality={k:mean(r['quality'][k] for r in samples) for k in METRICS},
                                      mean_seconds=mean(r['seconds'] for r in samples), nfe=300)
        pairs = [(ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])]
        if mode == 'sr':
            pairs += [(ROW, ARMS[2]), (ROW, ARMS[1])]
            steps = [r['arms'][ROW]['projection'] for r in cell]
            group['projection'] = dict(steps=sum(s['steps'] for s in steps),
                max_response_relative_gap=max(s['max_response_relative_gap'] for s in steps),
                removed_null_energy_fraction=sum(s['removed_energy'] for s in steps) /
                max(sum(s['extra_energy'] for s in steps), 1e-30))
        for cand, ref in pairs:
            samples = [(row['arms'][cand], row['arms'][ref]) for row in cell]
            delta = {k:mean(a['quality'][k]-b['quality'][k] for a, b in samples) for k in METRICS}
            errors = {k:mean(a['errors'][k]-b['errors'][k] for a, b in samples)
                      for k in samples[0][0]['errors']}
            wins = sum(a['quality']['psnr'] > b['quality']['psnr'] for a, b in samples)
            item = dict(candidate=cand, reference=ref, delta=delta, error_delta=errors, psnr_wins=wins,
                        positive=len(cell) == 10 and wins >= 7 and delta['psnr'] >= .10
                        and delta['ssim'] >= 0 and delta['lpips'] <= 0)
            if mode == 'sr':
                item['raw_mse_wins'] = sum(a['errors']['raw_mse'] < b['errors']['raw_mse'] for a, b in samples)
                if cand == ROW and ref == ARMS[2]:
                    item['null_removal_reduced_raw_mse'] = (len(cell) == 10 and errors['raw_mse'] < 0
                                                            and item['raw_mse_wins'] >= 7)
            group['comparisons'].append(item)
        groups.append(group)
    return groups


def execute(mode, parent, root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import diag_equal_nfe_feedback as fixtures
    from utils import diag_shared_prefix_feedback as baseline, historical_ffhq_runtime as precision
    from utils import equal_nfe_feedback_runtime_r3 as bridge, paper_clock_runtime as clock
    from utils.a67_model_freeze import frozen_network
    from utils.diag_noise_feasible_zaps import metrics
    from utils.shared_prefix_feedback_runtime_multinoise import make_operator, run_suffix, train_prefix

    check = baseline.check
    if mode == 'sr':
        from utils.diag_shared_prefix_feedback_multinoise import check
    plan = check(parent)
    complete = read(parent/'summary.json')
    require(complete['completed_cases'] == complete['planned_cases'], 'Completed parent required')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 6 * 1024**3, 'Need at least 6 GiB free')
    require(plan['grid'] == grid() and digest(PROJECTS/'modules/models/ffhq_10m.pt') ==
            plan['checkpoint']['sha256'], 'Grid or checkpoint differs')
    require(not root.resolve().is_relative_to(parent.resolve()), 'Use an independent result directory')
    entries = []
    for old in select_cases(mode, plan):
        folder, _, y_path = source_case(parent, old)
        case = old if mode == 'sr' else rebuild_observation_case(old)
        entries.append(dict(case=case, parent_case=old, observation=str(y_path),
                            observation_sha256=digest(y_path), parent_record_sha256=digest(folder/'run.json')))

    experiment = 'A110' if mode == 'sr' else 'A111'
    arms = (*ARMS, ROW) if mode == 'sr' else ARMS
    card = PROJECTS/'experiment_cards'/CARDS[mode]
    kernel_path = parent/'motion_kernel.npy'
    root.mkdir(parents=True)
    if mode == 'blur':
        (root/'motion_kernel.npy').write_bytes(kernel_path.read_bytes())
        kernel = np.load(root/'motion_kernel.npy', allow_pickle=False)
        require(digest(root/'motion_kernel.npy') == plan['motion_kernel_sha256'], 'Motion kernel differs')
    else:
        kernel = None
    write(root/'plan.json', dict(experiment=experiment, parent=str(parent), parent_sha256=digest(parent/'plan.json'),
          cases=entries, arms=arms, grid=plan['grid'], checkpoint=plan['checkpoint'], configs=plan['configs'],
          motion_kernel_sha256=plan['motion_kernel_sha256'], observation_seed_offset=OBS_SEED_OFFSET if mode=='blur' else 0,
          source_sha256=digest(__file__), card_sha256=digest(card), nfe_per_arm=300,
          joint_forward_cap=len(entries)*(270+30*len(arms)),
          code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip()))
    (root/'experiment_card.md').write_bytes(card.read_bytes())
    report = dict(experiment=experiment, status='running', planned_cases=len(entries), completed_cases=0)
    if mode == 'blur':
        report['original_observation_comparisons'] = [r for r in complete['comparisons']
            if r['task'] in ('gaussian_deblur', 'motion_deblur')]
    write(root/'summary.json', report)
    rows, active = [], None
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        descending = plan['grid'][::-1]
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            projection = (ObservableProjection(make_operator(entries[0]['case'], model.device), model.device)
                          if mode == 'sr' else None)
            if projection is not None:
                report['projection_gate'] = dict(native_H_error=projection.forward_error, axes=projection.audit.info)
            for entry in entries:
                case = entry['case']
                dest = root/'cases'/case['key']
                active = dict(status='running', case=case, counts={}, arms={})
                write(dest/'run.json', active)
                y_path = Path(entry['observation'])
                require(digest(case['image']) == case['image_sha256'] and digest(y_path) == entry['observation_sha256'],
                        'Frozen image or observation changed')
                if mode == 'blur':
                    observation = fixtures.observation(root, case, model.device)
                    original_y = torch.load(y_path, map_location='cpu', weights_only=True)['y']
                    y_path = Path(observation['path'])
                active['observation_sha256'] = digest(y_path)
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y']
                if mode == 'blur':
                    require(y.shape == original_y.shape and not torch.equal(y, original_y),
                            'New observation matches parent or changes shape')
                    active['observation_delta_mse'] = float((y-original_y).double().square().mean())
                y = y.to(model.device)
                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    prefix_op = make_operator(case, model.device, count=count, kernel=kernel)
                    prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, prefix_op, y, case))
                baseline.validate(count, 270, 9)
                require(count['calls'] == descending*9, 'Prefix grid differs')
                final_rng, outputs = None, {}
                for arm in arms:
                    mode_patch = (patch.object(bridge, 'exact_diagonal_with_native_straight_through', projection.update)
                                  if arm == ROW else nullcontext())
                    with mode_patch, clock.counters(model.model) as count:
                        active['counts'][arm] = count
                        op = make_operator(case, model.device, count=count, kernel=kernel)
                        (output, detail), seconds = clock.clocks(torch.cuda.synchronize,
                            lambda: run_suffix(model, op, y, case, ARMS[2] if arm == ROW else arm, prefix))
                    baseline.validate(count, 30, 1)
                    require(count['calls'] == descending, 'Suffix grid differs')
                    if final_rng is not None:
                        require(baseline.rng_equal(final_rng, detail['final_rng']), 'Suffix RNG differs')
                    final_rng = detail['final_rng']
                    results = dict(seconds=prefix_seconds+seconds, suffix_seconds=seconds,
                                   module_counts=detail['module_counts'])
                    if arm == ROW:
                        trace = detail['trace']
                        require(len(trace) == 30, 'Incomplete projection trace')
                        results['projection'] = dict(steps=len(trace),
                            max_response_relative_gap=max(s['response_relative_gap'] for s in trace),
                            removed_energy=sum(s['removed_null_energy'] for s in trace),
                            extra_energy=sum(s['extra_energy'] for s in trace))
                    active['arms'][arm] = results
                    outputs[arm] = output
                    write(dest/(arm+'_trace.json'), detail['trace'])
                gt = load_image_as_tensor(case['image']).cpu()
                for arm, output in outputs.items():
                    if projection is not None:
                        errors = projection.risk(output, gt, y)
                    else:
                        with torch.no_grad():
                            measured = op.H(output.to(model.device))
                            errors = dict(noisy_y_mse=float((measured-y).double().square().mean()),
                                          clean_H_mse=float((measured-op.H(gt.to(model.device))).double().square().mean()))
                    active['arms'][arm].update(quality=metrics(output, gt), errors=errors)
                    torch.save(output, dest/(arm+'.pt'))
                active['status'] = 'complete'
                write(dest/'run.json', active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), groups=summarize(rows, mode))
                write(root/'summary.json', report)
                print(f'{experiment} COMPLETE: {len(rows)}/{len(entries)} {case["key"]}', flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if active is not None:
            active.update(status='failed', error=repr(error))
            write(root/'cases'/active['case']['key']/'run.json', active)
        raise
    finally:
        attempted = rows + ([active] if active is not None else [])
        counts = [count for row in attempted for count in row['counts'].values()]
        report['actual_cost'] = {k:sum(c.get(k, 0) for c in counts) for k in
            ('forward_attempts', 'forwards', 'optimizer_backward_attempts', 'input_vjp_attempts', 'input_jvp_attempts')}
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(root/'summary.json', report)
    print(f'{experiment} SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=tuple(ROOTS))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    root = args.root or ROOTS[args.mode]
    if args.status:
        print(json.dumps(read(root/'summary.json'), ensure_ascii=False, indent=2))
    else:
        execute(args.mode, args.parent or (A103 if args.mode == 'sr' else A102), root)


if __name__ == '__main__':
    main()
