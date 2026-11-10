"""Read A104, audit cached A102/A103 outputs, and run the A105 noise ablation."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
sys.path.insert(0, str(PROJECTS.parent))

from utils.paper_clock_alignment import digest, read, require, write

A104 = Path('/data0/lzy/ZAPS-results/diag_actual_response_budget/A104R1_20261111')
A103 = Path('/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A103_20261111')
A105 = Path('/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A105_20261111')
TASKS = ('super_resolution', 'inpainting')
IMAGES = ('00001', '00004', '00008')
BASE = 'ZAPS_SUFFIX'
ARMS = ('CLOSED_SCALAR_SUFFIX', 'CLOSED_DIAG_SUFFIX')
METRICS = ('psnr', 'ssim', 'lpips')
FROZEN_NOISE = .10
CONTROLLER_NOISES = (.10, .05)
CARD = PROJECTS / 'experiment_cards/A105_controller_noise_ablation.md'


def a104_status(root):
    plan = read(root / 'plan.json')
    cases = [case for case in plan['cases'] if case['task'] in TASKS]
    rows = []
    session = root.with_name(root.name + '_session') / 'session.json'
    if session.is_file():
        print('SESSION:', read(session).get('status'))
    for case in cases:
        record = root / 'cases' / case['key'] / 'run.json'
        row = read(record) if record.is_file() else {'status': 'not_started'}
        rows.append((case, row))
        print(case['key'], row['status'], row.get('error', ''))
        for arm, result in row.get('arms', {}).items():
            scores = ' '.join(f'{name}={result["metrics"][name]:.5f}' for name in METRICS)
            count = result['counts']
            print(f'  {arm} {scores} NFE={count["forwards"]}'
                  f' VJP={count.get("input_vjp_attempts", 0)} JVP={count.get("input_jvp_attempts", 0)}'
                  f' seconds={result["method_seconds"]:.3f}')
    for task in TASKS:
        expected = [row for case, row in rows if case['task'] == task]
        complete = [row for row in expected if row['status'] == 'complete']
        print(f'{task}: {len(complete)}/{len(expected)} complete')
        for arm in plan['arms']:
            if complete:
                scores = {key: mean(row['arms'][arm]['metrics'][key] for row in complete)
                          for key in METRICS}
                print(f'  {arm} paired_n={len(complete)} {scores}')
        if complete:
            for candidate, reference in (('REAL_DIAG445', 'REAL_RAY445'),
                                         ('REAL_DIAG445', 'ZAPS450'), ('ZAPS450', 'ZAPS300')):
                delta = {k: mean(r['arms'][candidate]['metrics'][k] -
                                 r['arms'][reference]['metrics'][k] for r in complete) for k in METRICS}
                print(f'  {candidate} - {reference}: {delta}')


def source_case(root, case):
    folder = root / 'cases' / case['key']
    record = folder / 'run.json'
    row = read(record)
    require(row['case'] == case and row['status'] == 'complete',
            f'Incomplete or mismatched case: {record}')
    observation = root / 'observations' / case['key'] / 'observation.json'
    y = read(observation)
    require(y['sigma'] == case['sigma'] and digest(y['path']) == y['sha256'],
            f'Observation changed: {observation}')
    require(digest(case['image']) == case['image_sha256'], f'Image changed: {record}')
    return folder, row, Path(y['path'])


def artifact(folder, row, arm, name):
    path = folder / name
    require(digest(path) == row['arms'][arm]['artifacts'][name], f'Artifact changed: {path}')
    return path


def cached_audit(root, report_path):
    import numpy as np
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.historical_ffhq_runtime import make_operator
    from utils import sr_observable_counterfactual as sr
    from utils.terminal_mask_counterfactual import region_errors

    torch.set_num_threads(2)
    op = make_operator('cpu')
    ay, ax = sr.native_matrices(op.base.down_sample, (1, 3, 256, 256))
    projector = sr.Projector(ay, ax)
    probe = np.random.default_rng(20261111).standard_normal((1, 3, 256, 256))
    relative_error = sr.relative(projector.H(probe), op.H(torch.from_numpy(probe)).numpy())
    require(relative_error <= 1e-10, 'Separable SR forward differs from native CPU H')
    plan = read(root / 'plan.json')
    cases = [case for case in plan['cases'] if case['task'] in TASKS]
    require(bool(cases), 'No registered SR/inpainting cases')
    output_rows = []
    for case in cases:
        folder, row, y_path = source_case(root, case)
        load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
        y = load(y_path)['y'].double()
        gt = load_image_as_tensor(case['image']).double()
        mask = None
        if case['task'] == 'inpainting':
            mask = load(y_path.parent / 'mask.pt').bool()
            require(mask.shape == gt[:, :1].shape, 'Cached mask shape differs')
        for arm in (BASE, *ARMS):
            x = load(artifact(folder, row, arm, arm + '.pt')).double()
            if mask is None:
                observed = projector.H(x.numpy())
                errors = sr.risks(x.numpy(), gt.numpy(), projector)
                errors.update(noisy_y_mse=float(np.mean((observed-y.numpy())**2)),
                              clean_H_mse=float(np.mean((observed-projector.H(gt.numpy()))**2)))
            else:
                errors = region_errors(x, gt, mask)
                visible = mask.expand_as(x)
                errors['noisy_y_mse'] = float((x[visible]-y[visible]).square().mean())
            zero_steps = []
            if arm != BASE:
                trace = read(artifact(folder, row, arm, arm + '_trace.json'))
                require(len(trace) == 30, 'Incomplete trace')
                zero_steps = [step['step'] + 1 for step in trace if step['scalar']['weight'] == 0]
            output_rows.append(dict(key=case['key'], task=case['task'], sigma=case['sigma'],
                                    arm=arm, errors=errors, omega_zero_steps=zero_steps,
                                    metrics=row['arms'][arm]['metrics']))
    groups = []
    for task in TASKS:
        for sigma in sorted({row['sigma'] for row in output_rows if row['task'] == task}):
            for arm in (BASE, *ARMS):
                rows = [r for r in output_rows if (r['task'], r['sigma'], r['arm']) == (task, sigma, arm)]
                errors = {key: mean(row['errors'][key] for row in rows)
                          for key in rows[0]['errors'] if not key.startswith('float_psnr')}
                zeros = {k: sum(k in row['omega_zero_steps'] for row in rows) for k in range(1, 31)}
                groups.append(dict(task=task, sigma=sigma, arm=arm, n=len(rows), errors=errors,
                                   omega_zero_by_step={k: n for k, n in zeros.items() if n}))
    report = dict(parent=str(root), new_nfe=0, sr_forward_relative_error=relative_error,
                  groups=groups, cases=output_rows)
    if report_path is not None:
        require(not report_path.resolve().is_relative_to(root.resolve()), 'Write report outside parent')
        require(not report_path.exists(), f'Existing audit retained: {report_path}')
        write(report_path, report)
    print('CACHED AUDIT:', json.dumps(groups, ensure_ascii=False), flush=True)


def sigma_ablation(parent, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import diag_shared_prefix_feedback as base, paper_clock_runtime as clock
    from utils import historical_ffhq_runtime as precision
    from utils.diag_noise_feasible_zaps import metrics
    from utils.diag_shared_prefix_feedback_multinoise import check
    from utils.shared_prefix_feedback_runtime_multinoise import make_operator, run_suffix, train_prefix
    from utils.a67_model_freeze import frozen_network
    from utils.terminal_mask_counterfactual import region_errors

    plan = check(parent)
    cases = [c for c in plan['cases'] if c['task'] in TASKS and c['image_id'] in IMAGES
             and c['sigma'] == FROZEN_NOISE]
    require({(c['task'], c['image_id']) for c in cases} ==
            {(task, image) for task in TASKS for image in IMAGES} and len(cases) == 6,
            'Expected exactly six fixed A103 cases')
    sources = {}
    for case in cases:
        folder, row, y_path = source_case(parent, case)
        for arm in ARMS:
            artifact(folder, row, arm, arm + '.pt')
        sources[case['key']] = dict(run_sha256=digest(folder / 'run.json'),
                                    observation_sha256=digest(y_path))
        if case['task'] == 'inpainting':
            sources[case['key']]['mask_sha256'] = digest(y_path.parent / 'mask.pt')
    require(torch.cuda.is_available(), 'A105 requires one assigned CUDA GPU')
    require(torch.cuda.mem_get_info(0)[0] >= 6 * 1024**3, 'A105 needs at least 6 GiB free')
    expected = plan['checkpoint']
    checkpoint = PROJECTS / 'modules/models/ffhq_10m.pt'
    require(digest(checkpoint) == expected['sha256'], 'FFHQ checkpoint differs from A103')
    require(not root.exists(), f'Existing A105 run retained: {root}')
    require(not root.resolve().is_relative_to(parent.resolve()), 'Use a separate result root')
    root.mkdir(parents=True)
    write(root / 'plan.json', dict(experiment='A105_controller_noise_ablation', parent=str(parent),
          cases=cases, inputs=sources, checkpoint=expected, parent_plan_sha256=digest(parent/'plan.json'),
          source_sha256=digest(__file__), card_sha256=digest(CARD),
          observation_sigma=FROZEN_NOISE, controller_sigmas=CONTROLLER_NOISES,
          code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
          torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
          actual_new_nfe_cap=2340, method_nfe=300))
    (root / 'experiment_card.md').write_bytes(CARD.read_bytes())
    report = dict(status='running', completed_cases=0, planned_cases=6, cases=[])
    write(root / 'summary.json', report)
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        grid = plan['grid'][::-1]
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            for case in cases:
                folder, old, y_path = source_case(parent, case)
                save = root / 'cases' / case['key']
                row = dict(case=case, counts={}, arms={})
                report['cases'].append(row)
                write(save / 'run.json', dict(row, status='running'))
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                mask = None
                if case['task'] == 'inpainting':
                    mask_path = y_path.parent / 'mask.pt'
                    require(digest(mask_path) == sources[case['key']]['mask_sha256'], 'Cached mask changed')
                    mask = torch.load(mask_path, map_location='cpu', weights_only=True)
                with clock.counters(model.model) as prefix_count:
                    row['counts']['prefix'] = prefix_count
                    operator = make_operator(case, model.device, count=prefix_count)
                    if mask is not None:
                        require(torch.equal(operator._mask_cache.cpu(), mask), 'A103 mask differs')
                    prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, operator, y, case))
                base.validate(prefix_count, 270, 9)
                require(prefix_count['calls'] == grid * 9, 'Unexpected prefix grid')
                final_rng = None
                outputs = {}
                for sigma in CONTROLLER_NOISES:
                    for arm in ARMS:
                        key = f'{arm}_sigma{sigma:.2f}'
                        with clock.counters(model.model) as count:
                            row['counts'][key] = count
                            operator = make_operator(case, model.device, count=count)
                            if mask is not None:
                                require(torch.equal(operator._mask_cache.cpu(), mask), 'A103 mask differs')
                            controlled = dict(case, sigma=sigma)
                            (output, detail), seconds = clock.clocks(torch.cuda.synchronize,
                                lambda: run_suffix(model, operator, y, controlled, arm, prefix))
                        base.validate(count, 30, 1)
                        require(count['calls'] == grid, 'Unexpected suffix grid')
                        if final_rng is None:
                            final_rng = detail['final_rng']
                        else:
                            require(base.rng_equal(final_rng, detail['final_rng']), 'Unpaired suffix RNG')
                        if sigma == FROZEN_NOISE:
                            path = artifact(folder, old, arm, arm + '.pt')
                            cached = torch.load(path, map_location='cpu', weights_only=True).double()
                            error = float((output.double()-cached).norm()/cached.norm().clamp_min(1e-30))
                            row.setdefault('replay_relative_L2', {})[arm] = error
                            require(error <= 1e-6, f'A103 replay differs for {case["key"]}/{arm}: {error:.3g}')
                        outputs[key] = output
                        row['arms'][key] = dict(sigma_controller=sigma, suffix_seconds=seconds,
                            method_seconds=prefix_seconds+seconds,
                            omega_zero=sum(r['scalar']['weight'] == 0 for r in detail['trace']))
                        write(save / (key+'_trace.json'), detail['trace'])
                gt = load_image_as_tensor(case['image']).cpu()
                for key, output in outputs.items():
                    row['arms'][key]['metrics'] = metrics(output, gt)
                    if mask is not None:
                        row['arms'][key]['regional_errors'] = region_errors(output, gt, mask)
                    torch.save(output, save / (key + '.pt'))
                row['zaps300_metrics'] = old['arms'][BASE]['metrics']
                write(save / 'run.json', dict(row, status='complete'))
                report['completed_cases'] += 1
                write(root / 'summary.json', report)
                print('A105 COMPLETE:', case['key'], row['arms'], flush=True)
        report['comparisons'] = []
        for task in TASKS:
            rows = [row for row in report['cases'] if row['case']['task'] == task]
            for arm in ARMS:
                paired = [(row['arms'][arm+'_sigma0.10']['metrics'],
                           row['arms'][arm+'_sigma0.05']['metrics']) for row in rows]
                delta = {key: mean(fixed[key]-true[key] for true, fixed in paired) for key in METRICS}
                wins = sum(fixed['psnr'] > true['psnr'] for true, fixed in paired)
                losses = sum(fixed['psnr'] < true['psnr'] for true, fixed in paired)
                direction = ('FIXED_BETTER' if delta['psnr'] >= .10 and wins >= 2 and
                             delta['ssim'] >= 0 and delta['lpips'] <= 0 else
                             'TRUE_BETTER' if delta['psnr'] <= -.10 and losses >= 2 and
                             delta['ssim'] <= 0 and delta['lpips'] >= 0 else 'MIXED_OR_SMALL')
                report['comparisons'].append(dict(task=task, arm=arm, fixed_minus_true=delta,
                                                   psnr_wins=wins, images=len(paired), decision=direction))
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if report['cases'] and report['completed_cases'] < len(report['cases']):
            row = report['cases'][-1]
            write(root / 'cases' / row['case']['key'] / 'run.json',
                  dict(row, status='failed', error=repr(exc)))
        raise
    finally:
        report['recorded_new_nfe_lower_bound'] = sum(
            count.get('forward_attempts', 0) for row in report['cases'] for count in row['counts'].values())
        report['cost_is_lower_bound'] = report['status'] != 'complete'
        write(root / 'summary.json', report)
    print('A105 FIXED MINUS TRUE:', json.dumps(report['comparisons'], ensure_ascii=False), flush=True)
    print('A105 summary:', root / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('a104-status').add_argument('--root', type=Path, default=A104)
    audit = commands.add_parser('cached-audit')
    audit.add_argument('--parent', type=Path, default=A103)
    audit.add_argument('--report', type=Path)
    ablation = commands.add_parser('sigma-ablation')
    ablation.add_argument('--parent', type=Path, default=A103)
    ablation.add_argument('--root', type=Path, default=A105)
    args = parser.parse_args()
    if args.command == 'a104-status':
        a104_status(args.root)
    elif args.command == 'cached-audit':
        cached_audit(args.parent, args.report)
    else:
        sigma_ablation(args.parent, args.root)


if __name__ == '__main__':
    main()
