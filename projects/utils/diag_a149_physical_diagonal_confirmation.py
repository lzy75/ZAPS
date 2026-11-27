"""Confirm the frozen physical diagonal at equal NFE on all FFHQ cells."""
import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from pathlib import Path
import statistics
import time

from utils import diag_a145_temporal_panel as panel
from utils.a54_session import exclusive
from utils.diag_shared_prefix_feedback import rng_equal

files, native, history, shared, zaps = panel.files, panel.native, panel.history, panel.shared, panel.zaps
clock, audit, bridge = panel.clock, panel.audit, panel.bridge
ROOT = panel.ROOT.with_name('diag_a149_physical_diagonal_confirmation')
CELLS = tuple((host, task) for host in panel.HOSTS for task in panel.TASKS)
IDS = tuple(f'{n:05d}' for n in range(10, 20))
PANELS = {'00010', '00014', '00018'}
ARMS = ('BASE', 'SCALAR', 'DIAGONAL')
ZAPS_SUFFIXES = dict(BASE='ZAPS_SUFFIX', **panel.ZAPS_ARMS)
METRICS = ('psnr', 'ssim', 'lpips')


def prepare(host, task, image_dir, parent):
    plan_path = parent/'plan.json'
    a102 = files.read(plan_path)
    files.require(a102['experiment'] == zaps.rules.EXPERIMENT, 'Wrong A102 source')
    checkpoint = a102['checkpoint']
    inputs = {plan_path}
    settings = {}
    excluded = dict((Path(c['image']), c['image_sha256']) for c in a102['cases'])
    if host != 'zaps':
        old = history.load_sources(host, 'ffhq')
        for entry in old:
            c, plan = entry['case'], entry['plan']
            excluded[Path(c['image'])] = plan['inputs'][c['image']]
        same_task = next(entry for entry in old if entry['case']['task'] == task)
        native_plan = same_task['plan']
        files.require(native_plan['checkpoint']['sha256'] == checkpoint['sha256'],
                      'Model checkpoint differs across sources')
        inputs.add(same_task['root']/'plan.json')
        if host == 'dps':
            settings.update({k: native_plan[k] for k in ('diffusion_config', 'dps_root')})
            kernel_root = Path(native_plan.get('operator_root', native_plan['parent']))
            kernel_source = files.read(kernel_root/'plan.json')
            inputs.add(kernel_root/'plan.json')
        else:
            kernel_root, kernel_source = same_task['root'], native_plan
    else:
        kernel_root, kernel_source = parent, a102
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint changed')
    kernel = kernel_root/'motion_kernel.npy' if task == 'motion_deblur' else None
    if kernel is not None:
        files.require(files.digest(kernel) == kernel_source['motion_kernel_sha256'], 'Frozen kernel changed')
        inputs.add(kernel)
    development = set(excluded.values())
    cases = []
    for image_id in IDS:
        number = int(image_id)
        image = image_dir/f'{image_id}.png'
        sha = files.digest(image)
        files.require(sha not in development, 'Confirmation image bytes overlap with the source panel')
        files.require(all(sha != c['image_sha256'] for c in cases), 'Duplicate confirmation image bytes')
        seed = 1000+number
        cases.append(dict(image_id=image_id, image=str(image), image_sha256=sha, task=task,
                          key=f'{task}_{image_id}_n050_s{seed}', sigma=.05, seed=seed,
                          measurement_seed=(2000 if host == 'pigdm' else 1000)+number,
                          kernel_seed=3000+number, mask_seed=2001000+number))
    grid, active, _ = panel.schedule(host)
    return dict(experiment='A149', host=host, task=task, checkpoint=checkpoint, cases=cases,
                grid=grid, positions=grid if host == 'zaps' else list(active), arms=list(ARMS),
                kernel=None if kernel is None else str(kernel),
                source_sha256={str(p): files.digest(p) for p in sorted(inputs)}, **settings)


def operator(host, case, device, kernel, count=None):
    if host == 'pigdm':
        from utils.pigdm_noisy_runtime import Operator
        return Operator(case['task'], device, seed=case['kernel_seed'], kernel=kernel, count=count)
    return zaps.make_operator(case, device, count=count, kernel=kernel)


def observation(folder, host, case, truth, device, kernel):
    import torch

    with torch.no_grad():
        op = operator(host, case, device, kernel)
        expected = op.H(truth.to(device))
        shared.seed(case['measurement_seed'])
        y = expected + case['sigma']*torch.randn_like(expected)
        mask = op.mask if host == 'pigdm' else getattr(op, '_mask_cache', None)
        torch.save(dict(y=y.cpu(), mask=None if mask is None else mask.cpu()), folder/'observation.pt')
    return files.digest(folder/'observation.pt')


def checked_observation(folder, config, case, device, kernel, count=None):
    import torch

    payload = torch.load(folder/'observation.pt', map_location=device, weights_only=True)
    op = operator(config['host'], case, device, kernel, count)
    mask = op.mask if config['host'] == 'pigdm' else getattr(op, '_mask_cache', None)
    files.require((mask is None and payload['mask'] is None) or
                  (mask is not None and payload['mask'] is not None and torch.equal(mask, payload['mask'])),
                  'Paired measurement mask changed')
    return payload['y'], op


def native_run(config, case, arm, model, wave, y, op, counts, trace):
    host = config['host']
    runtime = history.native({'case': case}, host)
    run = runtime.run
    if arm == 'DIAGONAL':
        physical = panel.parent.ObservedResponse(op)
        solve = panel.temporal.late.terminal.controller('DIAGONAL_K12', wave, counts, trace)

        def refine(q, residual, alpha, scalar, t, _clean):
            return solve(physical, q, residual, alpha, scalar, t)

        run = panel.native_instrument(runtime, host, config['positions'], (), None, refine)
    if host == 'pigdm':
        return run('A149', model, op, y, case, 'BASE' if arm == 'BASE' else 'SHRINK', counts)
    return run(model, op, y, case, 'DPS' if arm == 'BASE' else 'SHRINK',
               config['diffusion_config'], config['dps_root'], counts)


def save_panel(folder, truth, outputs, y, op, host):
    from PIL import Image, ImageDraw
    from modules.dataset_loader import tensor_to_image

    mask = getattr(op, '_mask_cache', None)
    display = y.cpu()
    if host == 'pigdm' and op.mask is not None:
        display = op.Ht(y).cpu()
        mask = op.mask.cpu()
    if mask is not None:
        mask = mask.cpu().bool()
        display = display.where(mask, -1.)
    labels = ('Ground truth', 'Observation', 'Baseline', 'Scalar', 'Physical diagonal')
    canvas = Image.new('RGB', (5*256, 276), 'white')
    for index, (label, tensor) in enumerate(zip(labels, (truth, display, *(outputs[arm] for arm in ARMS)))):
        image = tensor_to_image(tensor.squeeze(0), denormalize=True)
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.BICUBIC)
        canvas.paste(image, (index*256, 20))
        ImageDraw.Draw(canvas).text((index*256+8, 3), label, fill='black')
    canvas.save(folder/'panel.png')
    return files.digest(folder/'panel.png')


def run_case(folder, config, case, model, wave, kernel, order):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_pigdm_noisy_r1 import fingerprint

    host, device, grid = config['host'], model.device, config['grid']
    truth = (shared.load_image if host == 'pigdm' else load_image_as_tensor)(case['image'])
    actual, arms, outputs = Counter(), {}, {}
    observation_sha = observation(folder, host, case, truth, device, kernel)
    prefix, prefix_seconds = None, 0.
    if host == 'zaps':
        with clock.counters(model.model) as counts:
            def train():
                y, op = checked_observation(folder, config, case, device, kernel, counts)
                return zaps.train_prefix(model, op, y, case)
            prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize, train)
        panel.count_budget(counts, 270, 9, 0, grid*9)
        actual.update({k: counts.get(k, 0) for k in panel.COSTS})
    final_rng = None
    y_eval, op_eval = checked_observation(folder, config, case, device, kernel)
    y_clean = op_eval.H(truth.to(device))
    mask = getattr(op_eval, '_mask_cache', None)
    for arm in order:
        first, trace = [], []
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)

            def at_entry(_network, args):
                if len(counts['calls']) == len(grid)-len(config['positions'])+1:
                    first.append(fingerprint(args[0]))

            def sample():
                y, op = checked_observation(folder, config, case, device, kernel, counts)
                if host == 'zaps':
                    return zaps.run_suffix(model, op, y, case, ZAPS_SUFFIXES[arm], prefix)
                return native_run(config, case, arm, model, wave, y, op, counts, trace)

            hook = model.model.register_forward_pre_hook(at_entry) if host != 'zaps' else nullcontext()
            with hook:
                (out, detail), seconds = clock.clocks(torch.cuda.synchronize, sample)
            rng = shared.rng() if host != 'zaps' else detail['final_rng']
        nfe, vjp = (300, 0) if host == 'zaps' else (len(grid), len(grid))
        panel.count_budget(counts, len(grid), int(host == 'zaps'), vjp, grid)
        if host == 'zaps':
            if final_rng is not None:
                files.require(rng_equal(final_rng, rng), 'ZAPS suffix RNG differs')
            final_rng = rng
            files.require([p['t'] for p in detail['trace']] == ([] if arm == 'BASE' else grid),
                          'ZAPS suffix correction coverage differs')
        else:
            files.require(len(first) == 1 and detail['grid'] == grid and
                          [p['t'] for p in trace] == (config['positions'] if arm == 'DIAGONAL' else []) and
                          [p['t'] for p in detail['trace'] if 'weight' in p] == ([] if arm == 'BASE' else grid),
                          'Native grid or correction coverage differs')
        files.require(bool(torch.isfinite(out).all()), 'Nonfinite reconstruction')
        with torch.no_grad():
            predicted = op_eval.H(out.to(device))
            if mask is not None:
                selection = mask.bool().expand_as(y_eval)
                predicted, clean, noisy = predicted[selection], y_clean[selection], y_eval[selection]
            else:
                clean, noisy = y_clean, y_eval
            scores = dict(clean_observation_mse=audit.mse(predicted-clean),
                          noisy_observation_mse=audit.mse(predicted-noisy))
        metrics = bridge.output_metrics(out, truth)
        info = dict(quality=metrics['quality'], raw_mse=metrics['raw_mse'], **scores,
                    full_seconds=prefix_seconds+seconds, method_nfe=nfe, input_vjp=vjp,
                    optimizer_backward=10 if host == 'zaps' else 0, counts=counts)
        if host == 'zaps':
            for name, value in detail['module_counts'].items():
                if name in counts:
                    counts[name] += value
        else:
            info.update(pairing=history.fingerprint(detail, rng, counts, host), late_entry=first[0])
        if trace:
            files.write(folder/(arm+'_trace.json'), trace)
            info['physical_updates'] = sum(p['updates'] for p in trace)
        if case['image_id'] in PANELS:
            outputs[arm] = out
        arms[arm] = info
        actual.update({k: counts.get(k, 0) for k in panel.COSTS})
        print('A149 COMPLETE', host, case['key'], arm, metrics['quality'], flush=True)
    if host != 'zaps':
        files.require(arms['BASE']['pairing'] == arms['SCALAR']['pairing'] == arms['DIAGONAL']['pairing'] and
                      arms['SCALAR']['late_entry'] == arms['DIAGONAL']['late_entry'], 'A149 native pairing differs')
    artifacts = {'observation.pt': observation_sha}
    if outputs:
        artifacts['panel.png'] = save_panel(folder, truth, outputs, y_eval, op_eval, host)
    return dict(status='complete', case=case, arms=arms, actual_cost=dict(actual), artifacts=artifacts)


def summarize(cases):
    arms = {}
    for arm in ARMS:
        rows = [row['arms'][arm] for row in cases]
        arms[arm] = dict(quality={m: statistics.mean(v['quality'][m] for v in rows) for m in METRICS},
                         seconds=statistics.mean(v['full_seconds'] for v in rows),
                         clean_observation_mse=statistics.mean(v['clean_observation_mse'] for v in rows),
                         noisy_observation_mse=statistics.mean(v['noisy_observation_mse'] for v in rows),
                         nfe=rows[0]['method_nfe'], input_vjp=rows[0]['input_vjp'],
                         optimizer_backward=rows[0]['optimizer_backward'])
    pairs = []
    for candidate, reference in (('SCALAR', 'BASE'), ('DIAGONAL', 'SCALAR'), ('DIAGONAL', 'BASE')):
        result = bridge.paired(cases, candidate, reference)
        d = result['delta']
        result['passes'] = (len(cases) == len(IDS) and result['psnr_wins'] >= 7 and
                            d['psnr'] >= .10 and d['ssim'] >= 0 and d['lpips'] <= 0)
        pairs.append(result)
    return dict(images=len(cases), arms=arms, paired=pairs)


def execute(host, task, root, image_dir, parent):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    files.require(not root.is_relative_to(parent) and not root.is_relative_to(image_dir),
                  'A149 output overlaps a parent source')
    config = prepare(host, task, image_dir, parent)
    sources = (Path(__file__), Path(panel.__file__), Path(zaps.__file__),
               Path(panel.temporal.late.terminal.__file__), Path(panel.temporal.late.depth.__file__),
               Path(history.native({'case': {'task': task}}, host).__file__) if host != 'zaps' else
               Path(zaps.repaired.__file__),
               panel.parent.transfer.PROJECTS/'experiment_cards/A149_physical_diagonal_confirmation.md')
    config['code_sha256'] = {str(p): files.digest(p) for p in sources}
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/'worker.lock'):
        plan, summary = root/'resolved_config.json', root/'summary.json'
        if plan.is_file():
            files.require(files.read(plan) == config, 'A149 frozen configuration differs')
        else:
            files.write(plan, config)
        if summary.is_file() and files.read(summary)['status'] == 'complete':
            print('A149 SUMMARY', host, task, files.read(summary)['group'], flush=True)
            return
        report = dict(status='running', experiment='A149', host=host, task=task,
                      planned_cases=len(IDS), completed_cases=0, actual_cost=dict.fromkeys(panel.COSTS, 0),
                      cost_is_lower_bound=False, cases=[])
        files.write(summary, report)
        try:
            report['resource'] = panel.admission.check(torch, root)
            started = time.perf_counter()
            model = load_diffusion_model('ffhq', 'cuda:0')
            model.model.load_state_dict(torch.load(config['checkpoint']['path'], map_location='cpu',
                                                   weights_only=True), strict=True)
            model.model.eval()
            wave = OrthogonalDWT2D(wave='db4', level=3).to(model.device) if host != 'zaps' else None
            kernel = None if config['kernel'] is None else np.load(config['kernel'], allow_pickle=False)
            torch.cuda.synchronize()
            report['model_load_seconds'] = time.perf_counter()-started
            scope = native.dps_reference_scope() if host == 'dps' else nullcontext()
            derivative = input_vjp_parameters(model.model) if host != 'zaps' else nullcontext()
            with precision('LEGACY_IEEE'), frozen_network(model.model), derivative, scope:
                for i, case in enumerate(config['cases']):
                    folder = root/'cases'/case['key']
                    path = folder/'run.json'
                    if path.is_file():
                        row = files.read(path)
                        files.require(row['status'] == 'complete' and row['case'] == case,
                                      f'Incomplete case retained at {folder}')
                        for name, sha in row['artifacts'].items():
                            files.require(files.digest(folder/name) == sha, f'Output changed in {folder}')
                    else:
                        folder.mkdir(parents=True)
                        files.write(path, dict(status='running', case=case))
                        order = ARMS[i % len(ARMS):]+ARMS[:i % len(ARMS)]
                        row = run_case(folder, config, case, model, wave, kernel, order)
                        files.write(path, row)
                    report['cases'].append(row)
                    report['completed_cases'] += 1
                    for key, count in row['actual_cost'].items():
                        report['actual_cost'][key] += count
                    files.write(summary, report)
            per_case = 360 if host == 'zaps' else 3*len(config['grid'])
            files.require(report['actual_cost']['forwards'] == len(IDS)*per_case, 'A149 total NFE differs')
            report.update(status='complete', group=summarize(report['cases']))
        except BaseException as exc:
            report.update(status='failed', error=repr(exc), cost_is_lower_bound=True)
            raise
        finally:
            files.write(summary, report)
    print('A149 SUMMARY', host, task, report['group'], flush=True)


def matrix(root):
    root.mkdir(parents=True, exist_ok=True)
    matrix_rows = []
    with (root/'quality.csv').open('w', newline='') as stream:
        csv_writer = csv.writer(stream)
        csv_writer.writerow(('host', 'task', 'method', 'images', 'psnr', 'ssim', 'lpips', 'nfe',
                             'input_vjp', 'optimizer_backward', 'seconds'))
        for host, task in CELLS:
            source = root/f'{host}_{task}'/'summary.json'
            item = files.read(source) if source.is_file() else dict(status='missing')
            row = dict(host=host, task=task, status=item['status'], source=str(source),
                       completed_cases=item.get('completed_cases', 0))
            if item['status'] == 'complete':
                row.update(item['group'], actual_cost=item['actual_cost'])
                for arm, a in row['arms'].items():
                    csv_writer.writerow((host, task, arm, row['images'], *(a['quality'][k] for k in METRICS),
                                         a['nfe'], a['input_vjp'], a['optimizer_backward'], a['seconds']))
            elif item.get('error'):
                row['error'] = item['error']
            matrix_rows.append(row)
            print('A149 MATRIX', row, flush=True)
    files.write(root/'matrix.json', matrix_rows)
    complete = sum(row['status'] == 'complete' for row in matrix_rows)
    print('A149 REPORT', root/'matrix.json', f'{complete}/{len(CELLS)}', root/'quality.csv', flush=True)
    return complete == len(CELLS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=(*panel.HOSTS, 'matrix'))
    parser.add_argument('task', choices=panel.TASKS, nargs='?')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--images', type=Path, default=Path('/home/lzy/FFHQ/00000/00000'))
    parser.add_argument('--zaps-parent', type=Path, default=Path(zaps.rules.ROOT))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.host == 'matrix':
        raise SystemExit(0 if matrix(root) else 1)
    if args.task is None:
        parser.error('a task is required')
    execute(args.host, args.task, root/f'{args.host}_{args.task}', args.images.resolve(),
            args.zaps_parent.resolve())


if __name__ == '__main__':
    main()
