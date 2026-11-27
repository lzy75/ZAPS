"""Add the frozen A146 temporal increment to the full A149 FFHQ panel."""
import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from pathlib import Path
import statistics
from unittest.mock import patch

from utils import diag_a149_physical_diagonal_confirmation as parent
from utils import diag_a146_guarded_panel as guarded
from utils.a54_session import exclusive

panel, rules = parent.panel, guarded.rules
files, clock, native, zaps = parent.files, parent.clock, parent.native, parent.zaps
history, shared, audit, bridge = parent.history, parent.shared, parent.audit, parent.bridge
CELLS, METRICS = parent.CELLS, parent.METRICS
ROOT = parent.ROOT.with_name('diag_a150_guarded_temporal_confirmation')
ARM = 'TEMPORAL'
STRENGTH = 1.


def parent_records(host, task, source):
    plan, summary = source/'resolved_config.json', source/'summary.json'
    config, report = files.read(plan), files.read(summary)
    grid, active, warm = panel.schedule(host)
    files.require(config['experiment'] == report['experiment'] == 'A149' and
                  config['host'] == report['host'] == host and config['task'] == report['task'] == task and
                  report['status'] == 'complete' and report['completed_cases'] == len(parent.IDS) and
                  tuple(c['image_id'] for c in config['cases']) == parent.IDS and
                  config['grid'] == grid and config['positions'] == (grid if host == 'zaps' else list(active)),
                  'A150 requires a complete compatible A149 cell')
    checkpoint = config['checkpoint']
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'], 'FFHQ checkpoint changed')
    for key in ('source_sha256', 'code_sha256'):
        for path, sha in config[key].items():
            files.require(files.digest(path) == sha, f'A149 frozen source changed: {path}')
    archived = {r['case']['key']: r for r in report['cases']}
    sources = {str(p): files.digest(p) for p in (plan, summary)}
    cases = []
    for case in config['cases']:
        folder = source/'cases'/case['key']
        row = files.read(folder/'run.json')
        files.require(row == archived[case['key']] and row['case'] == case and row['status'] == 'complete' and
                      set(row['arms']) == set(parent.ARMS) and case['sigma'] == .05 and
                      files.digest(case['image']) == case['image_sha256'],
                      'A149 case fingerprint or image differs')
        expected_vjp = 0 if host == 'zaps' else len(grid)
        for arm in row['arms'].values():
            actual = arm['method_nfe'], arm['input_vjp'], arm['optimizer_backward']
            expected = (300, 0, 10) if host == 'zaps' else (len(grid), expected_vjp, 0)
            files.require(actual == expected, 'A149 reference budget differs')
        required = {'observation.pt', 'panel.png'} if case['image_id'] in parent.PANELS else {'observation.pt'}
        files.require(required <= row['artifacts'].keys(), f'Incomplete A149 artifacts at {folder}')
        sources[str(folder/'run.json')] = files.digest(folder/'run.json')
        for name, sha in row['artifacts'].items():
            path = folder/name
            files.require(files.digest(path) == sha, f'A149 artifact changed: {path}')
            sources[str(path)] = sha
        cases.append((folder, row))
    return config, cases, sources, grid, active, warm


def zaps_hook(geometry, wave, sigma, active, warm, count, trace):
    from utils.shrink_multitask_runtime import informative

    remember, refine = rules.controller(geometry, wave, 'GUARDED', sigma, count, trace, strength=STRENGTH)
    original = ('            delta, detail, module = repaired.exact_diagonal_with_native_straight_through(\n'
                '                zaps, zaps.A, clean, q, residual, ab[t], rules.SIGMA, module)')
    replacement = (original+'\n'
                   '            if t in warm: remember(clean)\n'
                   '            if t in active:\n'
                   '                with torch.no_grad():\n'
                   '                    altered = refine(q.detach(), informative(residual, zaps.A), float(ab[t]),\n'
                   "                                     detail['scalar'], t, clean.detach(), delta.detach())\n"
                   '                delta = altered + (delta - delta.detach())')
    return audit.insert_probe(zaps._custom_suffix, [(original, replacement)], active=set(active), warm=set(warm),
                              remember=remember, refine=refine, informative=informative)


def zaps_run(config, case, source, model, wave, kernel, grid, active, warm, trace):
    import torch

    device = model.device
    with clock.counters(model.model) as n:
        def train():
            y, op = parent.checked_observation(source, config, case, device, kernel, n)
            return zaps.train_prefix(model, op, y, case)
        prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize, train)
    panel.count_budget(n, 270, 9, 0, grid*9)
    prefix_counts = n

    def suffix(temporal):
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0, graph_actions=0)

            def sample():
                y, op = parent.checked_observation(source, config, case, device, kernel, counts)
                context = nullcontext()
                if temporal:
                    zero = torch.zeros((1, 3, 256, 256), device=device)
                    geom = panel.method.Geometry(case['task'], op, zero, counts)
                    hook = zaps_hook(geom, wave, case['sigma'], active, warm, counts, trace)
                    context = patch.object(zaps, '_custom_suffix', hook)
                with context:
                    return zaps.run_suffix(model, op, y, case, parent.ZAPS_SUFFIXES['DIAGONAL'], prefix)

            (out, detail), seconds = clock.clocks(torch.cuda.synchronize, sample)
        panel.count_budget(counts, 30, 1, 0, grid)
        files.require([p['t'] for p in detail['trace']] == grid, 'ZAPS physical coverage differs')
        for name, value in detail['module_counts'].items():
            if name in counts:
                counts[name] += value
        return out, detail['final_rng'], seconds, counts

    replay, before_rng, _, replay_counts = suffix(False)
    output, after_rng, seconds, counts = suffix(True)
    files.require(parent.rng_equal(before_rng, after_rng), 'A150 ZAPS suffix RNG differs')
    phases = dict(prefix=prefix_counts, reference_replay=replay_counts, temporal_suffix=counts)
    return output, prefix_seconds+seconds, phases, replay


def native_run(config, case, source, model, wave, kernel, grid, active, warm, trace, reference):
    import torch
    from utils.diag_pigdm_noisy_r1 import fingerprint

    host, device = config['host'], model.device
    first, physical_steps = [], []
    with clock.counters(model.model) as counts:
        native.initialize_counts(counts)
        counts.update(graph_builds=0, graph_actions=0)

        def at_entry(_module, args):
            if len(counts['calls']) == len(grid)-len(active)+1:
                first.append(fingerprint(args[0]))

        def sample():
            y, op = parent.checked_observation(source, config, case, device, kernel, counts)
            zero = torch.zeros((1, 3, 256, 256), device=device)
            geom = panel.method.Geometry(case['task'], op, zero, counts)
            physical = panel.temporal.late.terminal.controller('DIAGONAL_K12', wave, counts, physical_steps)
            remember, refine = rules.controller(geom, wave, 'GUARDED', case['sigma'], counts, trace,
                                                 strength=STRENGTH)

            def update(q, residual, alpha, scalar, t, clean):
                delta = physical(geom.view.operator, q, residual, alpha, scalar, t)
                return refine(q, residual, alpha, scalar, t, clean, delta)

            run = panel.native_instrument(history.native({'case': case}, host), host, active, warm, remember, update)
            if host == 'pigdm':
                return run('A150', model, op, y, case, 'SHRINK', counts)
            return run(model, op, y, case, 'SHRINK', config['diffusion_config'], config['dps_root'], counts)

        with model.model.register_forward_pre_hook(at_entry):
            (out, detail), seconds = clock.clocks(torch.cuda.synchronize, sample)
        pairing = history.fingerprint(detail, shared.rng(), counts, host)
    panel.count_budget(counts, len(grid), 0, len(grid), grid)
    files.require(first == [reference['late_entry']] and pairing == reference['pairing'] and
                  detail['grid'] == grid and tuple(row['t'] for row in physical_steps) == active and
                  [r['t'] for r in detail['trace'] if 'weight' in r] == grid,
                  'A150 native pairing, time grid, or physical coverage differs')
    return out, seconds, dict(native=counts), None


def append_panel(source, destination, output):
    from PIL import Image, ImageDraw
    from modules.dataset_loader import tensor_to_image

    with Image.open(source) as prior:
        files.require(prior.size == (1280, 276), 'A149 qualitative panel has unexpected dimensions')
        canvas = Image.new('RGB', (1536, 276), 'white')
        canvas.paste(prior, (0, 0))
    canvas.paste(tensor_to_image(output.squeeze(0), denormalize=True), (1280, 20))
    ImageDraw.Draw(canvas).text((1288, 3), 'Guarded temporal diagonal', fill='black')
    canvas.save(destination)


def measure(folder, config, source, original, model, wave, kernel, grid, active, warm):
    import torch
    from modules.main_single import load_image_as_tensor

    case, host, device = original['case'], config['host'], model.device
    truth = (shared.load_image if host == 'pigdm' else load_image_as_tensor)(case['image'])
    trace = []
    reference = original['arms']['DIAGONAL']
    if host == 'zaps':
        out, seconds, phases, replay = zaps_run(config, case, source, model, wave, kernel, grid, active, warm, trace)
    else:
        out, seconds, phases, replay = native_run(config, case, source, model, wave, kernel, grid,
                                                  active, warm, trace, reference)
    files.require(tuple(p['t'] for p in trace) == active, 'A150 temporal schedule differs')
    files.require(bool(torch.isfinite(out).all()), 'Nonfinite temporal output')
    eval_counts = {}
    native.initialize_counts(eval_counts)
    with torch.no_grad():
        y, op = parent.checked_observation(source, config, case, device, kernel)
        geom = panel.method.Geometry(case['task'], op, truth.to(device), eval_counts)
        gate = geom.check(truth.to(device))
        if geom.view.operator.selected is not None:
            y = y[geom.view.operator.selected]
    metrics = panel.evaluate_on_device(out, truth, y, geom, device)
    replay_check = None
    if replay is not None:
        current = panel.evaluate_on_device(replay, truth, y, geom, device)
        replay_check = {k: abs(current['quality'][k]-reference['quality'][k]) for k in METRICS}
        replay_check['raw_mse'] = abs(current['raw_mse']-reference['raw_mse'])
        tolerances = dict(psnr=.002, ssim=.0002, lpips=.0002, raw_mse=2e-6)
        files.require(all(replay_check[k] <= tol for k, tol in tolerances.items()),
                      f'A149 ZAPS physical replay differs: {replay_check}')
    files.write(folder/'trace.json', trace)
    torch.save(out, folder/'output.pt')
    artifacts = {name: files.digest(folder/name) for name in ('trace.json', 'output.pt')}
    if 'panel.png' in original['artifacts']:
        append_panel(source/'panel.png', folder/'panel.png', out)
        artifacts['panel.png'] = files.digest(folder/'panel.png')
    actual = {k: sum(phase.get(k, 0) for phase in phases.values()) for k in panel.COSTS}
    result = dict(method_nfe=reference['method_nfe'], input_vjp=reference['input_vjp'],
                  optimizer_backward=reference['optimizer_backward'], full_seconds=seconds,
                  guard=guarded.record_trace(trace), **metrics)
    row = dict(status='complete', case=case, arms={**original['arms'], ARM: result},
               geometry=gate, physical_replay=replay_check, phases=phases, actual_cost=actual,
               source_run_sha256=files.digest(source/'run.json'), artifacts=artifacts)
    print('A150 COMPLETE', host, case['key'], result['quality'], result['guard'], flush=True)
    return row


def summarize(cases):
    group = parent.summarize(cases)
    arms = [row['arms'][ARM] for row in cases]
    group['arms'][ARM] = dict(quality={k: statistics.mean(arm['quality'][k] for arm in arms) for k in METRICS},
                              seconds=statistics.mean(arm['full_seconds'] for arm in arms),
                              nfe=arms[0]['method_nfe'], input_vjp=arms[0]['input_vjp'],
                              optimizer_backward=arms[0]['optimizer_backward'],
                              clean_observation_mse=statistics.mean(arm['clean_observation_mse'] for arm in arms),
                              noisy_observation_mse=statistics.mean(arm['noisy_observation_mse'] for arm in arms),
                              mean_beta=statistics.mean(arm['guard']['mean_beta'] for arm in arms),
                              timing_source='A150')
    for reference in parent.ARMS:
        group['arms'][reference]['timing_source'] = 'A149'
        pair = bridge.paired(cases, ARM, reference)
        d = pair['delta']
        pair['passes'] = (len(cases) == len(parent.IDS) and pair['psnr_wins'] >= 7 and
                          d['psnr'] >= .10 and d['ssim'] >= 0 and d['lpips'] <= 0)
        group['paired'].append(pair)
    return group


def execute(host, task, root, source):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),
                  'A150 output overlaps A149')
    config, originals, input_hashes, grid, active, warm = parent_records(host, task, source)
    card = panel.parent.transfer.PROJECTS/'experiment_cards/A150_guarded_temporal_confirmation.md'
    sources = (Path(__file__), Path(rules.__file__), Path(panel.method.__file__),
               Path(panel.temporal.graph.__file__), card)
    plan = dict(experiment='A150', parent=str(source), host=host, task=task, strength=STRENGTH,
                active=list(active), warm=list(warm), input_sha256=input_hashes,
                code_sha256={str(p): files.digest(p) for p in sources})
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/'worker.lock'):
        frozen, path = root/'resolved_config.json', root/'summary.json'
        if frozen.is_file():
            files.require(files.read(frozen) == plan, 'Frozen A150 config differs')
        else:
            files.write(frozen, plan)
        if path.is_file() and files.read(path)['status'] == 'complete':
            print('A150 SUMMARY', host, task, files.read(path)['group'], flush=True)
            return
        report = dict(experiment='A150', host=host, task=task, status='running', cases=[],
                      completed_cases=0, actual_cost=dict.fromkeys(panel.COSTS, 0), cost_is_lower_bound=False)
        files.write(path, report)
        try:
            report['resource'] = panel.admission.check(torch, root)
            model = load_diffusion_model('ffhq', 'cuda:0')
            model.model.load_state_dict(torch.load(config['checkpoint']['path'], map_location='cpu',
                                                   weights_only=True), strict=True)
            model.model.eval()
            wave = OrthogonalDWT2D(wave='db4', level=3).to(model.device)
            kernel = None if config['kernel'] is None else np.load(config['kernel'], allow_pickle=False)
            scope = native.dps_reference_scope() if host == 'dps' else nullcontext()
            derivative = input_vjp_parameters(model.model) if host != 'zaps' else nullcontext()
            with precision('LEGACY_IEEE'), frozen_network(model.model), derivative, scope:
                for directory, original in originals:
                    case = original['case']
                    folder = root/'cases'/case['key']
                    status = folder/'run.json'
                    if status.is_file():
                        row = files.read(status)
                        files.require(row['status'] == 'complete' and row['case'] == case and
                                      row['source_run_sha256'] == input_hashes[str(directory/'run.json')],
                                      f'Incomplete or changed case retained at {folder}')
                        for name, sha in row['artifacts'].items():
                            files.require(files.digest(folder/name) == sha, f'A150 artifact changed: {name}')
                    else:
                        folder.mkdir(parents=True)
                        files.write(status, dict(status='running', case=case))
                        row = measure(folder, config, directory, original, model, wave, kernel, grid, active, warm)
                        files.write(status, row)
                    report['cases'].append(row)
                    report['completed_cases'] += 1
                    for key, count in row['actual_cost'].items():
                        report['actual_cost'][key] += count
                    files.write(path, report)
            expected = len(parent.IDS)*(330 if host == 'zaps' else len(grid))
            files.require(report['actual_cost']['forwards'] == expected, 'A150 total cost differs')
            report.update(status='complete', group=summarize(report['cases']))
        except BaseException as exc:
            report.update(status='failed', error=repr(exc), cost_is_lower_bound=True)
            raise
        finally:
            files.write(path, report)
    print('A150 SUMMARY', host, task, report['group'], flush=True)


def matrix(root):
    root.mkdir(parents=True, exist_ok=True)
    rows, quality = [], []
    for host, task in CELLS:
        path = root/f'{host}_{task}'/'summary.json'
        data = files.read(path) if path.is_file() else dict(status='missing')
        row = dict(host=host, task=task, status=data['status'], completed_cases=data.get('completed_cases', 0))
        if data['status'] == 'complete':
            row.update(data['group'], actual_cost=data['actual_cost'])
            for method, arm in row['arms'].items():
                quality.append((host, task, method, row['images'], *(arm['quality'][m] for m in METRICS),
                                arm['nfe'], arm['input_vjp'], arm['optimizer_backward'],
                                arm['seconds'], arm['timing_source']))
        if data.get('error'):
            row['error'] = data['error']
        rows.append(row)
        print('A150 MATRIX', row, flush=True)
    files.write(root/'matrix.json', rows)
    with (root/'quality.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('host', 'task', 'method', 'images', *METRICS, 'nfe', 'input_vjp',
                         'optimizer_backward', 'seconds', 'timing_source'))
        writer.writerows(quality)
    complete = sum(r['status'] == 'complete' for r in rows)
    print('A150 REPORT', root/'matrix.json', f'{complete}/{len(CELLS)}', root/'quality.csv', flush=True)
    return complete == len(CELLS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=(*panel.HOSTS, 'matrix'))
    parser.add_argument('task', choices=panel.TASKS, nargs='?')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--parent', type=Path, default=parent.ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.host == 'matrix':
        raise SystemExit(0 if matrix(root) else 1)
    if args.task is None:
        parser.error('a task is required')
    cell = f'{args.host}_{args.task}'
    execute(args.host, args.task, root/cell, args.parent.resolve()/cell)


if __name__ == '__main__':
    main()
