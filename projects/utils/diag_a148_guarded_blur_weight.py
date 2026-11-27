"""Sweep a shared temporal strength on the six frozen A145 blur cells."""
import argparse
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import statistics
import time

from utils import diag_a146_guarded_panel as guarded

panel, rules = guarded.panel, guarded.rules
files, audit, clock = guarded.files, guarded.audit, guarded.clock
history, native, shared, bridge = guarded.history, guarded.native, guarded.shared, guarded.bridge
ROOT = panel.ROOT.with_name('diag_a148_guarded_blur_weight')
TASKS = ('gaussian_deblur', 'motion_deblur')
CELLS = tuple((host, task) for host in panel.HOSTS for task in TASKS)
STRENGTHS = {'ETA025': .25, 'ETA050': .5, 'ETA100': 1.}
ARMS = (*panel.CONTROLS, *STRENGTHS)


def native_sample(entry, host, model, wave, arm, active, warm, counts, trace, physical_trace):
    import torch

    case, obs, plan = entry['case'], entry['observation'], entry['plan']
    operator = history.operator(entry, obs, host, model.device, counts)
    y = audit.checked_tensor(obs['path'], model.device)
    runtime = history.native(entry, host)
    run = runtime.run
    if arm != 'SCALAR':
        geometry = panel.method.Geometry(
            case['task'], operator, torch.zeros((1, 3, 256, 256), device=model.device), counts)
        solve = panel.temporal.late.terminal.controller('DIAGONAL_K12', wave, counts, physical_trace)
        physical = geometry.view.operator

        def diagonal(q, residual, alpha, scalar, t, _clean):
            return solve(physical, q, residual, alpha, scalar, t)

        if arm == 'DIAGONAL':
            run = panel.native_instrument(runtime, host, active, (), None, diagonal)
        else:
            remember, refine = rules.controller(
                geometry, wave, 'GUARDED', case['sigma'], counts, trace, strength=STRENGTHS[arm])

            def update(q, residual, alpha, scalar, t, clean):
                delta = diagonal(q, residual, alpha, scalar, t, clean)
                return refine(q, residual, alpha, scalar, t, clean, delta)

            run = panel.native_instrument(runtime, host, active, warm, remember, update)
    if host == 'pigdm':
        return run('A148', model, operator, y, case, 'SHRINK', counts)
    return run(model, operator, y, case, 'SHRINK', plan['diffusion_config'], plan['dps_root'], counts)


def native_case(root, entry, host, model, wave, grid, active, warm, index):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_pigdm_noisy_r1 import fingerprint

    case, obs = entry['case'], entry['observation']
    key, device = case['key'], model.device
    truth = (shared.load_image if host == 'pigdm' else load_image_as_tensor)(case['image'])
    y = audit.checked_tensor(obs['path'], device)
    evaluation_counts = {}
    native.initialize_counts(evaluation_counts)
    with torch.no_grad():
        geometry = panel.method.Geometry(
            case['task'], history.operator(entry, obs, host, device, None), truth.to(device),
            evaluation_counts)
        geometry_gate = geometry.check(truth.to(device))
    arms, actual = {}, Counter()
    order = ARMS if index % 2 == 0 else ('SCALAR', *reversed(ARMS[1:]))
    for arm in order:
        first, trace, physical_trace = [], [], []
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0, graph_actions=0)

            def at_entry(_network, args):
                if len(counts['calls']) == len(grid) - len(active) + 1:
                    first.append(fingerprint(args[0]))

            with model.model.register_forward_pre_hook(at_entry):
                (out, detail), seconds = clock.clocks(torch.cuda.synchronize, lambda: native_sample(
                    entry, host, model, wave, arm, active, warm, counts, trace, physical_trace))
            pairing = history.fingerprint(detail, shared.rng(), counts, host)
        panel.count_budget(counts, len(grid), 0, len(grid), grid)
        files.require(len(first) == 1 and detail['grid'] == grid and
                      tuple(point['t'] for point in trace) == (active if arm in STRENGTHS else ()) and
                      tuple(point['t'] for point in physical_trace) == (() if arm == 'SCALAR' else active) and
                      [p['t'] for p in detail['trace'] if 'weight' in p] == grid, 'A148 coverage differs')
        record = dict(status='complete', case=key, arm=arm, late_entry=first[0], pairing=pairing,
                      counts=counts, method_nfe=len(grid), input_vjp=len(grid), full_seconds=seconds,
                      **panel.evaluate_on_device(out, truth, y, geometry, device))
        if arm == 'SCALAR':
            ref, folder = entry['controls']['SCALAR'] if entry['controls'] else entry['legacy_scalar']
            record['archive_relative_l2'] = bridge.archived_error(out, folder, ref, 'output.pt')
        else:
            files.require(pairing == arms['SCALAR']['pairing'] and first[0] == arms['SCALAR']['late_entry'],
                          'A148 initial/late-entry state, RNG or call pairing differs')
        if arm == 'DIAGONAL' and arm in entry['controls']:
            ref, folder = entry['controls'][arm]
            record['archive_relative_l2'] = bridge.archived_error(out, folder, ref, 'output.pt')
        if arm in STRENGTHS:
            record['guard'] = guarded.record_trace(trace)
        if physical_trace:
            record['physical_updates'] = sum(point['updates'] for point in physical_trace)
        panel.save_candidate(root, key, arm, out, record, trace)
        arms[arm] = record
        actual.update({name: counts.get(name, 0) for name in panel.COSTS})
        print('A148 COMPLETE', host, key, arm, record['quality'], f'{seconds:.2f}s', flush=True)
    return dict(case=key, arms=arms, geometry=geometry_gate), dict(actual)


def summarize(cases):
    arms = {}
    for arm in ARMS:
        rows = [case['arms'][arm] for case in cases]
        arms[arm] = dict(
            quality={metric: statistics.mean(row['quality'][metric] for row in rows)
                     for metric in panel.parent.transfer.METRICS},
            regions={metric: statistics.mean(row[metric] for row in rows) for metric in panel.REGIONS},
            seconds=statistics.mean(row['full_seconds'] for row in rows),
            nfe=rows[0]['method_nfe'], input_vjp=rows[0]['input_vjp'])
        if arm in STRENGTHS:
            guards = [row['guard'] for row in rows]
            arms[arm]['guard'] = {name: sum(g[name] for g in guards)
                                  for name in ('points', 'full', 'zero', 'partial', 'physical_up', 'backtracks')}
            arms[arm]['guard']['mean_beta'] = statistics.mean(g['mean_beta'] for g in guards)
    pairs = (('DIAGONAL', 'SCALAR'), *((arm, ref) for arm in STRENGTHS for ref in panel.CONTROLS))
    return dict(images=len(cases), arms=arms, paired=[bridge.paired(cases, a, b) for a, b in pairs])


def execute(host, task, root, source):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    inputs, checkpoint, kernel_path, frozen = (
        panel.zaps_sources(task, source) if host == 'zaps' else panel.native_sources(host, task))
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'], 'Frozen checkpoint changed')
    files.require(all(not p.is_relative_to(root) for p in frozen), 'A148 output overlaps inputs')
    grid, active, warm = panel.schedule(host)
    card = panel.parent.transfer.PROJECTS / 'experiment_cards/A148_guarded_blur_weight.md'
    source_code = (Path(__file__), Path(guarded.__file__), Path(rules.__file__), Path(panel.__file__),
                   Path(panel.method.__file__), Path(panel.temporal.graph.__file__),
                   Path(panel.temporal.late.terminal.__file__), card)
    config = dict(experiment='A148', host=host, task=task, checkpoint=checkpoint,
                  arms=list(ARMS), strengths=STRENGTHS, grid=grid, temporal_positions=list(active),
                  physical_positions=grid if host == 'zaps' else list(active), warm=list(warm),
                  input_sha256={str(p): files.digest(p) for p in frozen},
                  code_sha256={str(p): files.digest(p) for p in source_code})
    path = root / 'summary.json'
    if path.exists():
        previous = files.read(path)
        files.require(previous['status'] == 'complete' and files.read(root/'resolved_config.json') == config,
                      f'A148 incomplete or incompatible result preserved at {root}')
        print('A148 SUMMARY', host, task, previous['group'], flush=True)
        return
    resource = panel.admission.check(torch, root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json', config)
    report = dict(experiment='A148', host=host, task=task, status='running', cases=[], resource=resource,
                  actual_cost=dict.fromkeys(panel.COSTS, 0), cost_is_lower_bound=False)
    files.write(path, report)
    try:
        started = time.perf_counter()
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True), strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4', level=3).to(model.device)
        kernel = None if kernel_path is None else np.load(kernel_path, allow_pickle=False)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter() - started
        scope = native.dps_reference_scope() if host == 'dps' else nullcontext()
        derivative = input_vjp_parameters(model.model) if host != 'zaps' else nullcontext()
        with precision('LEGACY_IEEE'), frozen_network(model.model), derivative, scope:
            for index, entry in enumerate(inputs):
                row, cost = (guarded.zaps_case(root, entry, model, wave, kernel, grid, active, warm, index,
                                               strengths=STRENGTHS) if host == 'zaps' else
                             native_case(root, entry, host, model, wave, grid, active, warm, index))
                report['cases'].append(row)
                for name, actual in cost.items():
                    report['actual_cost'][name] += actual
                files.write(path, report)
        expected = len(inputs) * (270 + 30 * len(ARMS) if host == 'zaps' else len(grid) * len(ARMS))
        files.require(report['actual_cost']['forwards'] == expected, 'A148 total forwards differ')
        report.update(status='complete', group=summarize(report['cases']))
    except BaseException as error:
        report.update(status='failed', error=repr(error), cost_is_lower_bound=True)
        raise
    finally:
        files.write(path, report)
    print('A148 SUMMARY', host, task, report['group'], flush=True)


def matrix(root):
    rows = []
    for host, task in CELLS:
        path = root / f'{host}_{task}' / 'summary.json'
        data = files.read(path) if path.is_file() else {'status': 'missing'}
        row = dict(host=host, task=task, status=data['status'], source=str(path))
        if data['status'] == 'complete':
            row.update(data['group'])
        elif data.get('error'):
            row['error'] = data['error']
        rows.append(row)
        print('A148 MATRIX', row, flush=True)
    complete = all(row['status'] == 'complete' for row in rows)
    files.write(root/'matrix.json', rows)
    if complete:
        evidence = {}
        for arm, strength in STRENGTHS.items():
            pairs = [{p['reference']: p for p in row['paired'] if p['candidate'] == arm} for row in rows]
            nonworse = sum(p['SCALAR']['delta']['psnr'] >= 0 and p['SCALAR']['delta']['ssim'] >= 0 and
                           p['SCALAR']['delta']['lpips'] <= 0 for p in pairs)
            passes_physical = sum(p['DIAGONAL']['passes'] for p in pairs)
            motion = next(p['delta'] for row in rows
                          if (row['host'], row['task']) == ('zaps', 'motion_deblur')
                          for p in row['paired'] if p['candidate'] == arm and p['reference'] == 'DIAGONAL')
            retains_motion = motion['psnr'] >= -.02 and motion['ssim'] >= -.001 and motion['lpips'] <= .001
            evidence[arm] = dict(strength=strength, nonworse_vs_scalar=nonworse,
                                 passes_vs_physical=passes_physical,
                                 zaps_motion_delta_vs_physical=motion, retains_motion=retains_motion,
                                 eligible=nonworse == len(CELLS) and passes_physical > 0 and retains_motion)
        candidate = next((arm for arm in STRENGTHS if evidence[arm]['eligible']), None)
        selection = dict(candidate=candidate, strengths=evidence)
        files.write(root/'weight_screen.json', selection)
        print('A148 WEIGHT SCREEN', selection, flush=True)
    print('A148 REPORT', root/'matrix.json', sum(row['status'] == 'complete' for row in rows), '/ 6', flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=(*panel.HOSTS, 'matrix'))
    parser.add_argument('task', choices=TASKS, nargs='?')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--zaps-parent', type=Path, default=Path(panel.zaps.rules.ROOT))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.host == 'matrix':
        raise SystemExit(0 if matrix(root) else 1)
    if (args.host, args.task) not in CELLS:
        parser.error('a blur task is required')
    execute(args.host, args.task, root/f'{args.host}_{args.task}', args.zaps_parent.resolve())


if __name__ == '__main__':
    main()
