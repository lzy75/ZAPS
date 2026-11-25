"""A140: test attenuation and an observed-neighborhood target on A139 inpainting."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import statistics
import time

from utils import diag_a139_extend_late_diagonal as parent

admission, audit, bridge = parent.admission, parent.audit, parent.bridge
clock, files, history = parent.clock, parent.files, parent.history
late, native, shared = parent.late, parent.native, parent.shared
ROOT = parent.ROOT.with_name('diag_a140_inpainting_refinement')
CONTROLS = ('SCALAR', 'DIAGONAL')
DAMPING = {'DAMP025': .25, 'DAMP050': .50}
LOCAL = 'LOCAL_TARGET'
CANDIDATES = (*DAMPING, LOCAL)
WINDOW = 5
MIN_SAMPLES = 3


def box_sum(value):
    from torch.nn.functional import avg_pool2d

    return avg_pool2d(value, WINDOW, stride=1, padding=WINDOW//2, divisor_override=1)


def local_target(view, e, support_count):
    """Mean of measured residuals on a centered window, returned in observation order."""
    numerator = box_sum(view.transpose(e))
    mean = (numerator / support_count.clamp_min(1)) * (support_count >= MIN_SAMPLES)
    return view.H(mean)


def controller(op, arm, wave, counts, trace):
    import math
    import torch

    physical = parent.ObservedResponse(op)
    solve = late.terminal.controller('DIAGONAL_K12', wave, counts, trace)
    support_count = None

    def refine(_op, q, residual, alpha, scalar, t):
        nonlocal support_count
        anchor = scalar['gamma']*q
        if arm == LOCAL:
            view = native.LinearView(physical, counts)
            response = view.H(anchor/math.sqrt(alpha)).double()
            error = scalar['weight']*residual.double().reshape_as(response)-response
            if support_count is None:
                support_count = box_sum(view.transpose(torch.ones_like(error)))
                counts['local_box_sums'] += 1
            # B and its adjoint pack and embed the same measured RGB entries.
            cleaned = local_target(view, error, support_count)
            counts['local_box_sums'] += 1
            adjusted_target = response+cleaned
            updated = solve(physical, q, adjusted_target, alpha, dict(scalar, weight=1.), t)
            trace[-1].update(goal='local_observed_increment', applied_fraction=1.)
            return updated
        updated = solve(physical, q, residual, alpha, scalar, t)
        fraction = DAMPING[arm]
        trace[-1].update(goal='raw_increment_before_damping', applied_fraction=fraction)
        return anchor+fraction*(updated-anchor)

    return refine


def load_inputs(host, source):
    config = files.read(source/'resolved_config.json')
    summary = files.read(source/'summary.json')
    grid = parent.GRIDS[host]
    positions = [t for t in grid if t <= late.POSITIONS[0]]
    files.require(
        config['experiment']==summary['experiment']=='A139' and
        config['host']==summary['host']==host and config['dataset']==summary['dataset']=='ffhq' and
        summary['status']=='complete' and summary['planned_runs']==summary['completed_runs']==
        len(parent.TASKS[host])*3*len(parent.ARMS) and
        summary['actual_cost']['forwards']==summary['actual_cost']['input_vjp_attempts']==
        summary['completed_runs']*len(grid) and config['grid']==grid and
        config['positions']==positions and config['max_updates']==late.depth.MAX_UPDATES and
        config['grad_rtol']==late.depth.GRAD_RTOL, 'Need a complete compatible A139')
    for name, digest in config['code_sha256'].items():
        files.require(files.digest(name)==digest, 'A139 scientific source changed: '+name)
    checkpoint = config['checkpoint']
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'], 'FFHQ checkpoint changed')
    cases = {row['case']:row for row in summary['cases'] if row['task']=='inpainting'}
    manifests = {row['case']['key']:row for row in config['cases'] if row['case']['task']=='inpainting'}
    entries = [entry for entry in history.load_sources(host, 'ffhq') if entry['case']['task']=='inpainting']
    files.require(set(cases)==set(manifests)=={e['case']['key'] for e in entries} and len(entries)==3,
                  'A139 inpainting panel differs')
    used = {}
    for entry in entries:
        case, plan = entry['case'],entry['plan']
        key = case['key']
        manifest = manifests[key]
        obs = history.observation(entry, host)
        files.require(case==manifest['case'] and plan['checkpoint']==checkpoint and
                      str(entry['root'].resolve())==manifest['source'] and
                      files.digest(entry['root']/'plan.json')==manifest['plan_sha256'] and
                      files.digest(case['image'])==manifest['image_sha256'] and
                      files.digest(obs['path'])==manifest['observation_sha256'], 'Frozen case changed: '+key)
        records = {}
        hashes = {}
        for arm in CONTROLS:
            folder = source/'cases'/key/arm
            path = folder/'run.json'
            row = files.read(path)
            files.require(row==cases[key]['arms'][arm] and row['status']=='complete' and
                          row['counts']['forwards']==row['counts']['input_vjp_attempts']==len(grid) and
                          row['counts']['calls']==grid and row['method_nfe']==len(grid),
                          'A139 result identity differs: '+key+'/'+arm)
            for artifact in ('output.pt', 'refinement_trace.json'):
                audit.qualified_artifact(folder, artifact, row)
            records[arm] = (row, folder)
            hashes[arm] = files.digest(path)
        scalar, diagonal = (records[arm][0] for arm in CONTROLS)
        files.require(scalar['pairing']==diagonal['pairing'] and scalar['late_entry']==diagonal['late_entry'],
                      'A139 controls are not paired')
        entry.update(observation=obs, controls=records)
        used[key] = dict(manifest, A139_record_sha256=hashes)
    return entries, checkpoint, tuple(positions), used


def region_metrics(physical, output, truth, noisy):
    import torch

    estimate = physical.H(output)
    mask = physical.Ht(torch.ones_like(estimate)).bool()
    if physical.selected is not None:
        noisy = noisy[physical.selected]
    error = (output-truth).double().square()
    measured = physical.H(truth)
    alignment = (estimate-measured).double()*(noisy-measured).double()
    return dict(observed_mse=float(error[mask].mean()), missing_mse=float(error[~mask].mean()),
                observed_fraction=float(mask.double().mean()), raw_mse=float(error.mean()),
                noisy_observation_mse=audit.mse(estimate-noisy),
                noise_alignment=float(alignment.mean()))


def sample(entry, host, model, wave, arm, positions, counts, trace, progress):
    y = audit.checked_tensor(entry['observation']['path'], model.device)
    op = history.operator(entry, entry['observation'], host, model.device, counts)
    runtime = history.native(entry, host)
    refine = controller(op, arm, wave, counts, trace)
    case, plan = entry['case'], entry['plan']
    if host=='pigdm':
        run = parent.pigdm_sampler(runtime, positions, refine)
        return run('A140', model, op, y, case, 'SHRINK', counts, progress=progress)
    run = late.matched.instrument(runtime, positions, refine)
    return run(model, op, y, case, 'SHRINK', plan['diffusion_config'], plan['dps_root'], counts, progress=progress)


def measure(root, entry, host, model, wave, arm, positions, evaluate):
    import torch
    from utils.diag_pigdm_noisy_r1 import fingerprint

    case = entry['case']['key']
    folder = root/'cases'/case/arm
    folder.mkdir(parents=True)
    path = folder/'run.json'
    row = dict(status='running', case=case, arm=arm)
    files.write(path, row)
    trace, counts = [], {}
    reference = entry['controls']['SCALAR'][0]
    grid = parent.GRIDS[host]
    entry_call = len(grid)-len(positions)+1
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts['local_box_sums'] = 0

            def check_entry(_network, args):
                if len(counts['calls'])==entry_call:
                    row['late_entry'] = fingerprint(args[0])
                    files.require(row['late_entry']==reference['late_entry'], 'A139 scalar prefix differs')

            def progress(step, state):
                row.update(native_step=step,counts=dict(state))
                files.write(path, row)

            with model.model.register_forward_pre_hook(check_entry):
                (output, detail), seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:sample(entry, host, model, wave, arm, positions, counts, trace, progress))
            pairing = history.fingerprint(detail, shared.rng(), counts, host)
        files.require(
            counts['forwards']==counts['forward_attempts']==counts['input_vjp_attempts']==len(grid) and
            counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0 and
            detail['grid']==counts['calls']==grid and pairing==reference['pairing'] and
            row['late_entry']==reference['late_entry'] and
            tuple(step['t'] for step in trace)==positions and
            [step['t'] for step in detail['trace'] if 'weight' in step]==grid, 'A140 pairing or budget differs')
        started = time.perf_counter()
        metrics = evaluate(output)
        metrics_seconds = time.perf_counter()-started
        torch.save(output, folder/'output.pt')
        files.write(folder/'refinement_trace.json', trace)
        row.update(status='complete', counts=counts, pairing=pairing, full_seconds=seconds,
                   metrics_seconds=metrics_seconds, method_nfe=len(grid), **metrics,
                   solver=late.solver_stats('DIAGONAL_K12', trace),
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','refinement_trace.json')})
        print('A140 COMPLETE', host, case, arm, row['quality'], f'{seconds:.2f}s', flush=True)
        return row
    except BaseException as error:
        row.update(status='failed', error=repr(error), counts=counts)
        files.write(folder/'partial_refinement.json', trace)
        raise
    finally:
        files.write(path, row)


REGION_KEYS = ('raw_mse', 'observed_mse', 'missing_mse', 'noisy_observation_mse', 'noise_alignment')


def summarize(cases):
    comparisons = []
    arms = {}
    for arm in (*CONTROLS, *CANDIDATES):
        rows = [case['arms'][arm] for case in cases]
        arms[arm] = dict(
            quality={key:statistics.mean(r['quality'][key] for r in rows) for key in parent.transfer.METRICS},
            regions={key:statistics.mean(r[key] for r in rows) for key in REGION_KEYS},
            full_seconds=statistics.mean(r['full_seconds'] for r in rows), nfe=rows[0]['method_nfe'],
            input_vjp=rows[0]['counts']['input_vjp_attempts'], inherited=arm in CONTROLS)
        if arm=='SCALAR':
            continue
        arms[arm]['mean_solver_updates'] = statistics.mean(r['solver']['total_updates'] for r in rows)
        for reference in ('SCALAR',) if arm=='DIAGONAL' else CONTROLS:
            pair = bridge.paired(cases, arm, reference)
            pair['region_delta'] = {key:statistics.mean(case['arms'][arm][key]-case['arms'][reference][key]
                                                       for case in cases) for key in REGION_KEYS}
            comparisons.append(pair)
    return dict(task='inpainting', images=len(cases), arms=arms, paired=comparisons)


def run_cases(report, path, inputs, host, root, checkpoint, positions):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    started = time.perf_counter()
    model = load_diffusion_model('ffhq', 'cuda:0')
    model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval()
    wave = OrthogonalDWT2D(wave='db4', level=3).to(model.device)
    torch.cuda.synchronize()
    report['model_load_seconds'] = time.perf_counter()-started
    scope = native.dps_reference_scope() if host=='dps' else nullcontext()
    with precision('LEGACY_IEEE'), frozen_network(model.model), input_vjp_parameters(model.model), scope:
        for index,entry in enumerate(inputs):
            case, obs = entry['case'], entry['observation']
            truth = (shared.load_image if host=='pigdm' else load_image_as_tensor)(case['image'])
            truth_device = truth.to(model.device)
            physical = parent.ObservedResponse(history.operator(entry, obs, host, model.device, None))
            noisy = audit.checked_tensor(obs['path'], model.device)

            def evaluate(output):
                with torch.no_grad():
                    regions = region_metrics(physical, output.to(model.device), truth_device, noisy)
                return dict(quality=bridge.output_metrics(output, truth)['quality'], **regions)

            arms = {}
            for name, (row,folder) in entry['controls'].items():
                output = audit.checked_tensor(folder/'output.pt', 'cpu')
                with torch.no_grad():
                    regions = region_metrics(physical, output.to(model.device), truth_device, noisy)
                files.require(abs(regions['observed_mse']-row['clean_observation_mse'])<=1e-9 and
                              abs(regions['raw_mse']-row['raw_mse'])<=1e-9,
                              'Archived region evaluation differs')
                arms[name] = dict(row, **regions)
            print('A140 CONTROL', host, case['key'], dict(
                stops=arms['DIAGONAL']['solver']['stops'],
                delta={k:arms['DIAGONAL'][k]-arms['SCALAR'][k] for k in REGION_KEYS}), flush=True)
            progress_row = dict(case=case['key'], arms=arms)
            report['cases'].append(progress_row)
            shift = index % len(CANDIDATES)
            for arm in CANDIDATES[shift:]+CANDIDATES[:shift]:
                row = measure(root, entry, host, model, wave, arm, positions, evaluate)
                arms[arm] = row
                report['completed_new_runs'] += 1
                for key in parent.NETWORK_COUNTS:
                    report['actual_cost'][key] += row['counts'].get(key,0)
                files.write(path, report)


def print_report(report, path):
    print('A140 SUMMARY', report['host'], report['group'], flush=True)
    print('A140 REPORT', path, 'runs=', report['completed_new_runs'],
          'nfe=', report['actual_cost']['forwards'], flush=True)


def execute(host, source, root):
    import torch

    files.require(not root.is_relative_to(source) and not source.is_relative_to(root), 'A140 output overlaps A139')
    inputs, checkpoint, positions, used = load_inputs(host, source)
    card = parent.transfer.PROJECTS/'experiment_cards/A140_inpainting_refinement.md'
    tests = Path(__file__).with_name('test_a140_inpainting_refinement.py')
    config = dict(experiment='A140', host=host, dataset='ffhq', task='inpainting',
                  source=str(source), A139_summary_sha256=files.digest(source/'summary.json'),
                  A139_config_sha256=files.digest(source/'resolved_config.json'), inputs=used,
                  window=WINDOW, min_samples=MIN_SAMPLES, damping=DAMPING, checkpoint=checkpoint,
                  grid=parent.GRIDS[host], positions=list(positions), max_updates=late.depth.MAX_UPDATES,
                  grad_rtol=late.depth.GRAD_RTOL,
                  code_sha256={str(p):files.digest(p) for p in (Path(__file__),card,tests)})
    path = root/'summary.json'
    if path.exists():
        previous = files.read(path)
        files.require(previous['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A140 retained at {root}')
        print_report(previous, path)
        return
    resources = admission.check(torch, root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json', config)
    report = dict(experiment='A140', host=host, status='running', cases=[], resource_check=resources,
                  planned_new_runs=len(inputs)*len(CANDIDATES), completed_new_runs=0,
                  actual_cost=dict.fromkeys(parent.NETWORK_COUNTS,0), cost_is_lower_bound=False,
                  timing='Candidate clocks are from A140. Archived controls are timed in A139.')
    files.write(path, report)
    try:
        run_cases(report, path, inputs, host, root, checkpoint, positions)
        files.require(report['completed_new_runs']==report['planned_new_runs'] and
                      report['actual_cost']['forwards']==report['planned_new_runs']*len(parent.GRIDS[host]),
                      'A140 budget incomplete')
        report['group'] = summarize(report['cases'])
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), cost_is_lower_bound=True)
        raise
    finally:
        files.write(path, report)
    print_report(report, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=tuple(parent.TASKS))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.parent or parent.ROOT/f'A139_{args.host}_ffhq'
    root = args.output or ROOT/f'A140_{args.host}_ffhq'
    execute(args.host, source.resolve(), root.resolve())


if __name__=='__main__':
    main()
