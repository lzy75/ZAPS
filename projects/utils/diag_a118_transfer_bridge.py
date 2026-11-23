"""A131 ZAPS learned-direction ablation and A132 exact-grid DPS transfer."""
import argparse
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_transfer_objective_audit as audit
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import shared_prefix_feedback_runtime as zaps
from utils import transfer_diagonal_runtime as native

ROOT = transfer.history.BASE/'diag_a118_transfer_bridge'
DPS_PARENT = transfer.OUTPUTS/'A124R1_dps_ffhq'
IMAGE_IDS = ('00108', '00175', '00431')
ZAPS_ARMS = ('LEARNED_SCALAR', 'LEARNED_DIAGONAL', 'PHYSICAL_SCALAR', 'PHYSICAL_DIAGONAL')
DPS_ARMS = ('BASE', 'SCALAR', 'DIAGONAL')


def physical_suffix():
    return audit.insert_probe(zaps._custom_suffix, [
        ('        hv = zaps.dwt.synthesis(zaps.D[i]*zaps.dwt.analysis(v))\n'
         '        q = zaps.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))',
         '        q = zaps.zeta[i]*v/ab[t].sqrt().clamp(min=1e-8)')])


def exact_grid(diffusion, grid):
    def select(num_steps, _section_counts):
        files.require(num_steps == 1000 and len(grid) == 30 and len(set(grid)) == 30 and
                      min(grid) == 0 and max(grid) == num_steps-1, 'Invalid A118 training-index grid')
        return set(grid)
    return patch.object(diffusion, 'space_timesteps', select)


def output_metrics(out, truth):
    from utils.diag_noise_feasible_zaps import metrics
    return dict(quality=metrics(out, truth), raw_mse=audit.mse(out-truth))


def archived_error(out, folder, row, artifact):
    expected = audit.checked_tensor(audit.qualified_artifact(folder, artifact, row), 'cpu')
    ratio = audit.norm(out-expected)/max(audit.norm(expected), 1e-30)
    files.require(ratio <= 1e-4, f'Archived A118 control differs for {artifact}: {ratio:.3g}')
    return ratio


def zaps_case(model, kernel, case, y, truth, grid, folder):
    import torch
    from utils.diag_shared_prefix_feedback import rng_equal

    ref = files.read(folder/'run.json')
    files.require(ref['status'] == 'complete' and ref['case'] == case, 'A118 reference changed')
    with clock.counters(model.model) as count:
        operator = zaps.make_operator(case, model.device, count=count, kernel=kernel)
        prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize,
            lambda:zaps.train_prefix(model, operator, y, case))
    files.require(count['forwards'] == count['forward_attempts'] == 270 and count['calls'] == grid*9 and
                  count['optimizer_backward_attempts'] == count['optimizer_backward_completed'] == 9,
                  'A118 nine-unroll prefix differs')
    prefix_count = count
    runs = {}
    paired_rng = None
    altered = physical_suffix()
    for arm in ZAPS_ARMS:
        method = audit.DIAG if arm.endswith('DIAGONAL') else audit.SCALAR
        suffix = altered if arm.startswith('PHYSICAL') else zaps._custom_suffix
        with clock.counters(model.model) as count, patch.object(zaps, '_custom_suffix', suffix):
            operator = zaps.make_operator(case, model.device, count=count, kernel=kernel)
            (out, detail), seconds = clock.clocks(torch.cuda.synchronize,
                lambda:zaps.run_suffix(model, operator, y, case, method, prefix))
        files.require(count['forwards'] == count['forward_attempts'] == 30 and count['calls'] == grid and
                      count['optimizer_backward_attempts'] == count['optimizer_backward_completed'] == 1 and
                      count['input_vjp_attempts'] == count.get('input_jvp_attempts', 0) == 0,
                      'ZAPS suffix budget or grid differs')
        if paired_rng is None:
            paired_rng = detail['final_rng']
        files.require(rng_equal(paired_rng, detail['final_rng']), 'ZAPS suffix RNG differs')
        row = dict(counts=count, suffix_seconds=seconds, full_seconds=prefix_seconds+seconds,
                   method_nfe=300, **output_metrics(out, truth))
        if arm.startswith('LEARNED'):
            row['a118_replay_relative_l2'] = archived_error(out, folder, ref['arms'][method], method+'.pt')
        runs[arm] = row
        print('A131 COMPLETE', case['key'], arm, row['quality'], flush=True)
    return dict(case=case['key'], task=case['task'], prefix_counts=prefix_count, arms=runs)


def dps_case(model, wavelet, kernel, case, y, truth, grid, host_plan):
    import torch

    fixed = host_plan['dps_configs'][case['task']]
    runtime = transfer.history.native(dict(case=case), 'dps')
    dps, _, _ = runtime.reference(str(Path(fixed['dps_root']).resolve()))
    dense = dps.create_sampler(**fixed['diffusion_config'])
    files.require(dense.timestep_map == list(range(1000)), 'Expected original dense DPS reference')
    paired_rng = None
    runs = {}
    with exact_grid(dps, grid):
        sparse = dps.create_sampler(**fixed['diffusion_config'])
        files.require(sparse.timestep_map == grid[::-1] and
                      all(abs(float(a)-float(dense.alphas_cumprod[t])) <= 1e-12
                          for a, t in zip(sparse.alphas_cumprod, sparse.timestep_map)),
                      'DPS skip sampler did not preserve A118 indices and original alpha levels')
        for arm in DPS_ARMS:
            trace = []
            with clock.counters(model.model) as count:
                native.initialize_counts(count)

                def operation():
                    op = transfer.make_operator(host_plan, case, model.device, kernel, count)
                    original_arm = {'BASE':'BASE', 'SCALAR':'TAIL_SCALAR',
                                    'DIAGONAL':'A118_DIAGONAL'}[arm]
                    refine = transfer.runtime.refinement(wavelet, original_arm, count, trace)
                    sample = transfer.runtime.instrument(runtime, 'dps', original_arm, max(grid), refine)
                    return sample(model, op, y, case, 'DPS' if arm == 'BASE' else 'SHRINK',
                                  fixed['diffusion_config'], fixed['dps_root'], count)

                (out, detail), seconds = clock.clocks(torch.cuda.synchronize, operation)
                pairing = transfer.history.fingerprint(detail, shared.rng(), count, 'dps')
            files.require(count['forwards'] == count['forward_attempts'] == count['input_vjp_attempts'] == 30 and
                          count['optimizer_backward_attempts'] == count.get('input_jvp_attempts', 0) == 0 and
                          count['calls'] == detail['grid'] == grid, 'DPS must actually run only the A118 grid')
            files.require([step['t'] for step in detail['trace'] if 'weight' in step] ==
                          ([] if arm == 'BASE' else grid) and
                          [step['t'] for step in trace] == (grid if arm == 'DIAGONAL' else []),
                          'DPS controller coverage differs')
            if paired_rng is None:
                paired_rng = pairing
            files.require(pairing == paired_rng, 'DPS input or native RNG differs between arms')
            runs[arm] = dict(counts=count, full_seconds=seconds, method_nfe=30, **output_metrics(out, truth))
            print('A132 COMPLETE', case['key'], arm, runs[arm]['quality'], flush=True)
    return dict(case=case['key'], task=case['task'], arms=runs)


def paired(rows, candidate, reference):
    delta = [{m:row['arms'][candidate]['quality'][m]-row['arms'][reference]['quality'][m]
              for m in transfer.METRICS} for row in rows]
    mean = {m:statistics.mean(r[m] for r in delta) for m in transfer.METRICS}
    wins = sum(r['psnr'] > 0 for r in delta)
    return dict(candidate=candidate, reference=reference, images=len(rows), delta=mean, psnr_wins=wins,
                passes=len(rows)==3 and wins>=2 and mean['psnr']>=.1 and mean['ssim']>=0 and mean['lpips']<=0)


def summarize(rows, host):
    results = []
    for task in transfer.TASKS:
        subset = [row for row in rows if row['task'] == task]
        if len(subset) != 3:
            continue
        arms = {arm:dict(quality={m:statistics.mean(row['arms'][arm]['quality'][m] for row in subset)
                                  for m in transfer.METRICS},
                         seconds=statistics.mean(row['arms'][arm]['full_seconds'] for row in subset),
                         nfe=subset[0]['arms'][arm]['method_nfe']) for arm in subset[0]['arms']}
        if host == 'zaps':
            comparisons = [paired(subset,'LEARNED_DIAGONAL','LEARNED_SCALAR'),
                           paired(subset,'PHYSICAL_DIAGONAL','PHYSICAL_SCALAR')]
            interaction = {m:comparisons[0]['delta'][m]-comparisons[1]['delta'][m] for m in transfer.METRICS}
        else:
            comparisons = [paired(subset,'SCALAR','BASE'), paired(subset,'DIAGONAL','SCALAR'),
                           paired(subset,'DIAGONAL','BASE')]
            interaction = None
        results.append(dict(task=task, arms=arms, paired=comparisons, learned_minus_physical_increment=interaction))
    return results


def execute(host, zaps_root, dps_root, output):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision
    from contextlib import nullcontext

    label = 'A131' if host == 'zaps' else 'A132'
    card = transfer.PROJECTS/f'experiment_cards/{label}_a118_transfer_bridge.md'
    plan_path = zaps_root/'plan.json'
    source = files.read(plan_path)
    parent_summary = files.read(zaps_root/'summary.json')
    files.require(source['experiment'] == 'A118' and parent_summary['status'] == 'complete' and
                  parent_summary['completed_cases'] == 20 and source['method_nfe'] == 300,
                  'Completed A118 required')
    grid = sorted(source['grid'], reverse=True)
    files.require(grid == files.grid()[::-1], 'Original A118 30-position grid differs')
    cases = [case for case in source['cases'] if case['image_id'] in IMAGE_IDS and case['task'] in transfer.TASKS]
    files.require({(case['image_id'],case['task']) for case in cases} ==
                  {(image,task) for image in IMAGE_IDS for task in transfer.TASKS} and len(cases) == 6 and
                  all(case['sigma'] == .05 for case in cases), 'Frozen six-case panel differs')
    kernel_path = zaps_root/'motion_kernel.npy'
    files.require(files.digest(kernel_path) == source['motion_kernel_sha256'], 'Frozen A118 motion kernel changed')
    checkpoint = transfer.PROJECTS/'modules/models/ffhq_10m.pt'
    files.require(files.digest(checkpoint) == source['checkpoint_sha256'], 'A118 FFHQ checkpoint changed')
    sources = {str(p):files.digest(p) for p in (Path(__file__), card, plan_path)}
    dps_plan = None
    if host == 'dps':
        dps_plan_path = dps_root/'plan.json'
        dps_plan = files.read(dps_plan_path)
        files.require(dps_plan['host'] == 'dps' and dps_plan['nfe'] == 1000 and
                      files.read(dps_root/'summary.json')['status'] == 'complete' and
                      dps_plan['parent_plan_sha256'] == sources[str(plan_path)] and
                      dps_plan['checkpoint']['sha256'] == source['checkpoint_sha256'] and
                      dps_plan['motion_kernel_sha256'] == source['motion_kernel_sha256'],
                      'A124R1 DPS parent is incompatible')
        sources[str(dps_plan_path)] = files.digest(dps_plan_path)
    observations = {}
    for case in cases:
        key = case['key']
        obs = files.read(zaps_root/'observations'/key/'observation.json')
        files.require(obs['sigma'] == case['sigma'] and files.digest(obs['path']) == obs['sha256'] and
                      files.digest(case['image']) == case['image_sha256'], 'Frozen A118 input changed')
        observations[key] = dict(path=obs['path'], sha256=obs['sha256'])
        if dps_plan is not None:
            inherited = next(row for row in dps_plan['cases'] if row['key'] == key)
            files.require(inherited == dict(case, a118_y=obs['path'], a118_y_sha256=obs['sha256']),
                          'A124R1 DPS uses a different image, seed, or observation')
    summary_path = output/'summary.json'
    if summary_path.exists():
        report = files.read(summary_path)
        files.require(report['status'] == 'complete' and report['sources'] == sources,
                      f'Prior partial or incompatible run retained at {output}')
        for row in report['groups']:
            print(label, 'SUMMARY', row, flush=True)
        return
    admission.check(torch, output)
    output.mkdir(parents=True)
    report = dict(experiment=label, status='running', host=host, sources=sources, grid=grid,
                  observations=observations, cases=[], groups=[], actual_nfe=0)
    files.write(summary_path, report)
    start = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        model.model.eval()
        wavelet = OrthogonalDWT2D(wave='db4', level=3).to(model.device) if host == 'dps' else None
        kernel = np.load(kernel_path, allow_pickle=False)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-start
        params = input_vjp_parameters(model.model) if host == 'dps' else nullcontext()
        imports = native.dps_reference_scope() if host == 'dps' else nullcontext()
        with precision('LEGACY_IEEE'), frozen_network(model.model), params, imports:
            for case in cases:
                key = case['key']
                y = audit.checked_tensor(observations[key]['path'], model.device)
                if host == 'zaps':
                    from modules.main_single import load_image_as_tensor
                    truth = load_image_as_tensor(case['image'])
                    row = zaps_case(model, kernel, case, y, truth, grid, zaps_root/'cases'/key)
                else:
                    truth = shared.load_image(case['image'])
                    row = dps_case(model, wavelet, kernel, case, y, truth, grid, dps_plan)
                report['cases'].append(row)
                report['actual_nfe'] += sum(arm['counts']['forwards'] for arm in row['arms'].values())
                report['actual_nfe'] += row.get('prefix_counts', {}).get('forwards', 0)
                report['groups'] = summarize(report['cases'], host)
                files.write(summary_path, report)
        files.require(report['actual_nfe'] == (2340 if host == 'zaps' else 540) and
                      len(report['groups']) == 2, 'Incomplete task panel or unexpected actual cost')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), actual_nfe_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-start
        files.write(summary_path, report)
    for row in report['groups']:
        print(label, 'SUMMARY', row, flush=True)
    print(label, 'REPORT', summary_path, 'actual_nfe=', report['actual_nfe'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=('zaps', 'dps'))
    parser.add_argument('--zaps-root', type=Path, default=transfer.PARENT)
    parser.add_argument('--dps-root', type=Path, default=DPS_PARENT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT/('A131_zaps' if args.host == 'zaps' else 'A132_dps')).resolve()
    parents = (args.zaps_root.resolve(), args.dps_root.resolve())
    files.require(all(output != parent and not output.is_relative_to(parent) for parent in parents),
                  'Use a separate result directory')
    execute(args.host, *parents, output)


if __name__ == '__main__':
    main()
