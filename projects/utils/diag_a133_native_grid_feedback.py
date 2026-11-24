"""A133: distribute one A118 refinement step per coarse interval over native steps."""
import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a118_transfer_bridge as bridge
from utils import diag_transfer_objective_audit as audit
from utils import diagonal_parameter_tuning as timing
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = transfer.history.BASE/'diag_a133_native_grid_feedback'
ARMS = ('BASE', 'SCALAR', 'DIAGONAL')


def interval_weights(grid, anchors, alpha):
    """Partition in descending sampling order, using cumulative beta hazard."""
    files.require(grid == sorted(set(grid), reverse=True) and anchors == sorted(set(anchors), reverse=True) and
                  anchors[0] == grid[0] and anchors[-1] == grid[-1] == 0 and
                  set(anchors) <= set(grid), 'Coarse intervals must cover the native grid')
    hazard = {t:-math.log(alpha[t]) for t in grid}
    indices = {t:index for index,t in enumerate(grid)}
    weights, intervals = {0:1.}, []
    for upper, lower in zip(anchors, anchors[1:]):
        steps = grid[indices[upper]:indices[lower]+1]
        width = hazard[upper]-hazard[lower]
        files.require(width > 0, 'Diffusion hazard is not strictly increasing in training time')
        values = {t:(hazard[t]-hazard[s])/width for t,s in zip(steps,steps[1:])}
        files.require(all(0 < value <= 1 for value in values.values()) and
                      math.isclose(math.fsum(values.values()),1.,rel_tol=1e-12),
                      'Native weights do not conserve the interval budget')
        weights.update(values)
        intervals.append(dict(upper=upper, lower=lower, native_steps=len(values),
                              total_weight=math.fsum(values.values())))
    files.require(set(weights) == set(grid), 'Missing native weight')
    return weights, intervals


def schedule(host, old, anchors, model):
    if host == 'dps':
        grid = list(range(999, -1, -1))
        alpha = None
        for task in transfer.TASKS:
            rt = transfer.history.native(dict(case=dict(task=task)), 'dps')
            config = old['dps_configs'][task]
            dps,_,_ = rt.reference(str(Path(config['dps_root']).resolve()))
            src = dps.create_sampler(**config['diffusion_config'])
            files.require(src.timestep_map[::-1] == grid and not src.rescale_timesteps,
                          'Archived DPS is not the expected native training grid')
            current = dict(zip(src.timestep_map, map(float,src.alphas_cumprod)))
            files.require(alpha is None or alpha == current, 'DPS task schedules differ')
            alpha = current
    else:
        import torch
        from utils.pigdm_noisy_r1_runtime import original_diffusion
        grid = list(range(990, -1, -10))
        times = torch.tensor(grid, device=model.device, dtype=torch.long)
        alpha = dict(zip(grid, original_diffusion().alpha(times).tolist()))
        anchors = timing.mapped_times(anchors, grid)[::-1]
    weights, intervals = interval_weights(grid, anchors, alpha)
    return grid, anchors, weights, intervals


def refinement(wavelet, weights, counts, trace):
    full = transfer.runtime.refinement(wavelet, 'A118_DIAGONAL', counts, trace)

    def apply(operator, q, residual, alpha, scalar, t):
        candidate = full(operator, q, residual, alpha, scalar, t)
        rho = weights[t]
        row = trace[-1]
        # The line loss is exactly quadratic. A fraction rho of the line minimizer
        # takes rho*(2-rho) of the full objective decrease on this frozen state.
        row['applied_weight'] = rho
        row['proxy_after_fraction'] = row['proxy_initial']-rho*(2-rho)*(row['proxy_initial']-row['proxy_final'])
        if rho == 1:
            return candidate
        anchor = scalar['gamma']*q
        return anchor+rho*(candidate-anchor)

    return apply


def sample(host, old, case, y, model, wavelet, kernel, arm, weights, counts, trace, progress):
    op = transfer.make_operator(old, case, model.device, kernel, counts)
    rt = transfer.history.native(dict(case=case), host)
    # Use the original full-native scalar loop. The diagonal arm changes only
    # the correction expression after that loop has computed the scalar anchor.
    run = rt.run
    if arm == 'DIAGONAL':
        apply = refinement(wavelet, weights, counts, trace)
        run = transfer.runtime.instrument(rt,host,'A118_DIAGONAL',max(weights),apply)
    if host == 'pigdm':
        return run('A133',model,op,y,case,'BASE' if arm=='BASE' else 'SHRINK',counts,progress=progress)
    config = old['dps_configs'][case['task']]
    return run(model,op,y,case,'DPS' if arm=='BASE' else 'SHRINK',
               config['diffusion_config'],config['dps_root'],counts,progress=progress)


def measure(root, host, old, case, obs, model, wavelet, kernel, arm, grid, weights, source):
    import torch

    directory = root/'cases'/case['key']/arm
    directory.mkdir(parents=True)
    record = directory/'run.json'
    row = dict(status='running',case=case['key'],task=case['task'],arm=arm)
    files.write(record,row)
    trace, counts = [], {}
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)

            def progress(step, state):
                row.update(native_step=step, counts=dict(state))
                files.write(record,row)

            def operation():
                y = audit.checked_tensor(obs['path'],model.device)
                return sample(host,old,case,y,model,wavelet,kernel,arm,weights,counts,trace,progress)

            (out,detail),seconds = clock.clocks(torch.cuda.synchronize,operation)
            identity = transfer.history.fingerprint(detail,shared.rng(),counts,host)
        previous = dict(source['pairing'])
        previous.pop('feedback_entry_sha256',None)
        files.require(identity == previous, 'Native timestep grid, initial state or RNG differs')
        files.require(counts['forwards'] == counts['forward_attempts'] == counts['input_vjp_attempts'] == len(grid) and
                      counts['optimizer_backward_attempts'] == counts.get('input_jvp_attempts',0) == 0 and
                      counts['calls'] == detail['grid'] == grid, 'Unexpected native network budget')
        files.require([step['t'] for step in detail['trace'] if 'weight' in step] ==
                      ([] if arm=='BASE' else grid) and
                      [step['t'] for step in trace] == (grid if arm=='DIAGONAL' else []),
                      'Full native scalar/diagonal coverage differs')
        if arm == 'BASE':
            folder = Path(obs['baseline']).parent
            archived = audit.checked_tensor(audit.qualified_artifact(folder,'output.pt',source),'cpu')
            relative = audit.norm(out-archived)/max(audit.norm(archived),1e-30)
            files.require(relative <= 1e-4, f'Native original baseline replay differs: {relative:.3g}')
            row['archive_relative_l2'] = relative
        metric_start = time.perf_counter()
        q = bridge.output_metrics(out,shared.load_image(case['image']))
        metric_seconds = time.perf_counter()-metric_start
        torch.save(out,directory/'output.pt')
        files.write(directory/'refinement_trace.json',trace)
        row.update(status='complete',counts=counts,full_seconds=seconds,metric_seconds=metric_seconds,
                   method_nfe=len(grid),**q)
        print('A133 COMPLETE',host,case['key'],arm,q['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as exc:
        row.update(status='failed',error=repr(exc),counts=counts)
        files.write(directory/'partial_trace.json',trace)
        raise
    finally:
        files.write(record,row)


def load_plan(host, parent):
    plan = files.read(parent/'plan.json')
    files.require(plan['host'] == host and plan['nfe'] == (1000 if host=='dps' else 100) and
                  files.read(parent/'summary.json')['status'] == 'complete' and len(plan['cases']) == 20,
                  'A133 requires the completed A124R1 parent')
    a118_path = Path(plan['parent'])/'plan.json'
    files.require(files.digest(a118_path) == plan['parent_plan_sha256'], 'A118 grid record changed')
    a118 = files.read(a118_path)
    files.require(a118['experiment'] == 'A118' and sorted(a118['grid']) == files.grid(),
                  'Original A118 grid differs')
    files.require(files.digest(plan['motion_kernel']) == plan['motion_kernel_sha256'] ==
                  a118['motion_kernel_sha256'] and files.digest(plan['checkpoint']['path']) ==
                  plan['checkpoint']['sha256'] == a118['checkpoint_sha256'], 'A118 kernel or checkpoint changed')
    cases = [case for case in plan['cases'] if case['image_id'] in bridge.IMAGE_IDS]
    files.require(len(cases) == 6 and {(c['image_id'],c['task']) for c in cases} ==
                  {(i,t) for i in bridge.IMAGE_IDS for t in transfer.TASKS}, 'Frozen panel differs')
    items = []
    for case in cases:
        key = case['key']
        if host == 'dps':
            path, sha = case['a118_y'],case['a118_y_sha256']
        else:
            obs = files.read(parent/'observations'/key/'observation.json')
            path, sha = obs['path'],obs['sha256']
        baseline = parent/'cases'/key/'BASE'/'run.json'
        source = files.read(baseline)
        files.require(source['status'] == 'complete' and source['case'] == case and source['y_sha256'] == sha and
                      case['sigma'] == .05 and files.digest(path) == sha and
                      files.digest(case['image']) == case['image_sha256'], 'Frozen A124R1 case differs')
        items.append((case,dict(path=path,sha256=sha,baseline=str(baseline),baseline_sha256=files.digest(baseline)),source))
    return plan, sorted(a118['grid'],reverse=True), items, a118_path


def execute(host, parent, root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    old, anchors, inputs, a118_path = load_plan(host,parent)
    card = transfer.PROJECTS/'experiment_cards/A133_native_grid_feedback.md'
    sources = {str(p):files.digest(p) for p in (Path(__file__),card,parent/'plan.json',a118_path)}
    path = root/'summary.json'
    if path.exists():
        report = files.read(path)
        files.require(report['status']=='complete' and report['sources']==sources,
                      f'Existing partial or incompatible A133 run retained at {root}')
        for group in report['groups']:
            print('A133 SUMMARY', host, group, flush=True)
        print('A133 REPORT',path,flush=True)
        return
    admission.check(torch,root)
    root.mkdir(parents=True)
    report = dict(experiment='A133',host=host,status='running',cases=[],groups=[],actual_nfe=0,sources=sources)
    files.write(path,report)
    start = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(old['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = np.load(old['motion_kernel'],allow_pickle=False)
        scope = native.dps_reference_scope() if host=='dps' else nullcontext()
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-start
        with precision('LEGACY_IEEE'), frozen_network(model.model), input_vjp_parameters(model.model), scope:
            grid,coarse,weights,intervals = schedule(host,old,anchors,model)
            files.write(root/'resolved_config.json',dict(sources=sources,host=host,grid=grid,
                a118_grid=anchors,mapped_grid=coarse,weights=weights,intervals=intervals,
                terminal_weight=weights[0],cases=[dict(case,observation=obs) for case,obs,_ in inputs]))
            for index,(case,obs,source) in enumerate(inputs):
                runs = {}
                order = ARMS[index % len(ARMS):]+ARMS[:index % len(ARMS)]
                for arm in order:
                    runs[arm] = measure(root,host,old,case,obs,model,wave,kernel,arm,grid,weights,source)
                    report['actual_nfe'] += runs[arm]['counts']['forwards']
                    files.write(path,report)
                report['cases'].append(dict(case=case['key'],task=case['task'],arms=runs))
                report['groups'] = bridge.summarize(report['cases'],'dps')
                files.write(path,report)
        files.require(report['actual_nfe']==18*old['nfe'] and len(report['groups'])==2,
                      'Incomplete A133 native panel')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),actual_nfe_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-start
        files.write(path,report)
    for group in report['groups']:
        print('A133 SUMMARY',host,group,flush=True)
    print('A133 REPORT',path,'actual_nfe=',report['actual_nfe'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','pigdm'))
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    parent = (args.parent or transfer.OUTPUTS/f'A124R1_{args.host}_ffhq').resolve()
    root = (args.output or ROOT/f'A133_{args.host}_ffhq').resolve()
    files.require(root != parent and not root.is_relative_to(parent),'Use a separate A133 output directory')
    execute(args.host,parent,root)


if __name__ == '__main__':
    main()
