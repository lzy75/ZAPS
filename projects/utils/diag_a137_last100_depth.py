"""A137: compare one and twelve diagonal updates over the last 100 native DPS steps."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a118_transfer_bridge as bridge
from utils import diag_a133_native_grid_feedback as dense
from utils import diag_a134_matched_a118 as matched
from utils import diag_a135_dps_solver_depth as depth
from utils import diag_a136_terminal_depth_ablation as terminal
from utils import diag_transfer_objective_audit as audit
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = dense.ROOT.with_name('diag_a137_last100_depth')
POSITIONS = tuple(range(99,-1,-1))
ARMS = {'LAST100_K1':'DIAGONAL_K1','LAST100_K12':'DIAGONAL_K12'}


def solver_stats(method, trace):
    stops = Counter()
    updates = []
    gain = []
    for row in trace:
        if method=='DIAGONAL_K1':
            updates.append(int(row['proposed_extra_norm']>0))
            stops['one_step'] += 1
        else:
            updates.append(row['updates'])
            stops[row['stop']] += 1
            one = row['proxy']['one']
            if one > 1e-30:
                gain.append((one-row['proxy']['deep'])/one)
    return dict(total_updates=sum(updates),stops=dict(stops),
                median_extra_proxy_gain_from_k1=statistics.median(gain) if gain else None)


def measure(root, arm, config, case, observation, ref, grid, model, wave, kernel):
    import torch

    path = root/'cases'/case['key']/arm/'run.json'
    path.parent.mkdir(parents=True)
    row = dict(status='running',case=case['key'],task=case['task'],arm=arm)
    counts,trace = {},[]
    files.write(path,row)
    try:
        runtime = transfer.history.native(dict(case=case),'dps')
        settings = config['dps_configs'][case['task']]
        method = ARMS[arm]
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)

            def tick(step,state):
                row.update(native_step=step,counts=dict(state))
                files.write(path,row)

            def run():
                y = audit.checked_tensor(observation['path'],model.device)
                operator = transfer.make_operator(config,case,model.device,kernel,counts)
                refine = terminal.controller(method,wave,counts,trace)
                sample = matched.instrument(runtime,POSITIONS,refine)
                return sample(model,operator,y,case,'SHRINK',settings['diffusion_config'],
                              settings['dps_root'],counts,progress=tick)

            (out,detail),seconds = clock.clocks(torch.cuda.synchronize, run)
            identity = transfer.history.fingerprint(detail,shared.rng(),counts,'dps')
        expected = dict(ref['pairing'])
        expected.pop('feedback_entry_sha256',None)
        files.require(identity==expected and detail['grid']==grid==counts['calls'] and
                      counts['forwards']==counts['forward_attempts']==counts['input_vjp_attempts']==1000 and
                      counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0,
                      'A137 native trajectory budget or RNG differs')
        files.require([step['t'] for step in detail['trace'] if 'weight' in step]==grid and
                      tuple(step['t'] for step in trace)==POSITIONS, 'Full scalar or last-100 coverage differs')
        if method=='DIAGONAL_K1':
            files.require(all(step['fraction']==1. for step in trace),'K1 requires one full update')
        else:
            files.require(all(0<=step['updates']<=depth.MAX_UPDATES and step['stop'] in
                              ('projected_gradient_tolerance','iteration_budget') for step in trace),
                          'K12 solver budget or termination differs')
        result = bridge.output_metrics(out,shared.load_image(case['image']))
        torch.save(out,path.parent/'output.pt')
        files.write(path.parent/'refinement_trace.json',trace)
        row.update(status='complete',counts=counts,full_seconds=seconds,method_nfe=1000,
                   solver=solver_stats(method,trace),**result)
        print('A137 COMPLETE',case['key'],arm,row['quality'],row['solver'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as error:
        row.update(status='failed',error=repr(error),counts=counts)
        files.write(path.parent/'partial_trace.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(rows):
    groups = []
    for task in transfer.TASKS:
        subset = [row for row in rows if row['task']==task]
        if len(subset)!=3:
            continue
        comparisons = [bridge.paired(subset,candidate,reference) for candidate,reference in (
            ('LAST100_K1','SCALAR'),('LAST100_K12','SCALAR'),
            ('LAST100_K12','LAST100_K1'),('LAST100_K1','THREE_K1'),
            ('LAST100_K12','THREE_K12'),('THREE_K12','THREE_K1'))]
        arms = {}
        for arm in ('SCALAR','THREE_K1','THREE_K12',*ARMS):
            data = [row['arms'][arm] for row in subset]
            arms[arm] = dict(quality={key:statistics.mean(row['quality'][key] for row in data)
                                     for key in transfer.METRICS},
                             seconds=statistics.mean(row['full_seconds'] for row in data),
                             nfe=1000,input_vjp=1000,
                             timing_source='A133' if arm=='SCALAR' else ('A136' if arm.startswith('THREE') else 'A137'))
            if arm in ARMS:
                stops = Counter()
                for row in data:
                    stops.update(row['solver']['stops'])
                arms[arm]['mean_updates']=statistics.mean(row['solver']['total_updates'] for row in data)
                arms[arm]['stops']=dict(stops)
                arms[arm]['median_extra_proxy_gain_per_image']=[
                    row['solver']['median_extra_proxy_gain_from_k1'] for row in data]
        interaction = {metric:comparisons[2]['delta'][metric]-comparisons[5]['delta'][metric]
                       for metric in transfer.METRICS}
        groups.append(dict(task=task,arms=arms,paired=comparisons,coverage_depth_interaction=interaction))
    return groups


def source_cases(a136_root,a133_root,dps_root):
    config,_,items,_ = dense.load_plan('dps',dps_root)
    source133,source136 = a133_root/'summary.json',a136_root/'summary.json'
    a133,a136 = files.read(source133),files.read(source136)
    grid = list(range(999,-1,-1))
    case_keys = {case['key'] for case,_,_ in items}
    files.require(a133['status']=='complete' and a133['experiment']=='A133' and a133['host']=='dps' and
                  a133['actual_nfe']==18000 and a133['sources'][str(dps_root/'plan.json')]==files.digest(dps_root/'plan.json')
                  and files.read(a133_root/'resolved_config.json')['grid']==grid,'Completed matching A133 required')
    files.require(a136['experiment']=='A136' and a136['status']=='complete' and a136['actual_new_nfe']==12000 and
                  tuple(a136['positions'])==depth.POSITIONS and a136['max_updates']==depth.MAX_UPDATES and
                  a136['grad_rtol']==depth.GRAD_RTOL and a136['sources'][str(source133)]==files.digest(source133) and
                  a136['sources'][str(dps_root/'plan.json')]==files.digest(dps_root/'plan.json'),
                  'Completed matching A136 required')
    controls = {row['case']:row['arms'] for row in a136['cases']}
    original = {row['case']:row['arms']['SCALAR'] for row in a133['cases']}
    files.require(set(controls)==set(original)==case_keys,'Archived case sets differ')
    paths = [source133,source136,dps_root/'plan.json']
    for key in sorted(case_keys):
        files.require(controls[key]['SCALAR']==original[key],'A136 scalar source changed')
        for arm in ('SCALAR','DIAGONAL_K1','DIAGONAL_K12'):
            path = (a133_root if arm=='SCALAR' else a136_root)/'cases'/key/arm/'run.json'
            row = files.read(path)
            files.require(row==controls[key][arm] and row['status']=='complete' and
                          row['counts']['calls']==grid and row['counts']['forwards']==
                          row['counts']['input_vjp_attempts']==1000,'Archived paired arm differs')
            paths.append(path)
    return config,items,controls,grid,paths


def execute(a136_root,a133_root,dps_root,root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    config,items,controls,grid,paths = source_cases(a136_root,a133_root,dps_root)
    card = transfer.PROJECTS/'experiment_cards/A137_last100_depth.md'
    sources = {str(p):files.digest(p) for p in (*paths,card,Path(__file__),Path(depth.__file__),Path(terminal.__file__))}
    path = root/'summary.json'
    if path.exists():
        report = files.read(path)
        files.require(report['status']=='complete' and report['sources']==sources,
                      f'Existing partial or incompatible run retained at {root}')
        for group in report['groups']:
            print('A137 SUMMARY',group,flush=True)
        return
    admission.check(torch,root)
    root.mkdir(parents=True)
    report = dict(experiment='A137',status='running',sources=sources,cases=[],groups=[],actual_new_nfe=0)
    files.write(path,report)
    files.write(root/'resolved_config.json',dict(grid=grid,positions=POSITIONS,three_positions=depth.POSITIONS,
                max_updates=depth.MAX_UPDATES,grad_rtol=depth.GRAD_RTOL,sources=sources,
                cases=[dict(case,observation=obs) for case,obs,_ in items]))
    started = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(config['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = np.load(config['motion_kernel'],allow_pickle=False)
        with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),native.dps_reference_scope():
            for index,(case,obs,ref) in enumerate(items):
                key = case['key']
                old = controls[key]
                arms = dict(SCALAR=old['SCALAR'],THREE_K1=old['DIAGONAL_K1'],THREE_K12=old['DIAGONAL_K12'])
                for arm in ARMS if index%2==0 else reversed(ARMS):
                    row = measure(root,arm,config,case,obs,ref,grid,model,wave,kernel)
                    arms[arm] = row
                    report['actual_new_nfe'] += row['counts']['forwards']
                    files.write(path,report)
                report['cases'].append(dict(case=key,task=case['task'],arms=arms))
                report['groups'] = summarize(report['cases'])
                files.write(path,report)
        files.require(report['actual_new_nfe']==12000 and len(report['groups'])==2,'Incomplete A137 panel')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),actual_nfe_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        files.write(path,report)
    for group in report['groups']:
        print('A137 SUMMARY',group,flush=True)
    print('A137 REPORT',path,'actual_new_nfe=',report['actual_new_nfe'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a136-root',type=Path,default=terminal.ROOT)
    parser.add_argument('--a133-root',type=Path,default=matched.PARENT)
    parser.add_argument('--a124-root',type=Path,default=bridge.DPS_PARENT)
    parser.add_argument('--output',type=Path,default=ROOT)
    args = parser.parse_args()
    parents = (args.a136_root.resolve(),args.a133_root.resolve(),args.a124_root.resolve())
    root = args.output.resolve()
    files.require(all(root!=parent and not root.is_relative_to(parent) for parent in parents),
                  'Use a separate A137 result directory')
    execute(*parents,root)


if __name__=='__main__':
    main()
