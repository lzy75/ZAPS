"""A136: full native DPS comparison of one versus up to twelve diagonal updates."""
import argparse
from pathlib import Path
import statistics
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a118_transfer_bridge as bridge
from utils import diag_a133_native_grid_feedback as dense
from utils import diag_a134_matched_a118 as matched
from utils import diag_a135_dps_solver_depth as depth
from utils import diag_transfer_objective_audit as audit
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = dense.ROOT.with_name('diag_a136_terminal_depth_ablation')
CANDIDATES = ('DIAGONAL_K1', 'DIAGONAL_K12')


def gaussian_gate(a135):
    selected = {row['t']:row for row in a135['groups'] if row['task']=='gaussian_deblur' and row['t'] in (95,24)}
    files.require(set(selected)=={95,24} and all(row['images']==3 for row in selected.values()),
                  'A135 Gaussian t=95 and t=24 summaries required')
    readout = {t:next(value for value in row['comparisons'] if value['pair']=='deep_minus_one')
               for t,row in selected.items()}
    for t,comparison in readout.items():
        print('A136 A135 CHECK',t,comparison,flush=True)
    files.require(all(c['proxy_delta']<0 and c['errors']['image_mse']<0 and
                      c['errors']['clean_observation_mse']<0 and c['image_mse_wins']>=2
                      for c in readout.values()), 'A135 Gaussian gate did not support full-trajectory depth testing')
    return readout


def controller(arm, wave, counts, trace):
    if arm == 'DIAGONAL_K1':
        return transfer.runtime.refinement(wave,'A118_DIAGONAL',counts,trace)
    files.require(arm == 'DIAGONAL_K12', 'Unknown A136 arm')

    def apply(op,q,residual,alpha,scalar,t):
        entry = dict(t=t,q=q,residual=residual,alpha=alpha,scalar=scalar)
        _,delta,details = depth.solve_depth(wave,op,entry,counts)
        trace.append(dict(t=t,**details))
        return delta

    return apply


def measure(root, arm, config, case, obs, ref, grid, model, wave, kernel):
    import torch

    folder = root/'cases'/case['key']/arm
    folder.mkdir(parents=True)
    path = folder/'run.json'
    row = dict(status='running',case=case['key'],task=case['task'],arm=arm)
    counts, trace = {}, []
    files.write(path,row)
    try:
        runtime = transfer.history.native(dict(case=case),'dps')
        settings = config['dps_configs'][case['task']]
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)

            def tick(step,state):
                row.update(native_step=step,counts=dict(state))
                files.write(path,row)

            def run():
                y = audit.checked_tensor(obs['path'],model.device)
                op = transfer.make_operator(config,case,model.device,kernel,counts)
                refine = controller(arm,wave,counts,trace)
                sample = matched.instrument(runtime,depth.POSITIONS,refine)
                return sample(model,op,y,case,'SHRINK',settings['diffusion_config'],
                              settings['dps_root'],counts,progress=tick)

            (out,detail),seconds = clock.clocks(torch.cuda.synchronize,run)
            identity = transfer.history.fingerprint(detail,shared.rng(),counts,'dps')
        expected = dict(ref['pairing'])
        expected.pop('feedback_entry_sha256',None)
        files.require(identity==expected and grid==detail['grid']==counts['calls'] and
                      counts['forwards']==counts['forward_attempts']==counts['input_vjp_attempts']==1000 and
                      counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0,
                      'A136 native grid, budget, or RNG differs')
        files.require([step['t'] for step in detail['trace'] if 'weight' in step]==grid and
                      tuple(step['t'] for step in trace)==depth.POSITIONS,'A136 scalar or intervention coverage differs')
        if arm=='DIAGONAL_K1':
            files.require(all(step['fraction']==1. for step in trace),'Expected full one-step update')
        result = bridge.output_metrics(out,shared.load_image(case['image']))
        torch.save(out,folder/'output.pt')
        files.write(folder/'refinement_trace.json',trace)
        row.update(status='complete',counts=counts,full_seconds=seconds,method_nfe=1000,**result,
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','refinement_trace.json')})
        print('A136 COMPLETE',case['key'],arm,row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as error:
        row.update(status='failed',error=repr(error),counts=counts)
        files.write(folder/'partial_refinement.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(rows):
    groups = []
    for task in transfer.TASKS:
        subset = [row for row in rows if row['task']==task]
        if len(subset)!=3:
            continue
        pairs = [bridge.paired(subset,candidate,ref) for candidate,ref in
                 [('DIAGONAL_K1','SCALAR'),('DIAGONAL_K12','SCALAR'),('DIAGONAL_K12','DIAGONAL_K1')]]
        arms = {arm:dict(quality={key:statistics.mean(row['arms'][arm]['quality'][key] for row in subset)
                                  for key in transfer.METRICS},
                         seconds=statistics.mean(row['arms'][arm]['full_seconds'] for row in subset),
                         nfe=1000,input_vjp=1000,timing_source='A133' if arm=='SCALAR' else 'A136')
                for arm in ('SCALAR', *CANDIDATES)}
        groups.append(dict(task=task, arms=arms, paired=pairs))
    return groups


def execute(a133_root,a135_root,source_root,root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    config,_,inputs,_ = dense.load_plan('dps',source_root)
    p133,p135 = a133_root/'summary.json',a135_root/'summary.json'
    a133,a135 = files.read(p133),files.read(p135)
    grid = list(range(999,-1,-1))
    files.require(a133['experiment']=='A133' and a133['host']=='dps' and a133['status']=='complete' and
                  a133['actual_nfe']==18000 and a133['sources'][str(source_root/'plan.json')]==
                  files.digest(source_root/'plan.json') and files.read(a133_root/'resolved_config.json')['grid']==grid,
                  'Compatible A133 native scalar controls are required')
    files.require(a135['experiment']=='A135' and a135['status']=='complete' and
                  a135['sources'][str(p133)]==files.digest(p133) and tuple(a135['positions'])==depth.POSITIONS and
                  a135['max_updates']==depth.MAX_UPDATES and a135['projected_gradient_rtol']==depth.GRAD_RTOL and
                  a135['replay_nfe']==6000 and a135['probe_nfe']==36 and len(a135['groups'])==6,
                  'Compatible completed A135 probe is required')
    gate = gaussian_gate(a135)
    scalar = {row['case']:row['arms']['SCALAR'] for row in a133['cases']}
    files.require(set(scalar)=={case['key'] for case,_,_ in inputs},'A133 case set differs')
    card = transfer.PROJECTS/'experiment_cards/A136_terminal_depth_ablation.md'
    sources = {str(p):files.digest(p) for p in (Path(__file__),card,p133,p135,source_root/'plan.json')}
    for case,_,_ in inputs:
        key = case['key']
        ref = a133_root/'cases'/key/'SCALAR'/'run.json'
        row = files.read(ref)
        files.require(row==scalar[key] and row['status']=='complete' and row['counts']['calls']==grid and
                      row['counts']['forwards']==row['counts']['input_vjp_attempts']==1000,
                      'Archived scalar differs')
        sources[str(ref)] = files.digest(ref)
    path = root/'summary.json'
    if path.exists():
        report = files.read(path)
        files.require(report['status']=='complete' and report['sources']==sources,
                      f'Existing partial or incompatible A136 run retained at {root}')
        for row in report['groups']:
            print('A136 SUMMARY',row,flush=True)
        return
    admission.check(torch,root)
    root.mkdir(parents=True)
    report = dict(experiment='A136',status='running',sources=sources,cases=[],groups=[],actual_new_nfe=0,
                  positions=depth.POSITIONS,max_updates=depth.MAX_UPDATES,grad_rtol=depth.GRAD_RTOL)
    files.write(path,report)
    files.write(root/'resolved_config.json',dict(sources=sources,grid=grid,positions=depth.POSITIONS,
                max_updates=depth.MAX_UPDATES,grad_rtol=depth.GRAD_RTOL,a135_gaussian_gate=gate,
                cases=[dict(case,observation=obs) for case,obs,_ in inputs]))
    started = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(config['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = np.load(config['motion_kernel'],allow_pickle=False)
        with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),native.dps_reference_scope():
            for index,(case,obs,ref) in enumerate(inputs):
                key = case['key']
                arms = {'SCALAR':scalar[key]}
                for arm in CANDIDATES if index%2==0 else reversed(CANDIDATES):
                    row = measure(root,arm,config,case,obs,ref,grid,model,wave,kernel)
                    arms[arm] = row
                    report['actual_new_nfe'] += row['counts']['forwards']
                    files.write(path,report)
                report['cases'].append(dict(case=key,task=case['task'],arms=arms))
                report['groups'] = summarize(report['cases'])
                files.write(path,report)
        files.require(report['actual_new_nfe']==12000 and len(report['groups'])==2,'Incomplete A136 panel')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),actual_new_nfe_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        files.write(path,report)
    for row in report['groups']:
        print('A136 SUMMARY',row,flush=True)
    print('A136 REPORT',path,'actual_new_nfe=',report['actual_new_nfe'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a133-root',type=Path,default=matched.PARENT)
    parser.add_argument('--a135-root',type=Path,default=depth.ROOT)
    parser.add_argument('--a124-root',type=Path,default=bridge.DPS_PARENT)
    parser.add_argument('--output',type=Path,default=ROOT)
    args = parser.parse_args()
    parents = (args.a133_root.resolve(),args.a135_root.resolve(),args.a124_root.resolve())
    root = args.output.resolve()
    files.require(all(root!=parent and not root.is_relative_to(parent) for parent in parents),
                  'Use a separate A136 output directory')
    execute(*parents,root)


if __name__ == '__main__':
    main()
