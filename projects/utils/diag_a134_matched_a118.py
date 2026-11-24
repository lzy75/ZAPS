"""A134: keep full-native scalar DPS and apply the A118 step only at its 30 positions."""
import argparse
from pathlib import Path
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a118_transfer_bridge as bridge
from utils import diag_a133_native_grid_feedback as dense
from utils import diag_transfer_objective_audit as audit
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = dense.ROOT.with_name('diag_a134_matched_a118_dps')
PARENT = dense.ROOT/'A133_dps_ffhq'
ARM = 'MATCHED_A118_DIAGONAL'


def instrument(runtime, positions, refine):
    name = 'rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
    original = f'result=x_t-grad*{name} if gamma==1. else x_t-gamma*{name}*grad'
    return audit.insert_probe(runtime.run, [
        ('            '+original,
         '            if t in positions:\n'
         '                with torch.no_grad():\n'
         '                    result=x_t+refine(op,q,residual,a*a,row,t)\n'
         '            else:\n'
         '                '+original)], positions=set(positions),refine=refine)


def measure(root, case, obs, source, config, grid, positions, model, wave, kernel):
    import torch

    folder = root/'cases'/case['key']
    folder.mkdir(parents=True)
    path = folder/'run.json'
    row = dict(status='running', case=case['key'], task=case['task'], arm=ARM)
    counts, refinement = {}, []
    files.write(path,row)
    try:
        runtime = transfer.history.native(dict(case=case),'dps')
        settings = config['dps_configs'][case['task']]
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)

            def progress(step,state):
                row.update(native_step=step,counts=dict(state))
                files.write(path,row)

            def operation():
                y = audit.checked_tensor(obs['path'],model.device)
                op = transfer.make_operator(config,case,model.device,kernel,counts)
                refine = transfer.runtime.refinement(wave,'A118_DIAGONAL',counts,refinement)
                run = instrument(runtime,positions,refine)
                return run(model,op,y,case,'SHRINK',settings['diffusion_config'],
                           settings['dps_root'],counts,progress=progress)

            (out,detail),seconds = clock.clocks(torch.cuda.synchronize,operation)
            identity = transfer.history.fingerprint(detail,shared.rng(),counts,'dps')
        expected = dict(source['pairing'])
        expected.pop('feedback_entry_sha256',None)
        files.require(identity == expected and grid == detail['grid'] == counts['calls'] and
                      counts['forwards'] == counts['forward_attempts'] == counts['input_vjp_attempts'] == 1000 and
                      counts.get('input_jvp_attempts',0) == counts['optimizer_backward_attempts'] == 0,
                      'A134 native grid, budget or RNG differs')
        files.require([step['t'] for step in detail['trace'] if 'weight' in step] == grid and
                      [step['t'] for step in refinement] == positions and
                      all(step['fraction'] == 1. for step in refinement),
                      'A134 requires full scalar and the exact A118 interventions')
        torch.save(out, folder/'output.pt')
        files.write(folder/'refinement_trace.json',refinement)
        row.update(status='complete',counts=counts,full_seconds=seconds,method_nfe=1000,
                   **bridge.output_metrics(out,shared.load_image(case['image'])))
        print('A134 COMPLETE',case['key'],row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as exc:
        row.update(status='failed',error=repr(exc),counts=counts)
        files.write(folder/'partial_trace.json',refinement)
        raise
    finally:
        files.write(path,row)


def execute(parent, source_root, root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    config, positions, inputs, _ = dense.load_plan('dps',source_root)
    previous_path = parent/'summary.json'
    previous = files.read(previous_path)
    resolved = files.read(parent/'resolved_config.json')
    old_cases = {row['case']:row for row in previous['cases']}
    grid = list(range(999,-1,-1))
    files.require(previous['experiment']=='A133' and previous['host']=='dps' and previous['status']=='complete' and
                  previous['actual_nfe']==18000 and resolved['grid']==grid and resolved['a118_grid']==positions and
                  set(old_cases)=={case['key'] for case,_,_ in inputs} and
                  previous['sources'][str(source_root/'plan.json')]==files.digest(source_root/'plan.json'),
                  'Completed matching A133 DPS controls required')
    checks = {}
    for case,_,_ in inputs:
        key = case['key']
        for arm in ('BASE','SCALAR','DIAGONAL'):
            path = parent/'cases'/key/arm/'run.json'
            row = files.read(path)
            files.require(row == old_cases[key]['arms'][arm] and row['status']=='complete' and
                          row['counts']['forwards']==row['counts']['input_vjp_attempts']==1000 and
                          row['counts']['calls']==grid,'A133 reference case differs')
            checks[str(path)] = files.digest(path)
    path = root/'summary.json'
    if path.exists():
        completed = files.read(path)
        files.require(completed['status']=='complete',f'Existing partial audit retained at {root}')
        for group in completed['groups']:
            print('A134 SUMMARY',group,flush=True)
        return
    admission.check(torch,root)
    root.mkdir(parents=True)
    card = transfer.PROJECTS/'experiment_cards/A134_matched_a118_dps.md'
    files.write(root/'resolved_config.json',dict(grid=grid,active_positions=positions,
                source_hashes={str(p):files.digest(p) for p in
                    (Path(__file__),card,previous_path,parent/'resolved_config.json')},controls=checks,
                cases=[dict(case,observation=obs) for case,obs,_ in inputs]))
    report = dict(experiment='A134',status='running',actual_new_nfe=0,cases=[],groups=[])
    files.write(path,report)
    try:
        started = time.perf_counter()
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(config['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = np.load(config['motion_kernel'],allow_pickle=False)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-started
        with precision('LEGACY_IEEE'), frozen_network(model.model), input_vjp_parameters(model.model), native.dps_reference_scope():
            for case,obs,source in inputs:
                row = measure(root,case,obs,source,config,grid,positions,model,wave,kernel)
                arms = dict(old_cases[case['key']]['arms'],**{ARM:row})
                report['cases'].append(dict(case=case['key'],task=case['task'],arms=arms))
                report['actual_new_nfe'] += row['counts']['forwards']
                files.write(path,report)
        files.require(report['actual_new_nfe']==6000,'Incomplete A134 panel')
        for task in transfer.TASKS:
            matched = [row for row in report['cases'] if row['task']==task]
            group = dict(task=task, images=len(matched), paired=[bridge.paired(matched,ARM,reference)
                         for reference in ('SCALAR','DIAGONAL','BASE')],
                         timing_source={ARM:'A134', 'SCALAR':'A133', 'DIAGONAL':'A133', 'BASE':'A133'})
            group['arms'] = {arm:dict(
                quality={key:sum(row['arms'][arm]['quality'][key] for row in matched)/len(matched)
                         for key in transfer.METRICS},
                seconds=sum(row['arms'][arm]['full_seconds'] for row in matched)/len(matched),
                nfe=1000,input_vjp=1000) for arm in matched[0]['arms']}
            report['groups'].append(group)
            print('A134 SUMMARY',group,flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),actual_nfe_is_lower_bound=True)
        raise
    finally:
        files.write(path,report)
    print('A134 REPORT',path,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a133-root',type=Path,default=PARENT)
    parser.add_argument('--a124-root',type=Path,default=bridge.DPS_PARENT)
    parser.add_argument('--output',type=Path,default=ROOT)
    args = parser.parse_args()
    root = args.output.resolve()
    parents = (args.a133_root.resolve(),args.a124_root.resolve())
    files.require(all(root != parent and not root.is_relative_to(parent) for parent in parents),
                  'Choose a separate output directory')
    execute(*parents,root)


if __name__=='__main__':
    main()
