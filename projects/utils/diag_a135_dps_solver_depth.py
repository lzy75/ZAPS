"""A135: frozen-state DPS scalar, one-step A118, and deeper quadratic solve."""
import argparse
import math
from pathlib import Path
import statistics
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a133_native_grid_feedback as dense
from utils import diag_a134_matched_a118 as matched
from utils import diag_transfer_objective_audit as audit
from utils import matched_response_transfer_runtime as response
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = dense.ROOT.with_name('diag_a135_dps_solver_depth')
POSITIONS = (95, 24, 0)
MAX_UPDATES = 12
GRAD_RTOL = 1e-2


def instrument(runtime, capture):
    scale = 'rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
    expression = f'            result=x_t-grad*{scale} if gamma==1. else x_t-gamma*{scale}*grad'
    return audit.insert_probe(runtime.run, [
        (expression, expression+'\n'
         '            if t in selected_states:\n'
         '                capture(t,index-1,sampler,x_t,result,q,residual,a*a,row)')],
        selected_states=set(POSITIONS), capture=capture)


def solve_depth(wave, operator, entry, counts):
    """Use the exact A134 update first, then repeat its same frozen quadratic."""
    from utils.equal_nfe_feedback_runtime import _dot

    q, residual, alpha, row = (entry[key] for key in ('q','residual','alpha','scalar'))
    trace = []
    one = transfer.runtime.refinement(wave,'A118_DIAGONAL',counts,trace)(
        operator,q,residual,alpha,row,entry['t'])
    first = trace[0]
    anchor = row['gamma']*q
    view = native.LinearView(operator,counts)
    scale = math.sqrt(alpha)
    mask = wave.analysis(q) != 0
    counts['module_dwt_analysis'] += 1

    def proxy_error(delta):
        value = view.H(delta/scale)
        return value.double()-row['weight']*residual.double().reshape_as(value)

    def gradient(error):
        pulled = view.transpose(error.to(q.dtype))/scale
        counts['module_adjoint_actions'] += 1
        counts['module_dwt_analysis'] += 1
        return wave.analysis(pulled)*mask

    projected = gradient(proxy_error(anchor))
    g0 = math.sqrt(max(0.,_dot(projected,projected)))
    delta = one
    err = proxy_error(delta)
    losses = [first['proxy_initial'], .5*_dot(err,err)]
    files.require(math.isclose(losses[-1],first['proxy_final'],rel_tol=2e-5,abs_tol=2e-5),
                  'A134 line response differs from its directly evaluated objective')
    updates = int(first['proposed_extra_norm'] > 0)
    stop = 'iteration_budget'
    relative_gradient = 0.
    for _ in range(MAX_UPDATES+1):
        g = gradient(err)
        energy = _dot(g,g)
        files.require(math.isfinite(energy), 'Nonfinite projected gradient')
        relative_gradient = math.sqrt(max(0.,energy))/max(g0,1e-30)
        if g0 <= 1e-24 or relative_gradient <= GRAD_RTOL:
            stop = 'projected_gradient_tolerance'
            break
        if updates >= MAX_UPDATES:
            break
        direction = wave.synthesis(g)
        counts['module_dwt_synthesis'] += 1
        response_vector = view.H(direction/scale).double()
        curvature = _dot(response_vector,response_vector)
        files.require(math.isfinite(curvature) and curvature > 0, 'Unresolved projected curvature')
        delta = delta-(energy/curvature)*direction
        err = proxy_error(delta)
        loss = .5*_dot(err,err)
        files.require(math.isfinite(loss) and loss <= losses[-1]+2e-5*max(1.,losses[-1]),
                      'Frozen objective increased')
        losses.append(loss)
        updates += 1
    return one,delta,dict(updates=updates, stop=stop, relative_gradient=relative_gradient,
                          proxy=dict(scalar=losses[0],one=losses[1],deep=losses[-1]),loss_history=losses)


def replay(model, config, case, obs, kernel, grid, baseline, archived_path):
    import torch

    rt = transfer.history.native(dict(case=case),'dps')
    settings = config['dps_configs'][case['task']]
    states = []

    def capture(t,next_index,sampler,origin,scalar_state,q,residual,alpha,scalar):
        states.append(dict(t=t,next_index=next_index,sampler=sampler,origin=origin.detach().clone(),
                           scalar_state=scalar_state.detach().clone(),q=q.detach().clone(),
                           residual=residual.detach().clone(),alpha=alpha,scalar=dict(scalar)))

    started = time.perf_counter()
    with clock.counters(model.model) as count:
        native.initialize_counts(count)
        operator = transfer.make_operator(config,case,model.device,kernel,count)
        y = audit.checked_tensor(obs['path'],model.device)
        run = instrument(rt,capture)
        out, detail = run(model,operator,y,case,'SHRINK',settings['diffusion_config'],settings['dps_root'],count)
        identity = transfer.history.fingerprint(detail,shared.rng(),count,'dps')
    expected = dict(baseline['pairing'])
    expected.pop('feedback_entry_sha256',None)
    files.require(expected==identity and detail['grid']==count['calls']==grid and
                  count['forwards']==count['forward_attempts']==count['input_vjp_attempts']==1000 and
                  count['optimizer_backward_attempts']==count.get('input_jvp_attempts',0)==0 and
                  [e['t'] for e in states]==list(POSITIONS),'A135 scalar trajectory identity differs')
    archived = audit.checked_tensor(archived_path,'cpu')
    relative = audit.norm(out-archived)/max(audit.norm(archived),1e-30)
    files.require(relative<=1e-4 and torch.equal(states[-1]['scalar_state'].cpu(),out),
                  f'A133 scalar replay or terminal output differs: {relative:.3g}')
    return states,dict(counts=count,relative_l2=relative,seconds=time.perf_counter()-started)


def assess(model, config, case, obs, kernel, wave, states):
    import torch

    truth = shared.load_image(case['image'],model.device)
    rows = []
    with torch.no_grad(), clock.counters(model.model) as count:
        native.initialize_counts(count)
        op = transfer.make_operator(config,case,model.device,kernel,count)
        observed_truth = op.H(truth)
        y = audit.checked_tensor(obs['path'],model.device)
        for state in states:
            one,deep,detail = solve_depth(wave,op,state,count)
            sampler, index = state['sampler'], state['next_index']
            clean = {
                'scalar':response.dps_predict(sampler,model,state['scalar_state'],index),
                'one':response.dps_predict(sampler,model,state['origin']+one,index),
                'deep':response.dps_predict(sampler,model,state['origin']+deep,index),
            }
            quality = {}
            for name,estimate in clean.items():
                observed = op.H(estimate)
                quality[name] = dict(image_mse=audit.mse(estimate-truth),
                                     clean_observation_mse=audit.mse(observed-observed_truth),
                                     noisy_observation_mse=audit.mse(observed-y))
            rows.append(dict(t=state['t'], next_t=sampler.timestep_map[index] if index>=0 else -1,
                             quality=quality, **detail))
    files.require(count['forwards']==count['forward_attempts']==6 and
                  count['input_vjp_attempts']==count.get('input_jvp_attempts',0)==
                  count['optimizer_backward_attempts']==0,'Offline next-prediction budget differs')
    return rows,count


def summary(rows):
    tables = []
    for task in transfer.TASKS:
        for t in POSITIONS:
            selected = [info for row in rows if row['task']==task for info in row['states'] if info['t']==t]
            if len(selected)!=3:
                continue
            pairs = []
            for candidate, ref in (('one','scalar'),('deep','one'),('deep','scalar')):
                fields = ('image_mse','clean_observation_mse','noisy_observation_mse')
                errors = {key:[row['quality'][candidate][key]-row['quality'][ref][key]
                               for row in selected] for key in fields}
                objective = [row['proxy'][candidate]-row['proxy'][ref] for row in selected]
                objective_gain = [(-delta/max(row['proxy'][ref],1e-30)) for delta,row in zip(objective,selected)]
                pairs.append(dict(pair=candidate+'_minus_'+ref,
                                  errors={k:statistics.mean(values) for k,values in errors.items()},
                                  proxy_delta=statistics.mean(objective),
                                  median_relative_proxy_gain=statistics.median(objective_gain),
                                  image_mse_wins=sum(delta<0 for delta in errors['image_mse']),
                                  image_mse_relative_change=statistics.mean(
                                      delta/max(row['quality'][ref]['image_mse'],1e-30)
                                      for delta,row in zip(errors['image_mse'],selected))))
            tables.append(dict(task=task,t=t,images=3,comparisons=pairs,
                               updates=[row['updates'] for row in selected],
                               stops=[row['stop'] for row in selected]))
    return tables


def execute(a133_root,dps_root,root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    config, _, cases, _ = dense.load_plan('dps',dps_root)
    reference_path = a133_root/'summary.json'
    reference = files.read(reference_path)
    grid = list(range(999,-1,-1))
    files.require(reference['experiment']=='A133' and reference['status']=='complete' and
                  reference['host']=='dps' and reference['actual_nfe']==18000 and
                  reference['sources'][str(dps_root/'plan.json')]==files.digest(dps_root/'plan.json') and
                  files.read(a133_root/'resolved_config.json')['grid']==grid,
                  'Completed compatible native A133 DPS required')
    scalars = {row['case']:row['arms']['SCALAR'] for row in reference['cases']}
    files.require(set(scalars)=={case['key'] for case,_,_ in cases},'A133 scalar cases differ')
    card = transfer.PROJECTS/'experiment_cards/A135_dps_solver_depth.md'
    sources = {str(path):files.digest(path) for path in (Path(__file__),card,reference_path,dps_root/'plan.json')}
    archives = {}
    for case,_,_ in cases:
        folder = a133_root/'cases'/case['key']/'SCALAR'
        run_path = folder/'run.json'
        row = files.read(run_path)
        files.require(row==scalars[case['key']] and row['status']=='complete' and
                      row['counts']['calls']==grid,'A133 scalar record changed')
        saved = folder/'output.pt'
        archives[case['key']] = saved
        sources[str(run_path)] = files.digest(run_path)
        sources[str(saved)] = files.digest(saved)
    path = root/'summary.json'
    if path.exists():
        report = files.read(path)
        files.require(report['status']=='complete' and report['sources']==sources,
                      f'Existing partial or different result retained at {root}')
        for row in report['groups']:
            print('A135 SUMMARY',row,flush=True)
        print('A135 REPORT',path,flush=True)
        return
    admission.check(torch,root)
    root.mkdir(parents=True)
    report = dict(experiment='A135',status='running',sources=sources,cases=[],groups=[],
                  positions=POSITIONS,max_updates=MAX_UPDATES,projected_gradient_rtol=GRAD_RTOL,
                  replay_nfe=0,probe_nfe=0)
    files.write(path,report)
    start = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(config['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = np.load(config['motion_kernel'],allow_pickle=False)
        with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),native.dps_reference_scope():
            for case,obs,baseline in cases:
                key = case['key']
                states,replay_info = replay(model,config,case,obs,kernel,grid,baseline,archives[key])
                report['replay_nfe'] += replay_info['counts']['forwards']
                files.write(path,report)
                rows,counts = assess(model,config,case,obs,kernel,wave,states)
                report['probe_nfe'] += counts['forwards']
                report['cases'].append(dict(case=key,task=case['task'],replay=replay_info,states=rows,probe_counts=counts))
                report['groups'] = summary(report['cases'])
                files.write(path,report)
                print('A135 COMPLETE',key,[(row['t'],row['updates'],row['stop']) for row in rows],flush=True)
        files.require(report['replay_nfe']==6000 and report['probe_nfe']==36 and len(report['groups'])==6,
                      'Incomplete diagnostic budget')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),actual_nfe_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-start
        files.write(path,report)
    for row in report['groups']:
        print('A135 SUMMARY',row,flush=True)
    print('A135 REPORT',path,'replay_nfe=',report['replay_nfe'],'probe_nfe=',report['probe_nfe'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a133-root',type=Path,default=matched.PARENT)
    parser.add_argument('--a124-root',type=Path,default=matched.bridge.DPS_PARENT)
    parser.add_argument('--output',type=Path,default=ROOT)
    args = parser.parse_args()
    parents = (args.a133_root.resolve(),args.a124_root.resolve())
    root = args.output.resolve()
    files.require(all(root != parent and not root.is_relative_to(parent) for parent in parents),
                  'Choose a separate A135 result directory')
    execute(*parents,root)


if __name__ == '__main__':
    main()
