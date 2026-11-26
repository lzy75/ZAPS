"""A141: scalar-initialized wavelet refinement with edge-aware mask coupling."""
import argparse
from collections import Counter
from contextlib import nullcontext
import math
from pathlib import Path
import statistics
import time

from utils import diag_a140_inpainting_refinement as previous
from utils.equal_nfe_feedback_runtime import _dot

parent = previous.parent
admission, audit, bridge = parent.admission, parent.audit, parent.bridge
clock, files, history = parent.clock, parent.files, parent.history
late, native, shared = parent.late, parent.native, parent.shared

ROOT = parent.ROOT.with_name('diag_a141_graph_inpainting')
STRENGTHS = {'GRAPH050': .5, 'GRAPH200': 2.}
RIDGE = .1
EDGE_SIGMAS = 2.


def edge_weights(clean, sigma):
    scale_squared = (EDGE_SIGMAS*sigma)**2
    east = (clean[..., :, 1:]-clean[..., :, :-1]).square().mean(dim=1, keepdim=True)
    south = (clean[..., 1:, :]-clean[..., :-1, :]).square().mean(dim=1, keepdim=True)
    return (1+east/scale_squared).reciprocal(), (1+south/scale_squared).reciprocal()


def laplacian(value, weights):
    east, south = weights
    horizontal = east*(value[..., :, :-1]-value[..., :, 1:])
    vertical = south*(value[..., :-1, :]-value[..., 1:, :])
    result = value.new_zeros(value.shape)
    result[..., :, :-1] += horizontal
    result[..., :, 1:] -= horizontal
    result[..., :-1, :] += vertical
    result[..., 1:, :] -= vertical
    return result


def conjugate_gradient(apply, rhs, dot, max_updates, rtol):
    """Solve the frozen SPD quadratic from the feasible zero increment."""
    files.require(max_updates>0 and rtol>0, 'Invalid CG stopping settings')
    solution = rhs*0
    residual = rhs
    direction = rhs
    energy = initial = dot(rhs, rhs)
    files.require(math.isfinite(initial) and initial>=0, 'Invalid initial CG residual')
    objective_path = [0.]
    if initial<=1e-24:
        return solution, objective_path, 0., 'projected_gradient_tolerance'
    for _ in range(max_updates):
        response = apply(direction)
        curvature = dot(direction, response)
        files.require(math.isfinite(curvature) and curvature>0, 'Graph normal operator lost positive curvature')
        step = energy/curvature
        solution = solution+step*direction
        residual = residual-step*response
        new_energy = dot(residual, residual)
        predicted = -.5*dot(solution, rhs+residual)
        files.require(math.isfinite(predicted) and math.isfinite(new_energy) and
                      predicted<=objective_path[-1]+1e-5*max(1e-12,abs(objective_path[-1])),
                      'Graph quadratic increased')
        objective_path.append(predicted)
        relative = math.sqrt(new_energy/initial)
        if relative<=rtol:
            return solution, objective_path, relative, 'projected_gradient_tolerance'
        direction = residual+(new_energy/energy)*direction
        energy = new_energy
    return solution, objective_path, relative, 'iteration_budget'


def controller(operator, wave, strength, sigma, count, trace):
    import torch

    physical = parent.ObservedResponse(operator)
    view = native.LinearView(physical, count)
    observed = None

    def refine(q, residual, alpha, scalar, t, clean):
        nonlocal observed
        scale = math.sqrt(alpha)
        anchor = scalar['gamma']*q
        response = view.H(anchor/scale)
        if observed is None:
            observed = view.transpose(torch.ones_like(response)).bool()
        error = scalar['weight']*residual.double().reshape_as(response)-response.double()
        support = wave.analysis(q)!=0
        count['module_dwt_analysis'] += 1
        weights = edge_weights(clean, sigma)
        count['graph_builds'] += 1

        def project(value):
            count['module_dwt_analysis'] += 1
            return wave.analysis(value)*support

        def synthesis(coefficients):
            count['module_dwt_synthesis'] += 1
            return wave.synthesis(coefficients)

        def graph(value):
            count['graph_actions'] += 1
            return laplacian(value, weights)

        def normal(coefficients):
            value = synthesis(coefficients)
            # For the binary inpainting restriction, P*P is the image-space mask.
            return project(observed*value+strength*graph(value)+RIDGE*value)

        rhs = project(view.transpose(error.to(q.dtype)))
        coefficients, path, relative, stop = conjugate_gradient(
            normal, rhs, _dot, late.depth.MAX_UPDATES, late.depth.GRAD_RTOL)
        increment = synthesis(coefficients) if len(path)>1 else torch.zeros_like(q)
        remaining = view.H(increment).double()-error
        gain = .5*(_dot(remaining,remaining)-_dot(error,error))+.5*(
            strength*_dot(increment,graph(increment))+RIDGE*_dot(increment,increment))
        # Audit the actually applied field, independently of the CG recurrence.
        files.require(math.isfinite(gain) and gain<=2e-5*max(1e-12,_dot(error,error)),
                      'Applied graph refinement increased the frozen objective')
        total = _dot(increment,increment)
        missing = _dot(increment[~observed],increment[~observed])
        trace.append(dict(t=t,updates=len(path)-1,stop=stop,relative_gradient=relative,
                          objective_delta=gain,missing_energy=missing,total_energy=total))
        return anchor+scale*increment

    return refine


def instrument(runtime, host, positions, refine):
    if host=='pigdm':
        original = "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
        replacement = (
            "   delta=refine(q,r,a,row,t,m.detach()) if t in positions else row['gamma']*q\n"
            "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    else:
        original = "            result=x_t-grad*scale if gamma==1. else x_t-gamma*scale*grad"
        replacement = (
            "            if t in positions:\n"
            "                with torch.no_grad():\n"
            "                    result=x_t+refine(q,residual,a*a,row,t,clean)\n"
            "            else:\n"+original.replace("            ", "                ", 1))
    return audit.insert_probe(runtime.run, [(original,replacement)], positions=set(positions), refine=refine)


def sample(entry, host, model, wave, arm, positions, counts, trace, progress):
    y = audit.checked_tensor(entry['observation']['path'],model.device)
    operator = history.operator(entry,entry['observation'],host,model.device,counts)
    case,plan = entry['case'],entry['plan']
    refine = controller(operator,wave,STRENGTHS[arm],case['sigma'],counts,trace)
    run = instrument(history.native(entry,host),host,positions,refine)
    if host=='pigdm':
        return run('A141',model,operator,y,case,'SHRINK',counts,progress=progress)
    return run(model,operator,y,case,'SHRINK',plan['diffusion_config'],plan['dps_root'],counts,progress=progress)


def measure(root, entry, host, model, wave, arm, positions, evaluate):
    import torch
    from utils.diag_pigdm_noisy_r1 import fingerprint

    case = entry['case']['key']
    folder = root/'cases'/case/arm
    folder.mkdir(parents=True)
    path = folder/'run.json'
    row = dict(status='running',case=case,arm=arm)
    files.write(path,row)
    trace,counts = [],{}
    scalar = entry['controls']['SCALAR'][0]
    grid = parent.GRIDS[host]
    entry_call = len(grid)-len(positions)+1
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)

            def check_entry(_network, args):
                if len(counts['calls'])==entry_call:
                    row['late_entry'] = fingerprint(args[0])
                    files.require(row['late_entry']==scalar['late_entry'],'A139 scalar prefix differs')

            def progress(step, state):
                row.update(native_step=step,counts=dict(state))
                files.write(path,row)

            with model.model.register_forward_pre_hook(check_entry):
                (output,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:sample(entry,host,model,wave,arm,positions,counts,trace,progress))
            pairing = history.fingerprint(detail,shared.rng(),counts,host)
        files.require(counts['forward_attempts']==counts['forwards']==counts['input_vjp_attempts']==len(grid) and
                      counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0 and
                      grid==detail['grid']==counts['calls'] and pairing==scalar['pairing'] and
                      row['late_entry']==scalar['late_entry'] and tuple(p['t'] for p in trace)==positions and
                      [p['t'] for p in detail['trace'] if 'weight' in p]==grid,'A141 pairing or budget differs')
        started = time.perf_counter()
        metrics = evaluate(output)
        metrics_seconds = time.perf_counter()-started
        torch.save(output,folder/'output.pt')
        files.write(folder/'trace.json',trace)
        total = sum(step['total_energy'] for step in trace)
        solver = dict(updates=sum(step['updates'] for step in trace),
                      stops=dict(Counter(step['stop'] for step in trace)),
                      missing_energy_share=sum(step['missing_energy'] for step in trace)/total if total else 0.)
        row.update(status='complete',counts=counts,pairing=pairing,full_seconds=seconds,
                   method_nfe=len(grid),metrics_seconds=metrics_seconds,solver=solver,**metrics,
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','trace.json')})
        print('A141 COMPLETE',host,case,arm,row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as exc:
        row.update(status='failed',error=repr(exc),counts=counts)
        files.write(folder/'partial_trace.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(cases):
    arms = {}
    keys = previous.REGION_KEYS
    for name in (*previous.CONTROLS,*STRENGTHS):
        rows = [item['arms'][name] for item in cases]
        arms[name] = dict(quality={k:statistics.mean(r['quality'][k] for r in rows) for k in parent.transfer.METRICS},
                          regions={k:statistics.mean(r[k] for r in rows) for k in keys},
                          full_seconds=statistics.mean(r['full_seconds'] for r in rows),nfe=rows[0]['method_nfe'],
                          input_vjp=rows[0]['counts']['input_vjp_attempts'],inherited=name in previous.CONTROLS)
        if name in STRENGTHS:
            arms[name]['solver'] = {k:statistics.mean(r['solver'][k] for r in rows) for k in
                                    ('updates','missing_energy_share')}
    pairs = []
    for name in STRENGTHS:
        for control in previous.CONTROLS:
            comparison = bridge.paired(cases,name,control)
            comparison['region_delta'] = {k:arms[name]['regions'][k]-arms[control]['regions'][k] for k in keys}
            pairs.append(comparison)
    return dict(task='inpainting',images=len(cases),arms=arms,paired=pairs)


def run_cases(report, path, inputs, host, root, checkpoint, positions):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    started = time.perf_counter()
    model = load_diffusion_model('ffhq','cuda:0')
    model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
    model.model.eval()
    wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
    torch.cuda.synchronize()
    report['model_load_seconds'] = time.perf_counter()-started
    scope = native.dps_reference_scope() if host=='dps' else nullcontext()
    with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),scope:
        for index,entry in enumerate(inputs):
            case,obs = entry['case'],entry['observation']
            truth = (shared.load_image if host=='pigdm' else load_image_as_tensor)(case['image'])
            truth_device = truth.to(model.device)
            physical = parent.ObservedResponse(history.operator(entry,obs,host,model.device,None))
            y = audit.checked_tensor(obs['path'],model.device)

            def evaluate(output):
                with torch.no_grad():
                    regions = previous.region_metrics(physical,output.to(model.device),truth_device,y)
                return dict(quality=bridge.output_metrics(output,truth)['quality'],**regions)

            arms = {}
            for name,(row,folder) in entry['controls'].items():
                output = audit.checked_tensor(folder/'output.pt','cpu')
                with torch.no_grad():
                    regions = previous.region_metrics(physical,output.to(model.device),truth_device,y)
                files.require(abs(regions['observed_mse']-row['clean_observation_mse'])<=1e-9 and
                              abs(regions['raw_mse']-row['raw_mse'])<=1e-9,'Archived region evaluation differs')
                arms[name] = dict(row,**regions)
            report['cases'].append(dict(case=case['key'],arms=arms))
            order = tuple(STRENGTHS)
            for arm in order if index%2==0 else reversed(order):
                row = measure(root,entry,host,model,wave,arm,positions,evaluate)
                arms[arm] = row
                report['completed_new_runs'] += 1
                for key in report['actual_cost']:
                    report['actual_cost'][key] += row['counts'].get(key,0)
                files.write(path,report)


def print_report(report, path):
    print('A141 SUMMARY',report['host'],report['group'],flush=True)
    print('A141 REPORT',path,'runs=',report['completed_new_runs'],'nfe=',report['actual_cost']['forwards'],flush=True)


def execute(host, source, root):
    import torch

    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Use a separate A141 output')
    inputs,checkpoint,positions,manifest = previous.load_inputs(host,source)
    card = parent.transfer.PROJECTS/'experiment_cards/A141_graph_inpainting.md'
    sources = (Path(__file__),Path(previous.__file__),card,Path(__file__).with_name('test_a141_graph_inpainting.py'))
    config = dict(experiment='A141',host=host,dataset='ffhq',task='inpainting',source=str(source),
                  source_sha256={name:files.digest(source/name) for name in ('summary.json','resolved_config.json')},
                  code_sha256={str(p):files.digest(p) for p in sources},inputs=manifest,checkpoint=checkpoint,
                  positions=list(positions),strengths=STRENGTHS,ridge=RIDGE,edge_sigmas=EDGE_SIGMAS,
                  max_updates=late.depth.MAX_UPDATES,grad_rtol=late.depth.GRAD_RTOL)
    path = root/'summary.json'
    if path.exists():
        prior = files.read(path)
        files.require(prior['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A141 retained at {root}')
        print_report(prior,path)
        return
    resources = admission.check(torch,root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json',config)
    report = dict(experiment='A141',host=host,status='running',resource_check=resources,cases=[],
                  planned_new_runs=len(inputs)*len(STRENGTHS),completed_new_runs=0,cost_is_lower_bound=False,
                  timing='Candidate clocks from A141; scalar and original diagonal clocks inherited from A139.',
                  actual_cost=dict.fromkeys((*parent.NETWORK_COUNTS,'graph_builds','graph_actions',
                                            'refinement_H','refinement_Ht','module_dwt_analysis','module_dwt_synthesis'),0))
    files.write(path,report)
    try:
        run_cases(report,path,inputs,host,root,checkpoint,positions)
        files.require(report['completed_new_runs']==report['planned_new_runs'] and
                      report['actual_cost']['forwards']==report['planned_new_runs']*len(parent.GRIDS[host]),
                      'A141 budget incomplete')
        report.update(status='complete',group=summarize(report['cases']))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),cost_is_lower_bound=True)
        raise
    finally:
        files.write(path,report)
    print_report(report,path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=tuple(parent.TASKS))
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    source = args.parent or parent.ROOT/f'A139_{args.host}_ffhq'
    root = args.output or ROOT/f'A141_{args.host}_ffhq'
    execute(args.host,source.resolve(),root.resolve())


if __name__=='__main__':
    main()
