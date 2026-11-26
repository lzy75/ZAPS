"""A143: test a causal clean-prediction target for the missing-region increment."""
import argparse
from collections import Counter, deque
from contextlib import nullcontext
import math
from pathlib import Path
import statistics
import time

from utils import diag_a141_graph_inpainting as graph
from utils import diag_a142_inpainting_direction as endpoint
from utils.equal_nfe_feedback_runtime import _dot

parent, previous = graph.parent, graph.previous
admission, audit, bridge = graph.admission, graph.audit, graph.bridge
clock, files, history = graph.clock, graph.files, graph.history
late, native, shared = graph.late, graph.native, graph.shared
ROOT = graph.ROOT.with_name('diag_a143_temporal_inpainting')
ARMS = ('ZERO_MISSING', 'TEMPORAL')
STRENGTH = graph.STRENGTHS['GRAPH200']
MISSING_WEIGHT = 1.


def temporal_target(clean, earlier, arm):
    import torch

    files.require(len(earlier)==2,'Two earlier clean predictions are required')
    if arm=='ZERO_MISSING':
        return torch.zeros_like(clean)
    return torch.stack((*earlier,clean)).median(dim=0).values-clean


def buffer_positions(grid, active):
    index = grid.index(active[0])
    files.require(index>=2 and tuple(grid[index:])==tuple(active),'Invalid native final window')
    return tuple(grid[index-2:index])


def controller(operator, wave, arm, sigma, count, trace):
    import torch

    files.require(arm in ARMS,'Unknown A143 arm')
    view = native.LinearView(parent.ObservedResponse(operator),count)
    earlier = deque(maxlen=2)
    observed = None

    def remember(clean):
        earlier.append(clean.detach())

    def refine(q, residual, alpha, scalar, t, clean):
        nonlocal observed
        scale = math.sqrt(alpha)
        anchor = scalar['gamma']*q
        response = view.H(anchor/scale)
        if observed is None:
            observed = view.transpose(torch.ones_like(response)).bool()
        missing = ~observed
        observed_error = scalar['weight']*residual.double().reshape_as(response)-response.double()
        missing_error = temporal_target(clean,earlier,arm)*missing
        remember(clean)
        support = wave.analysis(q)!=0
        count['module_dwt_analysis'] += 1
        weights = graph.edge_weights(clean,sigma)
        count['graph_builds'] += 1

        def project(value):
            count['module_dwt_analysis'] += 1
            return wave.analysis(value)*support

        def synthesis(coefficients):
            count['module_dwt_synthesis'] += 1
            return wave.synthesis(coefficients)

        def laplacian(value):
            count['graph_actions'] += 1
            return graph.laplacian(value,weights)

        def normal(coefficients):
            value = synthesis(coefficients)
            return project(observed*value+MISSING_WEIGHT*missing*value+
                           STRENGTH*laplacian(value)+graph.RIDGE*value)

        rhs = project(view.transpose(observed_error.to(q.dtype))+MISSING_WEIGHT*missing_error)
        coefficients,path,relative,stop = graph.conjugate_gradient(
            normal,rhs,_dot,late.depth.MAX_UPDATES,late.depth.GRAD_RTOL)
        increment = synthesis(coefficients) if len(path)>1 else torch.zeros_like(q)
        observed_left = view.H(increment).double()-observed_error
        missing_left = missing*increment-missing_error
        missing_energy = _dot(missing_error,missing_error)
        before = _dot(observed_error,observed_error)+MISSING_WEIGHT*missing_energy
        after = _dot(observed_left,observed_left)+MISSING_WEIGHT*_dot(missing_left,missing_left)
        gain = .5*(after-before+STRENGTH*_dot(increment,laplacian(increment))+
                   graph.RIDGE*_dot(increment,increment))
        files.require(math.isfinite(gain) and gain<=2e-5*max(1e-12,before),
                      'Applied temporal refinement increased the frozen objective')
        trace.append(dict(t=t,updates=len(path)-1,stop=stop,relative_gradient=relative,
                          objective_delta=gain,missing_target_energy=missing_energy,
                          missing_increment_energy=_dot(increment[missing],increment[missing])))
        return anchor+scale*increment

    return remember,refine


def instrument(runtime, host, positions, warm, remember, refine):
    if host=='pigdm':
        original = "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
        replacement = (
            "   if t in warm:remember(m)\n"
            "   delta=refine(q,r,a,row,t,m.detach()) if t in positions else row['gamma']*q\n"
            "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    else:
        original = "            result=x_t-grad*scale if gamma==1. else x_t-gamma*scale*grad"
        replacement = (
            "            if t in warm:remember(clean)\n"
            "            if t in positions:\n"
            "                with torch.no_grad():\n"
            "                    result=x_t+refine(q,residual,a*a,row,t,clean)\n"
            "            else:\n"+original.replace("            ", "                ", 1))
    return audit.insert_probe(runtime.run, [(original,replacement)], positions=set(positions), warm=set(warm),
                              remember=remember, refine=refine)


def sample(entry, host, model, wave, arm, positions, counts, trace, progress):
    y = audit.checked_tensor(entry['observation']['path'],model.device)
    operator = history.operator(entry,entry['observation'],host,model.device,counts)
    case,plan = entry['case'],entry['plan']
    warm = buffer_positions(parent.GRIDS[host],positions)
    remember,refine = controller(operator,wave,arm,case['sigma'],counts,trace)
    run = instrument(history.native(entry,host),host,positions,warm,remember,refine)
    if host=='pigdm':
        return run('A143',model,operator,y,case,'SHRINK',counts,progress=progress)
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
                      [p['t'] for p in detail['trace'] if 'weight' in p]==grid,'A143 pairing or budget differs')
        started = time.perf_counter()
        metrics = evaluate(output)
        metrics_seconds = time.perf_counter()-started
        torch.save(output,folder/'output.pt')
        files.write(folder/'trace.json',trace)
        solver = dict(updates=sum(step['updates'] for step in trace),
                      stops=dict(Counter(step['stop'] for step in trace)),
                      missing_target_energy=sum(step['missing_target_energy'] for step in trace),
                      missing_increment_energy=sum(step['missing_increment_energy'] for step in trace))
        row.update(status='complete',counts=counts,pairing=pairing,full_seconds=seconds,
                   method_nfe=len(grid),metrics_seconds=metrics_seconds,solver=solver,**metrics,
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','trace.json')})
        print('A143 COMPLETE',host,case,arm,row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as exc:
        row.update(status='failed',error=repr(exc),counts=counts)
        files.write(folder/'partial_trace.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(cases):
    arms = {}
    regions = previous.REGION_KEYS
    for arm in (*previous.CONTROLS,*ARMS):
        rows = [case['arms'][arm] for case in cases]
        arms[arm] = dict(
            quality={key:statistics.mean(r['quality'][key] for r in rows) for key in parent.transfer.METRICS},
            regions={key:statistics.mean(r[key] for r in rows) for key in regions},
            full_seconds=statistics.mean(r['full_seconds'] for r in rows), nfe=rows[0]['method_nfe'],
            input_vjp=rows[0]['counts']['input_vjp_attempts'], inherited=arm in previous.CONTROLS)
    pairs = []
    for candidate,reference in (('ZERO_MISSING','SCALAR'),('TEMPORAL','SCALAR'),('TEMPORAL','ZERO_MISSING')):
        pair = bridge.paired(cases,candidate,reference)
        pair['region_delta'] = {key:arms[candidate]['regions'][key]-arms[reference]['regions'][key] for key in regions}
        pairs.append(pair)
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for region in ('full','observed','missing','noisy_observation'):
            directions = [case['directions'][arm][region] for case in cases]
            groups[arm][region] = dict(
                downhill_images=sum(row['initial_slope']<0 for row in directions),
                mean_initial_slope=statistics.mean(row['initial_slope'] for row in directions),
                mean_end_delta=statistics.mean(row['end_delta'] for row in directions))
    return dict(images=len(cases),arms=arms,paired=pairs,endpoint_direction=groups)


def run_cases(report, path, inputs, host, root, checkpoint, positions, manifest):
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
            key = case['key']
            truth = (shared.load_image if host=='pigdm' else load_image_as_tensor)(case['image'])
            mask,noisy = endpoint.load_mask(manifest[key],host,truth)
            arms, directions = {},{}
            scalar_row,scalar_folder = entry['controls']['SCALAR']
            scalar = audit.checked_tensor(audit.qualified_artifact(scalar_folder,'output.pt',scalar_row),'cpu')

            def evaluate(output):
                details = endpoint.direction(scalar,output,truth,mask,noisy)
                error = (output-truth).double().square()
                residual = (output[mask]-noisy).double()
                alignment = ((output-truth).double()[mask]*(noisy-truth[mask]).double()).mean()
                return dict(quality=bridge.output_metrics(output,truth)['quality'],raw_mse=float(error.mean()),
                            observed_mse=float(error[mask].mean()),missing_mse=float(error[~mask].mean()),
                            noisy_observation_mse=float(residual.square().mean()),noise_alignment=float(alignment),
                            endpoint_direction=details)

            for name,(row,folder) in entry['controls'].items():
                output = audit.checked_tensor(folder/'output.pt','cpu')
                endpoint.verify_endpoint(output,truth,mask,row)
                error = (output-truth).double().square()
                arms[name] = dict(row,observed_mse=float(error[mask].mean()),missing_mse=float(error[~mask].mean()),
                                  noise_alignment=float(((output-truth).double()[mask]*
                                                         (noisy-truth[mask]).double()).mean()))
            report['cases'].append(dict(case=key,arms=arms,directions=directions))
            for arm in ARMS if index%2==0 else reversed(ARMS):
                row = measure(root,entry,host,model,wave,arm,positions,evaluate)
                arms[arm] = row
                directions[arm] = row['endpoint_direction']
                report['completed_new_runs'] += 1
                for field in report['actual_cost']:
                    report['actual_cost'][field] += row['counts'].get(field,0)
                files.write(path,report)


def print_report(report, path):
    print('A143 SUMMARY',report['host'],report['group'],flush=True)
    print('A143 REPORT',path,'runs=',report['completed_new_runs'],'nfe=',report['actual_cost']['forwards'],flush=True)


def execute(host, source, root):
    import torch

    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Use a separate A143 output')
    inputs,checkpoint,positions,manifest = previous.load_inputs(host,source)
    card = parent.transfer.PROJECTS/'experiment_cards/A143_temporal_inpainting.md'
    sources = (Path(__file__),Path(graph.__file__),Path(endpoint.__file__),Path(previous.__file__),card,
               Path(__file__).with_name('test_a143_temporal_inpainting.py'))
    config = dict(experiment='A143',host=host,dataset='ffhq',task='inpainting',source=str(source),
                  source_sha256={name:files.digest(source/name) for name in ('summary.json','resolved_config.json')},
                  code_sha256={str(p):files.digest(p) for p in sources},inputs=manifest,checkpoint=checkpoint,
                  positions=list(positions),warm=list(buffer_positions(parent.GRIDS[host],positions)),
                  arms=list(ARMS),graph_strength=STRENGTH,ridge=graph.RIDGE,missing_weight=MISSING_WEIGHT,
                  max_updates=late.depth.MAX_UPDATES,grad_rtol=late.depth.GRAD_RTOL)
    path = root/'summary.json'
    if path.exists():
        prior = files.read(path)
        files.require(prior['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A143 retained at {root}')
        print_report(prior,path)
        return
    resources = admission.check(torch,root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json',config)
    report = dict(experiment='A143',host=host,status='running',resource_check=resources,cases=[],
                  planned_new_runs=len(inputs)*len(ARMS),completed_new_runs=0,cost_is_lower_bound=False,
                  timing='Same A143 session for two candidates. A139 control clocks inherited.',
                  actual_cost=dict.fromkeys((*parent.NETWORK_COUNTS,'graph_builds','graph_actions',
                                            'refinement_H','refinement_Ht','module_dwt_analysis','module_dwt_synthesis'),0))
    files.write(path,report)
    try:
        run_cases(report,path,inputs,host,root,checkpoint,positions,manifest)
        files.require(report['completed_new_runs']==report['planned_new_runs'] and
                      report['actual_cost']['forwards']==report['planned_new_runs']*len(parent.GRIDS[host]),
                      'A143 budget incomplete')
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
    root = args.output or ROOT/f'A143_{args.host}_ffhq'
    execute(args.host,source.resolve(),root.resolve())


if __name__=='__main__':
    main()
