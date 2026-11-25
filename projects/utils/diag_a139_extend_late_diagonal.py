"""A139: extend frozen late diagonal feedback to more FFHQ tasks and PiGDM."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import statistics
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a118_transfer_bridge as bridge
from utils import diag_a137_last100_depth as late
from utils import diag_complete_diagonal_transfer as history
from utils import diag_transfer_objective_audit as audit
from utils import native_ray_transfer_runtime as shared
from utils import paper_clock_alignment as files
from utils import paper_clock_runtime as clock
from utils import transfer_diagonal_runtime as native

ROOT = history.BASE/'diag_a139_extend_late_diagonal'
ARMS = ('BASE','SCALAR','DIAGONAL')
TASKS = {'dps':('super_resolution','inpainting'), 'pigdm':history.TASKS}
GRIDS = {'dps':list(range(999,-1,-1)), 'pigdm':list(range(990,-1,-10))}
NETWORK_COUNTS = ('forward_attempts','forwards','input_vjp_attempts',
                  'input_jvp_attempts','optimizer_backward_attempts')


class ObservedResponse:
    """P H and the actual adjoint H* P*: used only in the diagonal objective."""

    def __init__(self, operator):
        self.operator = operator
        self.mask = getattr(operator,'_mask_cache',None)
        self.selected = None

    def H(self, correction):
        mapped = self.operator.H(correction)
        if self.mask is None:
            return mapped
        if self.selected is None:
            self.selected = self.mask.bool().expand_as(mapped)
        return mapped[self.selected]

    def Ht(self, observation):
        if self.mask is not None:
            import torch

            files.require(self.selected is not None,'Observed response shape is not initialized')
            embedded = torch.zeros(self.selected.shape,dtype=observation.dtype,device=observation.device)
            embedded[self.selected] = observation.reshape(-1)
            observation = embedded
        if hasattr(self.operator,'Ht'):
            return self.operator.Ht(observation)
        return getattr(self.operator,'base',self.operator).transpose(observation)


def check_adjoint(entry, obs, host, model):
    import torch

    with torch.no_grad():
        operator = ObservedResponse(history.operator(entry,obs,host,model.device,None))
        generator = torch.Generator(device=model.device).manual_seed(139000)
        errors = []
        for _ in range(2):
            x = torch.randn((1,3,256,256),device=model.device,generator=generator)
            image = operator.H(x)
            y = torch.randn(image.shape,device=model.device,generator=generator)
            back = operator.Ht(y)
            lhs = float((image.double()*y.double()).sum())
            rhs = float((x.double()*back.double()).sum())
            error = abs(lhs-rhs)/max(1.,abs(lhs),abs(rhs))
            files.require(error<=2e-5,'Physical response and adjoint do not match')
            errors.append(error)
    return errors


def pigdm_sampler(runtime, positions, refine):
    expression = "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
    return audit.insert_probe(runtime.run, [(expression,
        "   delta=refine(op,q,r,a,row,t) if t in positions else row['gamma']*q\n"
        "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")],
        positions=set(positions),refine=refine)


def run_sampling(entry, obs, host, model, wave, arm, positions, count, trace, progress):
    y = audit.checked_tensor(obs['path'],model.device)
    op = history.operator(entry,obs,host,model.device,count)
    runtime = history.native(entry,host)
    sampler = runtime.run
    if arm=='DIAGONAL':
        physical = ObservedResponse(op)
        solve = late.terminal.controller('DIAGONAL_K12',wave,count,trace)

        def refine(_op,q,residual,alpha,scalar,t):
            return solve(physical,q,residual,alpha,scalar,t)

        if host=='pigdm':
            sampler = pigdm_sampler(runtime,positions,refine)
        else:
            sampler = late.matched.instrument(runtime,positions,refine)
    case,plan = entry['case'],entry['plan']
    if host=='pigdm':
        return sampler('A139',model,op,y,case,'BASE' if arm=='BASE' else 'SHRINK',count,progress=progress)
    return sampler(model,op,y,case,'DPS' if arm=='BASE' else 'SHRINK',
                   plan['diffusion_config'],plan['dps_root'],count,progress=progress)


def measure(root, entry, obs, host, model, wave, arm, positions, reference):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_pigdm_noisy_r1 import fingerprint

    case = entry['case']
    folder = root/'cases'/case['key']/arm
    folder.mkdir(parents=True)
    path = folder/'run.json'
    row = dict(status='running',case=case['key'],task=case['task'],arm=arm)
    trace,counts = [],{}
    files.write(path,row)
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            entry_step = len(GRIDS[host])-len(positions)+1

            def capture_entry(_network, args):
                if len(counts['calls']) == entry_step:
                    row['late_entry'] = fingerprint(args[0])

            def progress(step,state):
                row.update(native_step=step,counts=dict(state))
                files.write(path,row)

            with model.model.register_forward_pre_hook(capture_entry):
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:run_sampling(entry,obs,host,model,wave,arm,positions,counts,trace,progress))
            pairing = history.fingerprint(detail,shared.rng(),counts,host)
        grid = GRIDS[host]
        files.require(counts['forwards']==counts['forward_attempts']==counts['input_vjp_attempts']==len(grid)
                      and counts['calls']==detail['grid']==grid and
                      counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0,
                      'A139 native network budget differs')
        files.require([step['t'] for step in detail['trace'] if 'weight' in step]==
                      ([] if arm=='BASE' else grid) and
                      tuple(step['t'] for step in trace)==(positions if arm=='DIAGONAL' else ()),
                      'A139 scalar and late-direction coverage differs')
        if reference:
            row['archived_relative_l2'] = bridge.archived_error(out,reference[1],reference[0],'output.pt')
        metrics_started = time.perf_counter()
        truth_loader = shared.load_image if host=='pigdm' else load_image_as_tensor
        truth = truth_loader(case['image'])
        metrics = bridge.output_metrics(out,truth)
        with torch.no_grad():
            physical = ObservedResponse(history.operator(entry,obs,host,model.device,None))
            estimate = physical.H(out.to(model.device))
            noisy = audit.checked_tensor(obs['path'],model.device)
            if physical.selected is not None:
                noisy = noisy[physical.selected]
            metrics.update(clean_observation_mse=audit.mse(estimate-physical.H(truth.to(model.device))),
                           noisy_observation_mse=audit.mse(estimate-noisy))
        row['metrics_seconds'] = time.perf_counter()-metrics_started
        artifacts_started = time.perf_counter()
        torch.save(out,folder/'output.pt')
        files.write(folder/'refinement_trace.json',trace)
        row.update(status='complete',counts=counts,pairing=pairing,full_seconds=seconds,method_nfe=len(grid),
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','refinement_trace.json')},**metrics)
        row['artifact_seconds'] = time.perf_counter()-artifacts_started
        if trace:
            row['solver'] = late.solver_stats('DIAGONAL_K12',trace)
        print('A139 COMPLETE',host,case['key'],arm,row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as error:
        row.update(status='failed',error=repr(error),counts=counts)
        files.write(folder/'partial_refinement.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(rows, host):
    output = []
    for task in TASKS[host]:
        subset = [row for row in rows if row['task']==task]
        if len(subset)!=3:
            continue
        arms = {}
        for arm in ARMS:
            records = [row['arms'][arm] for row in subset]
            arms[arm] = dict(
                quality={key:statistics.mean(r['quality'][key] for r in records) for key in transfer.METRICS},
                errors={key:statistics.mean(r[key] for r in records) for key in
                        ('raw_mse','clean_observation_mse','noisy_observation_mse')},
                seconds=statistics.mean(r['full_seconds'] for r in records),
                nfe=len(GRIDS[host]),input_vjp=len(GRIDS[host]))
        solvers = [row['arms']['DIAGONAL']['solver'] for row in subset]
        output.append(dict(task=task,images=len(subset),arms=arms,
                           paired=[bridge.paired(subset,candidate,reference) for candidate,reference in
                                   (('SCALAR','BASE'),('DIAGONAL','SCALAR'),('DIAGONAL','BASE'))],
                           mean_diagonal_updates=statistics.mean(s['total_updates'] for s in solvers),
                           diagonal_clock_ratio=arms['DIAGONAL']['seconds']/arms['SCALAR']['seconds']))
    return output


def load_inputs(host, root):
    entries = [entry for entry in history.load_sources(host,'ffhq') if entry['case']['task'] in TASKS[host]]
    checkpoint = entries[0]['plan']['checkpoint']
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'], 'FFHQ checkpoint changed')
    manifests = []
    for entry in entries:
        plan,case,parent = entry['plan'],entry['case'],entry['root'].resolve()
        files.require(not root.is_relative_to(parent) and not parent.is_relative_to(root),
                      'Use a separate A139 output directory')
        files.require(case['sigma']==.05 and plan['checkpoint']==checkpoint and
                      files.digest(case['image'])==plan['inputs'][case['image']], 'FFHQ input identity differs')
        entry['observation'] = history.observation(entry,host)
        entry['references'] = {name:history.existing(entry,source,host) for name,source in
                               (('BASE','BASE'),('SCALAR','SHRINK'))}
        manifests.append(dict(case=case,source=str(parent),plan_sha256=files.digest(parent/'plan.json'),
                              image_sha256=plan['inputs'][case['image']],
                              observation_sha256=files.digest(entry['observation']['path']),
                              references={name:files.digest(folder/'run.json')
                                          for name,(_,folder) in entry['references'].items()}))
    files.require(len({entry['case']['key'] for entry in entries})==3*len(TASKS[host]),'Duplicate source cases')
    return entries,checkpoint,manifests


def execute(host, root):
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    entries,checkpoint,inputs = load_inputs(host,root)
    grid = GRIDS[host]
    positions = tuple(t for t in grid if t<=late.POSITIONS[0])
    card = transfer.PROJECTS/'experiment_cards/A139_extend_late_diagonal.md'
    sources = {str(p):files.digest(p) for p in (Path(__file__),card,Path(late.__file__),
               Path(late.depth.__file__),Path(late.terminal.__file__),Path(transfer.runtime.__file__),
               Path(late.matched.__file__),Path(history.__file__),Path(native.__file__))}
    config = dict(experiment='A139',host=host,dataset='ffhq',checkpoint=checkpoint,grid=grid,
                  positions=list(positions),max_updates=late.depth.MAX_UPDATES,grad_rtol=late.depth.GRAD_RTOL,
                  cases=inputs,code_sha256=sources)
    path = root/'summary.json'
    if path.exists():
        prior = files.read(path)
        files.require(prior['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A139 run retained at {root}')
        print_summary(prior,path)
        return
    limits = admission.check(torch,root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json',config)
    report = dict(experiment='A139',host=host,dataset='ffhq',status='running',planned_runs=len(entries)*len(ARMS),
                  cases=[],groups=[],completed_runs=0,resource_check=limits,
                  actual_cost=dict.fromkeys(NETWORK_COUNTS,0),cost_is_lower_bound=False,
                  timing='Full same-session native loops. Model, operator adjoint checks, metrics and writes separate.')
    files.write(path,report)
    started = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-started
        reference_scope = native.dps_reference_scope() if host=='dps' else nullcontext()
        with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),reference_scope:
            for index,entry in enumerate(entries):
                case,obs = entry['case'],entry['observation']
                gate_start = time.perf_counter()
                gate = check_adjoint(entry,obs,host,model)
                report_case = dict(case=case['key'],task=case['task'],adjoint_relative_errors=gate,
                                   adjoint_check_seconds=time.perf_counter()-gate_start,arms={})
                shift = index % len(ARMS)
                for arm in ARMS[shift:]+ARMS[:shift]:
                    row = measure(root,entry,obs,host,model,wave,arm,positions,entry['references'].get(arm))
                    report_case['arms'][arm] = row
                    for key in NETWORK_COUNTS:
                        report['actual_cost'][key] += row['counts'].get(key,0)
                    report['completed_runs'] += 1
                    files.write(path,report)
                base,scalar,diagonal = (report_case['arms'][arm] for arm in ARMS)
                files.require(base['pairing']==scalar['pairing']==diagonal['pairing'] and
                              scalar['late_entry']==diagonal['late_entry'], 'A139 pairing or late-entry state differs')
                report['cases'].append(report_case)
                report['groups'] = summarize(report['cases'],host)
                files.write(path,report)
        files.require(report['completed_runs']==report['planned_runs'] and
                      report['actual_cost']['forwards']==report['planned_runs']*len(grid) and
                      len(report['groups'])==len(TASKS[host]), 'Incomplete A139 comparison')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),cost_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        files.write(path,report)
    print_summary(report,path)


def print_summary(report, path):
    for row in report['groups']:
        print('A139 SUMMARY',report['host'],row,flush=True)
    print('A139 REPORT',path,'runs=',report['completed_runs'],'nfe=',report['actual_cost']['forwards'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=tuple(TASKS))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    root = args.output or ROOT/f'A139_{args.host}_ffhq'
    execute(args.host,root.resolve())


if __name__=='__main__':
    main()
