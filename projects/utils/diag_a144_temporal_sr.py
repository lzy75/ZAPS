"""A144: test time-consistent wavelet refinement in the exact SR4 nullspace."""
import argparse
from collections import deque
import math
from pathlib import Path
import statistics
import time

from utils import diag_a143_temporal_inpainting as temporal

parent, graph = temporal.parent, temporal.graph
admission, audit, files = temporal.admission, temporal.audit, temporal.files
bridge, clock, history, late, native, shared = (
    temporal.bridge, temporal.clock, temporal.history, temporal.late, temporal.native, temporal.shared
)
ROOT = parent.ROOT.with_name('diag_a144_temporal_sr')
HOST = 'pigdm'
TASK = 'super_resolution'
GRID = parent.GRIDS[HOST]
POSITIONS = tuple(t for t in GRID if t<=99)
WARM = temporal.buffer_positions(GRID,POSITIONS)
ARMS = ('ZERO_NULL','TEMPORAL_NULL')
CONTROLS = ('SCALAR','DIAGONAL')
HR_PER_LR = 16
REGION_KEYS = ('raw_mse','rowspace_mse','nullspace_mse','clean_observation_mse','noisy_observation_mse')
NETWORK_COUNTS = parent.NETWORK_COUNTS
OPS = ('graph_builds','graph_actions','refinement_H','refinement_Ht',
       'module_dwt_analysis','module_dwt_synthesis')


def null_projection(view, value):
    return value-HR_PER_LR*view.transpose(view.H(value))


def controller(operator, wave, arm, sigma, counts, trace):
    import torch

    files.require(arm in ARMS and operator.task==TASK,'A144 requires the native average-pool SR4 operator')
    view = native.LinearView(operator,counts)
    past = deque(maxlen=2)

    def remember(clean):
        past.append(clean.detach())

    def refine(q,residual,alpha,scalar,t,clean):
        scale = math.sqrt(alpha)
        anchor = scalar['gamma']*q
        observed_target = scalar['weight']*residual.double()-view.H(anchor/scale).double()
        files.require(len(past)==2,'Missing causal prediction history')
        if arm=='TEMPORAL_NULL':
            change = temporal.temporal_target(clean,past,'TEMPORAL')
            null_target = null_projection(view,change)
        else:
            null_target = torch.zeros_like(clean)
        remember(clean)
        support = wave.analysis(q)!=0
        weights = graph.edge_weights(clean,sigma)
        counts['module_dwt_analysis'] += 1
        counts['graph_builds'] += 1

        def project(value):
            counts['module_dwt_analysis'] += 1
            return wave.analysis(value)*support

        def synthesize(value):
            counts['module_dwt_synthesis'] += 1
            return wave.synthesis(value)

        def laplacian(value):
            counts['graph_actions'] += 1
            return graph.laplacian(value,weights)

        def apply(coefficients):
            value = synthesize(coefficients)
            # 16 H*H and its orthogonal complement sum to I.
            return project((1+graph.RIDGE)*value+temporal.STRENGTH*laplacian(value))

        rhs = project(HR_PER_LR*view.transpose(observed_target.to(q.dtype))+null_target)
        dot = temporal._dot
        coeffs,objective,relative,stop = graph.conjugate_gradient(
            apply,rhs,dot,late.depth.MAX_UPDATES,late.depth.GRAD_RTOL)
        increment = synthesize(coeffs) if len(objective)>1 else torch.zeros_like(q)
        hr_projection = view.H(increment)
        null_increment = increment-HR_PER_LR*view.transpose(hr_projection)
        obs_left = hr_projection.double()-observed_target
        null_left = null_increment-null_target
        before = HR_PER_LR*dot(observed_target,observed_target)+dot(null_target,null_target)
        after = HR_PER_LR*dot(obs_left,obs_left)+dot(null_left,null_left)
        gain = .5*(after-before+temporal.STRENGTH*dot(increment,laplacian(increment))+
                   graph.RIDGE*dot(increment,increment))
        files.require(math.isfinite(gain) and gain<=2e-5*max(1e-12,before),
                      'Applied SR refinement increased its frozen objective')
        trace.append(dict(t=t,updates=len(objective)-1,stop=stop,relative_gradient=relative,
                          objective_delta=gain,null_target_energy=dot(null_target,null_target),
                          null_increment_energy=dot(null_increment,null_increment)))
        return anchor+scale*increment

    return remember,refine


def load_inputs(source):
    config = files.read(source/'resolved_config.json')
    summary = files.read(source/'summary.json')
    files.require(config['experiment']==summary['experiment']=='A139' and
                  config['host']==summary['host']==HOST and config['dataset']==summary['dataset']=='ffhq' and
                  summary['status']=='complete' and config['grid']==GRID and tuple(config['positions'])==POSITIONS and
                  summary['planned_runs']==summary['completed_runs']==36 and
                  summary['actual_cost']['forwards']==summary['actual_cost']['input_vjp_attempts']==3600,
                  'Need a complete compatible A139 PiGDM')
    for name,digest in config['code_sha256'].items():
        files.require(files.digest(name)==digest,'A139 scientific source changed: '+name)
    checkpoint = config['checkpoint']
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Checkpoint changed')
    manifests = {row['case']['key']:row for row in config['cases'] if row['case']['task']==TASK}
    records = {row['case']:row['arms'] for row in summary['cases'] if row['task']==TASK}
    inputs = [entry for entry in history.load_sources(HOST,'ffhq') if entry['case']['task']==TASK]
    files.require(len(inputs)==3 and set(manifests)==set(records)=={e['case']['key'] for e in inputs},
                  'A139 SR panel differs')
    for entry in inputs:
        case = entry['case']
        key = case['key']
        original = manifests[key]
        obs = history.observation(entry,HOST)
        files.require(case==original['case'] and case['sigma']==.05 and entry['plan']['checkpoint']==checkpoint and
                      str(entry['root'].resolve())==original['source'] and
                      files.digest(entry['root']/'plan.json')==original['plan_sha256'] and
                      files.digest(case['image'])==original['image_sha256'] and
                      files.digest(obs['path'])==original['observation_sha256'],'A139 image or observation changed')
        entry['observation'] = obs
        entry['controls'] = {arm:(records[key][arm],source/'cases'/key/arm) for arm in CONTROLS}
    return inputs,checkpoint,manifests


def sample(entry, model, wave, arm, counts, trace, progress):
    obs = entry['observation']
    case = entry['case']
    y = audit.checked_tensor(obs['path'],model.device)
    operator = history.operator(entry,obs,HOST,model.device,counts)
    remember,refine = controller(operator,wave,arm,case['sigma'],counts,trace)
    run = temporal.instrument(history.native(entry,HOST),HOST,POSITIONS,WARM,remember,refine)
    return run('A144',model,operator,y,case,'SHRINK',counts,progress=progress)


def measure(root, entry, model, wave, arm, evaluate):
    import torch
    from utils.diag_pigdm_noisy_r1 import fingerprint

    key = entry['case']['key']
    folder = root/'cases'/key/arm
    folder.mkdir(parents=True)
    path = folder/'run.json'
    row = dict(status='running',case=key,arm=arm)
    files.write(path,row)
    trace,counts = [],{}
    reference = entry['controls']['SCALAR'][0]
    late_call = len(GRID)-len(POSITIONS)+1
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)

            def check_entry(_network,args):
                if len(counts['calls'])==late_call:
                    row['late_entry'] = fingerprint(args[0])
                    files.require(row['late_entry']==reference['late_entry'],'A139 scalar prefix differs')

            def progress(step,state):
                row.update(native_step=step,counts=dict(state))
                files.write(path,row)

            with model.model.register_forward_pre_hook(check_entry):
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:sample(entry,model,wave,arm,counts,trace,progress))
            pairing = history.fingerprint(detail,shared.rng(),counts,HOST)
        files.require(counts['forwards']==counts['forward_attempts']==counts['input_vjp_attempts']==len(GRID) and
                      counts['optimizer_backward_attempts']==counts.get('input_jvp_attempts',0)==0 and
                      pairing==reference['pairing'] and row['late_entry']==reference['late_entry'] and
                      GRID==detail['grid']==counts['calls'] and
                      [p['t'] for p in detail['trace'] if 'weight' in p]==GRID and
                      tuple(p['t'] for p in trace)==POSITIONS,'A144 pairing or budget differs')
        metrics = evaluate(out,True)
        torch.save(out,folder/'output.pt')
        files.write(folder/'trace.json',trace)
        row.update(status='complete',counts=counts,pairing=pairing,full_seconds=seconds,method_nfe=len(GRID),
                   **metrics,artifacts={name:files.digest(folder/name) for name in ('output.pt','trace.json')})
        print('A144 COMPLETE',key,arm,row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as exc:
        row.update(status='failed',error=repr(exc),counts=counts)
        files.write(folder/'partial_trace.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(rows):
    arms = {}
    for arm in (*CONTROLS,*ARMS):
        values = [row['arms'][arm] for row in rows]
        arms[arm] = dict(quality={k:statistics.mean(v['quality'][k] for v in values)
                                  for k in parent.transfer.METRICS},
                         regions={k:statistics.mean(v[k] for v in values) for k in REGION_KEYS},
                         seconds=statistics.mean(v['full_seconds'] for v in values),nfe=len(GRID),
                         input_vjp=len(GRID),inherited=arm in CONTROLS)
        if arm in ARMS:
            slopes = [v['null_initial_slope'] for v in values]
            arms[arm]['null_direction'] = dict(downhill_images=sum(s<0 for s in slopes),
                                                mean_initial_slope=statistics.mean(slopes))
    pairs = []
    for arm,ref in (('ZERO_NULL','SCALAR'),('TEMPORAL_NULL','SCALAR'),
                    ('TEMPORAL_NULL','ZERO_NULL'),('TEMPORAL_NULL','DIAGONAL')):
        row = bridge.paired(rows,arm,ref)
        row['region_delta'] = {k:arms[arm]['regions'][k]-arms[ref]['regions'][k] for k in REGION_KEYS}
        pairs.append(row)
    return dict(task=TASK,images=len(rows),arms=arms,paired=pairs)


def run_cases(root, report_path, report, inputs, checkpoint):
    import torch
    from modules.main_single import load_diffusion_model
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
    with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model):
        for index,entry in enumerate(inputs):
            case,obs = entry['case'],entry['observation']
            reference = entry['controls']['SCALAR'][0]
            truth = shared.load_image(case['image'])
            operator = history.operator(entry,obs,HOST,'cpu',None)
            noisy = audit.checked_tensor(obs['path'],'cpu')
            scalar,scalar_folder = entry['controls']['SCALAR']
            scalar_out = temporal.endpoint.read_output(scalar_folder,scalar,reference,HOST)
            scalar_error = (scalar_out-truth).double()
            scalar_null = scalar_error-HR_PER_LR*operator.Ht(operator.H(scalar_error))

            def evaluate(out, quality):
                error = (out-truth).double()
                high_row = HR_PER_LR*operator.Ht(operator.H(error))
                high_null = error-high_row
                raw,row,null = (audit.mse(v) for v in (error,high_row,high_null))
                files.require(abs(raw-row-null)<=1e-10,'SR error decomposition failed')
                regions = dict(raw_mse=raw,rowspace_mse=row,nullspace_mse=null,
                               clean_observation_mse=audit.mse(operator.H(out)-operator.H(truth)),
                               noisy_observation_mse=audit.mse(operator.H(out)-noisy))
                if quality:
                    regions.update(quality=bridge.output_metrics(out,truth)['quality'],
                                   null_initial_slope=2*float((scalar_null*(high_null-scalar_null)).mean()))
                return regions

            arms = {}
            for name,(row,folder) in entry['controls'].items():
                out = scalar_out if name=='SCALAR' else temporal.endpoint.read_output(folder,row,reference,HOST)
                regions = evaluate(out,False)
                files.require(abs(regions['raw_mse']-row['raw_mse'])<=1e-9 and
                              abs(regions['clean_observation_mse']-row['clean_observation_mse'])<=1e-8,
                              'Archived SR metric differs')
                arms[name] = dict(row,**regions)
            report['cases'].append(dict(case=case['key'],arms=arms))
            for arm in ARMS if index%2==0 else reversed(ARMS):
                result = measure(root,entry,model,wave,arm,evaluate)
                arms[arm] = result
                report['completed_new_runs'] += 1
                for key in report['actual_cost']:
                    report['actual_cost'][key] += result['counts'].get(key,0)
                files.write(report_path,report)


def print_report(report,path):
    print('A144 SUMMARY',HOST,report['group'],flush=True)
    print('A144 REPORT',path,'runs=',report['completed_new_runs'],'nfe=',report['actual_cost']['forwards'],flush=True)


def execute(source,root):
    import torch

    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Use a separate A144 output')
    inputs,checkpoint,manifests = load_inputs(source)
    card = parent.transfer.PROJECTS/'experiment_cards/A144_temporal_sr.md'
    code = (Path(__file__),Path(temporal.__file__),Path(graph.__file__),card,
            Path(__file__).with_name('test_a144_temporal_sr.py'))
    config = dict(experiment='A144',host=HOST,task=TASK,dataset='ffhq',source=str(source),
                  source_sha256={n:files.digest(source/n) for n in ('summary.json','resolved_config.json')},
                  code_sha256={str(p):files.digest(p) for p in code},checkpoint=checkpoint,inputs=manifests,
                  arms=list(ARMS),positions=list(POSITIONS),warm=list(WARM),graph_strength=temporal.STRENGTH,
                  ridge=graph.RIDGE,null_weight=1.,measurement_factor=HR_PER_LR,
                  max_updates=late.depth.MAX_UPDATES,grad_rtol=late.depth.GRAD_RTOL)
    path = root/'summary.json'
    if path.exists():
        previous = files.read(path)
        files.require(previous['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A144 retained at {root}')
        print_report(previous,path)
        return
    admission_result = admission.check(torch,root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json',config)
    report = dict(experiment='A144',host=HOST,status='running',resource_check=admission_result,
                  planned_new_runs=len(inputs)*len(ARMS),completed_new_runs=0,cases=[],
                  timing='A144 candidate clocks; A139 historical controls.',
                  actual_cost=dict.fromkeys((*NETWORK_COUNTS,*OPS),0),cost_is_lower_bound=False)
    files.write(path,report)
    try:
        run_cases(root,path,report,inputs,checkpoint)
        expected = report['planned_new_runs']*len(GRID)
        files.require(report['planned_new_runs']==report['completed_new_runs']==6 and
                      report['actual_cost']['forwards']==report['actual_cost']['input_vjp_attempts']==expected,
                      'A144 native budget incomplete')
        report.update(status='complete',group=summarize(report['cases']))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),cost_is_lower_bound=True)
        raise
    finally:
        files.write(path,report)
    print_report(report,path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=parent.ROOT/'A139_pigdm_ffhq')
    parser.add_argument('--output',type=Path,default=ROOT/'A144_pigdm_ffhq')
    args = parser.parse_args()
    execute(args.parent.resolve(),args.output.resolve())


if __name__=='__main__':
    main()
