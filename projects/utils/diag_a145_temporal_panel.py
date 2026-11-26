"""A145: native-budget FFHQ matrix for temporal diagonal transfer."""
import argparse
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from utils import a145_temporal_geometry as method
from utils import diag_a143_temporal_inpainting as temporal
from utils import shared_prefix_feedback_runtime as zaps

parent, endpoint = temporal.parent, temporal.endpoint
files, audit, clock, history, native, shared = (
    temporal.files, temporal.audit, temporal.clock, temporal.history, temporal.native, temporal.shared
)
bridge, admission = temporal.bridge, temporal.admission
ROOT = parent.ROOT.with_name('diag_a145_temporal_panel')
TASKS = ('inpainting','super_resolution','gaussian_deblur','motion_deblur')
HOSTS = ('zaps','dps','pigdm')
CONTROLS = ('SCALAR','DIAGONAL')
ARMS = (*CONTROLS,*method.ARMS)
ZAPS_ARMS = dict(SCALAR='CLOSED_SCALAR_SUFFIX',DIAGONAL='CLOSED_DIAG_SUFFIX')
ZAPS_IMAGES = ('00001','00004','00008')
OPS = ('graph_builds','graph_actions','refinement_H','refinement_Ht',
       'module_dwt_analysis','module_dwt_synthesis','geometry_setups','sr_separable_lifts')
COSTS = (*parent.NETWORK_COUNTS,*OPS)
REUSED = {
    ('dps','inpainting'): (temporal.ROOT/'A143_dps_ffhq','TEMPORAL','ZERO_MISSING'),
    ('pigdm','inpainting'): (temporal.ROOT/'A143_pigdm_ffhq','TEMPORAL','ZERO_MISSING'),
    ('pigdm','super_resolution'): (ROOT.with_name('diag_a144_temporal_sr')/'A144_pigdm_ffhq','TEMPORAL_NULL','ZERO_NULL'),
}


def schedule(host):
    grid = files.grid()[::-1] if host=='zaps' else parent.GRIDS[host]
    active = tuple(t for t in grid if t<=99)
    return grid,active,temporal.buffer_positions(grid,active)


def native_sources(host,task):
    grid,active,_ = schedule(host)
    source = parent.ROOT/f'A139_{host}_ffhq'
    plan_path = source/'resolved_config.json'
    summary_path = source/'summary.json'
    plan,summary = files.read(plan_path),files.read(summary_path)
    files.require(plan['experiment']==summary['experiment']=='A139' and plan['host']==summary['host']==host and
                  plan['dataset']==summary['dataset']=='ffhq' and summary['status']=='complete' and
                  plan['grid']==grid and plan['positions']==list(active),'Need compatible A139 archive')
    cases = [entry for entry in history.load_sources(host,'ffhq') if entry['case']['task']==task]
    archived = {r['case']:r['arms'] for r in summary['cases'] if r['task']==task}
    manifests = {r['case']['key']:r for r in plan['cases'] if r['case']['task']==task}
    frozen = [plan_path,summary_path]
    for entry in cases:
        case = entry['case']
        key = case['key']
        obs = history.observation(entry,host)
        entry['observation'] = obs
        source_plan = entry['root']/'plan.json'
        archive = archived.get(key)
        checkpoint = plan['checkpoint']
        image_hash = entry['plan']['inputs'][case['image']]
        files.require(case['sigma']==.05 and entry['plan']['checkpoint']==checkpoint and
                      files.digest(case['image'])==image_hash,'Native source changed')
        if archive is not None:
            manifest = manifests[key]
            files.require(case==manifest['case'] and image_hash==manifest['image_sha256'] and
                          files.digest(source_plan)==manifest['plan_sha256'] and
                          files.digest(obs['path'])==manifest['observation_sha256'],'A139 case identity differs')
            entry['controls'] = {arm:(archive[arm],source/'cases'/key/arm) for arm in CONTROLS}
            for arm,(row,folder) in entry['controls'].items():
                files.require(row==files.read(folder/'run.json') and row['status']=='complete',
                              'A139 control record changed')
                audit.qualified_artifact(folder,'output.pt',row)
                frozen.append(folder/'run.json')
        else:
            files.require(host=='dps' and task in ('gaussian_deblur','motion_deblur'),
                          'Missing archive for expected reused arm')
            entry['controls'] = {}
            entry['legacy_scalar'] = history.existing(entry,'SHRINK',host)
            frozen.append(entry['legacy_scalar'][1]/'run.json')
        frozen.extend((source_plan,Path(obs['path']),Path(case['image'])))
    files.require(len(cases)==3,'Expected three native cases')
    return cases,plan['checkpoint'],None,frozen


def zaps_sources(task,source):
    plan_path,summary_path = source/'plan.json',source/'summary.json'
    plan,summary = files.read(plan_path),files.read(summary_path)
    files.require(plan['experiment']==summary['experiment']==zaps.rules.EXPERIMENT and
                  summary['completed_cases']==summary['planned_cases']==40 and plan['grid']==files.grid(),
                  'Need a complete compatible ZAPS A102 archive')
    kernel = source/'motion_kernel.npy'
    files.require(files.digest(kernel)==plan['motion_kernel_sha256'],'A102 motion kernel changed')
    cases = []
    frozen = [plan_path,summary_path,kernel]
    for case in plan['cases']:
        if case['task']!=task or case['image_id'] not in ZAPS_IMAGES:
            continue
        key = case['key']
        directory = source/'cases'/key
        row = files.read(directory/'run.json')
        files.require(row['status']=='complete' and row['case']==case and
                      row['plan_sha256']==files.digest(plan_path),'A102 control identity differs')
        obs_path = source/'observations'/key/'observation.json'
        obs = files.read(obs_path)
        files.require(case['sigma']==.05 and files.digest(obs['path'])==obs['sha256'] and
                      files.digest(case['image'])==case['image_sha256'],'A102 inputs differ')
        controls = {}
        for arm,name in ZAPS_ARMS.items():
            record = row['arms'][name]
            audit.qualified_artifact(directory,name+'.pt',record)
            controls[arm] = (record,directory)
        frozen.extend((directory/'run.json',obs_path,Path(obs['path']),Path(case['image'])))
        saved = dict(case=case,observation=obs,controls=controls)
        if task=='inpainting':
            saved['mask'] = obs_path.with_name('mask.pt')
            frozen.append(saved['mask'])
        cases.append(saved)
    files.require(len(cases)==3,'Expected three fixed ZAPS images per task')
    return cases,plan['checkpoint'],kernel,frozen


def evaluate(out,truth,y,geometry,quality=True):
    error = out-truth
    seen = geometry.observable(error)
    weak = error-seen
    dot = method._dot
    raw = audit.mse(error)
    obs_mse,weak_mse = dot(error,seen)/error.numel(),dot(error,weak)/error.numel()
    files.require(min(obs_mse,weak_mse)>=-1e-10 and abs(raw-obs_mse-weak_mse)<=1e-8,
                  'Invalid observable/weak error decomposition')
    image = geometry.view.H(out)
    result = dict(raw_mse=raw,observable_mse=obs_mse,weak_mse=weak_mse,
                  clean_observation_mse=audit.mse(image-geometry.view.H(truth)),
                  noisy_observation_mse=audit.mse(image-y))
    if quality:
        result.update(quality=bridge.output_metrics(out,truth)['quality'])
    return result


def count_budget(count,nfe,backward,vjp,grid):
    files.require(count['forwards']==count['forward_attempts']==nfe and
                  count['optimizer_backward_attempts']==count['optimizer_backward_completed']==backward and
                  count['input_vjp_attempts']==vjp and count.get('input_jvp_attempts',0)==0 and
                  count['calls']==grid,'A145 native budget or grid mismatch')


def native_instrument(runtime,host,active,warm,remember,refine):
    if not runtime.__name__.endswith('dps_multiplier_transfer_runtime'):
        return temporal.instrument(runtime,host,active,warm,remember,refine)
    original = '            result=x_t-grad*rules.SCALE if gamma==1. else x_t-gamma*rules.SCALE*grad'
    new = (
        '            if t in warm:remember(clean)\n'
        '            if t in positions:\n'
        '                with torch.no_grad():\n'
        '                    result=x_t+refine(q,residual,a*a,row,t,clean)\n'
        '            else:\n'+original.replace('            ','                ',1))
    return audit.insert_probe(runtime.run,[(original,new)],positions=set(active),warm=set(warm),
                              remember=remember,refine=refine)


def native_sample(entry,host,model,wave,arm,active,warm,counts,trace):
    obs,case,plan = entry['observation'],entry['case'],entry['plan']
    if arm in CONTROLS:
        return parent.run_sampling(entry,obs,host,model,wave,arm,active,counts,trace,None)
    y = audit.checked_tensor(obs['path'],model.device)
    op = history.operator(entry,obs,host,model.device,counts)
    # Build once per reconstruction; the spatial operator is fixed.
    import torch
    geom = method.Geometry(case['task'],op,torch.zeros((1,3,256,256),device=model.device),counts)
    remember,refine = method.controller(geom,wave,arm,case['sigma'],counts,trace)
    run = native_instrument(history.native(entry,host),host,active,warm,remember,refine)
    if host=='pigdm':
        return run('A145',model,op,y,case,'SHRINK',counts)
    return run(model,op,y,case,'SHRINK',plan['diffusion_config'],plan['dps_root'],counts)


def zaps_sampler(operator,wave,arm,sigma,active,warm,geom,counts,trace):
    remember,refine = method.controller(geom,wave,arm,sigma,counts,trace)
    original = '            delta = row[\'gamma\']*q; detail = dict(status=\'CLOSED_SCALAR\', scalar=row)'
    new = (
        '            if t in warm:remember(clean)\n'
        '            if t in active:\n'
        '                with torch.no_grad():\n'
        '                    altered=refine(q.detach(),r,float(ab[t]),row,t,clean.detach())\n'
        '                delta=altered+(q-q.detach())\n'
        '            else:\n'
        '                delta=row[\'gamma\']*q\n'
        '            detail = dict(status=\'A145_TEMPORAL\', scalar=row)')
    return audit.insert_probe(zaps._custom_suffix,[(original,new)],warm=set(warm),active=set(active),
                              remember=remember,refine=refine)


def save_candidate(root,key,arm,out,record,trace):
    import torch

    folder = root/'cases'/key/arm
    folder.mkdir(parents=True)
    torch.save(out,folder/'output.pt')
    files.write(folder/'trace.json',trace)
    record.update(artifacts={name:files.digest(folder/name) for name in ('output.pt','trace.json')})
    files.write(folder/'run.json',record)


def evaluate_on_device(out, truth, y, geometry, device, quality=True):
    import torch

    with torch.no_grad():
        metrics = evaluate(out.to(device),truth.to(device),y,geometry,False)
    if quality:
        metrics.update(quality=bridge.output_metrics(out,truth)['quality'])
    return metrics


def native_case(root,entry,host,model,wave,grid,active,warm):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_pigdm_noisy_r1 import fingerprint

    case,obs = entry['case'],entry['observation']
    key = case['key']
    device = model.device
    truth = (shared.load_image if host=='pigdm' else load_image_as_tensor)(case['image'])
    op = history.operator(entry,obs,host,device,None)
    y = audit.checked_tensor(obs['path'],device)
    eval_counts = {}
    native.initialize_counts(eval_counts)
    with torch.no_grad():
        geometry = method.Geometry(case['task'],op,truth.to(device),eval_counts)
        gate = geometry.check(truth.to(device))
        if host=='dps' and case['task']=='inpainting':
            geometry.view.H(truth.to(device))
            y = y[geometry.view.operator.selected]
    arms,actual = {},Counter()
    for name,(row,folder) in entry['controls'].items():
        output = endpoint.read_output(folder,row,entry['controls']['SCALAR'][0],host)
        regions = evaluate_on_device(output,truth,y,geometry,device,False)
        files.require(abs(regions['raw_mse']-row['raw_mse'])<=1e-9,'A139 control MSE differs')
        arms[name] = dict(row,**regions,inherited=True)
    order = method.ARMS if arms else ARMS
    for arm in order:
        trace = []
        first = []
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)

            def at_entry(_network,args):
                if len(counts['calls'])==len(grid)-len(active)+1:
                    first.append(fingerprint(args[0]))
                    if 'SCALAR' in arms:
                        files.require(first[0]==arms['SCALAR']['late_entry'],'Native scalar prefix differs')

            with model.model.register_forward_pre_hook(at_entry):
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:native_sample(entry,host,model,wave,arm,active,warm,counts,trace))
            pairing = history.fingerprint(detail,shared.rng(),counts,host)
        count_budget(counts,len(grid),0,len(grid),grid)
        files.require(detail['grid']==grid and len(first)==1 and
                      [p['t'] for p in detail['trace'] if 'weight' in p]==grid and
                      tuple(p['t'] for p in trace)==(() if arm=='SCALAR' else active),
                      'Native time or intervention pairing differs')
        if arm=='SCALAR':
            ref,folder = entry['legacy_scalar']
            bridge.archived_error(out,folder,ref,'output.pt')
        else:
            files.require(pairing==arms['SCALAR']['pairing'],'Native RNG pairing differs')
        record = dict(status='complete',case=key,arm=arm,counts=counts,pairing=pairing,late_entry=first[0],
                      method_nfe=len(grid),input_vjp=len(grid),full_seconds=seconds,inherited=False,
                      **evaluate_on_device(out,truth,y,geometry,device))
        save_candidate(root,key,arm,out,record,trace)
        arms[arm] = record
        actual.update({k:counts.get(k,0) for k in COSTS})
        print('A145 COMPLETE',host,key,arm,record['quality'],f'{seconds:.2f}s',flush=True)
    return dict(case=key,arms=arms,geometry=gate),dict(actual)


def zaps_case(root,entry,model,wave,kernel,grid,active,warm):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_shared_prefix_feedback import rng_equal

    case,obs = entry['case'],entry['observation']
    key = case['key']
    truth = load_image_as_tensor(case['image'])
    device = model.device
    y = audit.checked_tensor(obs['path'],device)

    def operator(count=None):
        op = zaps.make_operator(case,device,count=count,kernel=kernel)
        if 'mask' in entry:
            op._mask_cache = audit.checked_tensor(entry['mask'],device)
        return op

    with clock.counters(model.model) as counts:
        prefix_op = operator(counts)
        prefix,prefix_seconds = clock.clocks(torch.cuda.synchronize,
            lambda:zaps.train_prefix(model,prefix_op,y,case))
    count_budget(counts,270,9,0,grid*9)
    actual = Counter({k:counts.get(k,0) for k in COSTS})

    eval_counts = {}
    native.initialize_counts(eval_counts)
    with torch.no_grad():
        geometry = method.Geometry(case['task'],operator(),truth.to(device),eval_counts)
        gate = geometry.check(truth.to(device))
        seen_y = y if case['task']!='inpainting' else y[geometry.view.operator.selected]
    arms = {}
    for arm,(row,folder) in entry['controls'].items():
        name = ZAPS_ARMS[arm]+'.pt'
        out = audit.checked_tensor(audit.qualified_artifact(folder,name,row),'cpu')
        errors = evaluate_on_device(out,truth,seen_y,geometry,device,False)
        arms[arm] = dict(quality=row['metrics'],full_seconds=row['method_equivalent_seconds'],
                         method_nfe=300,input_vjp=0,optimizer_backward=10,inherited=True,**errors)

    original_rng = None
    for arm in ('SCALAR',*method.ARMS):
        trace = []
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)
            torch.cuda.synchronize()
            setup_started = time.perf_counter()
            op = operator(counts)
            if arm=='SCALAR':
                context = nullcontext()
            else:
                geom = method.Geometry(case['task'],op,truth.to(device),counts)
                suffix = zaps_sampler(op,wave,arm,case['sigma'],active,warm,geom,counts,trace)
                context = patch.object(zaps,'_custom_suffix',suffix)
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter()-setup_started
            with context:
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:zaps.run_suffix(model,op,y,case,ZAPS_ARMS['SCALAR'],prefix))
        count_budget(counts,30,1,0,grid)
        actual.update({k:counts.get(k,0) for k in COSTS})
        if arm=='SCALAR':
            original_rng = detail['final_rng']
            row,folder = entry['controls'][arm]
            replay = bridge.archived_error(out,folder,row,ZAPS_ARMS[arm]+'.pt')
            arms[arm].update(replay_relative_error=replay,same_session_seconds=prefix_seconds+seconds)
            continue
        files.require(rng_equal(original_rng,detail['final_rng']) and tuple(x['t'] for x in trace)==active,
                      'ZAPS suffix RNG or temporal coverage differs')
        record = dict(status='complete',case=key,arm=arm,counts=counts,method_nfe=300,input_vjp=0,
                      optimizer_backward=10,full_seconds=prefix_seconds+setup_seconds+seconds,inherited=False,
                      suffix_setup_seconds=setup_seconds,
                      **evaluate_on_device(out,truth,seen_y,geometry,device))
        save_candidate(root,key,arm,out,record,trace)
        arms[arm] = record
        print('A145 COMPLETE zaps',key,arm,record['quality'],f'{record["full_seconds"]:.2f}s',flush=True)
    return dict(case=key,arms=arms,geometry=gate),dict(actual)


REGIONS = ('raw_mse','observable_mse','weak_mse','clean_observation_mse','noisy_observation_mse')


def summarize(cases):
    arms = {}
    for arm in ARMS:
        rows = [row['arms'][arm] for row in cases]
        arms[arm] = dict(quality={m:statistics.mean(r['quality'][m] for r in rows)
                                  for m in parent.transfer.METRICS},
                         regions={k:statistics.mean(r[k] for r in rows) for k in REGIONS},
                         seconds=statistics.mean(r['full_seconds'] for r in rows),
                         nfe=rows[0]['method_nfe'],input_vjp=rows[0]['input_vjp'] if 'input_vjp' in rows[0]
                         else rows[0]['counts']['input_vjp_attempts'],inherited=rows[0]['inherited'])
    pairs = []
    for arm,ref in (('ZERO_WEAK','SCALAR'),('TEMPORAL_WEAK','SCALAR'),
                    ('TEMPORAL_WEAK','ZERO_WEAK'),('TEMPORAL_WEAK','DIAGONAL')):
        comparison = bridge.paired(cases,arm,ref)
        comparison['region_delta'] = {k:arms[arm]['regions'][k]-arms[ref]['regions'][k] for k in REGIONS}
        pairs.append(comparison)
    return dict(images=len(cases),arms=arms,paired=pairs)


def execute(host,task,root,source):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    sources = zaps_sources(task,source) if host=='zaps' else native_sources(host,task)
    inputs,checkpoint,kernel_path,frozen = sources
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Frozen checkpoint changed')
    files.require(all(root!=p and not p.is_relative_to(root) for p in frozen),'A145 root overlaps inputs')
    grid,active,warm = schedule(host)
    card = parent.transfer.PROJECTS/'experiment_cards/A145_temporal_panel.md'
    config = dict(experiment='A145',host=host,task=task,grid=grid,positions=list(active),warm=list(warm),
                  source_sha256={str(p):files.digest(p) for p in frozen},
                  code_sha256={str(p):files.digest(p) for p in (Path(__file__),Path(method.__file__),
                               Path(temporal.__file__),Path(temporal.graph.__file__),card)},
                  checkpoint=checkpoint,
                  graph_strength=temporal.STRENGTH,ridge=temporal.graph.RIDGE,
                  tolerance=temporal.late.depth.GRAD_RTOL,max_updates=temporal.late.depth.MAX_UPDATES)
    path = root/'summary.json'
    if path.exists():
        previous = files.read(path)
        files.require(previous['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A145 retained at {root}')
        print('A145 SUMMARY',host,task,previous['group'],flush=True)
        return
    resource = admission.check(torch,root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json',config)
    report = dict(experiment='A145',host=host,task=task,status='running',cases=[],resource=resource,
                  actual_cost=dict.fromkeys(COSTS,0),cost_is_lower_bound=False)
    files.write(path,report)
    try:
        start = time.perf_counter()
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = None if kernel_path is None else np.load(kernel_path,allow_pickle=False)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-start
        scope = native.dps_reference_scope() if host=='dps' else nullcontext()
        derivative = input_vjp_parameters(model.model) if host!='zaps' else nullcontext()
        with precision('LEGACY_IEEE'),frozen_network(model.model),derivative,scope:
            for entry in inputs:
                if host=='zaps':
                    row,cost = zaps_case(root,entry,model,wave,kernel,grid,active,warm)
                else:
                    row,cost = native_case(root,entry,host,model,wave,grid,active,warm)
                report['cases'].append(row)
                for key,value in cost.items():
                    report['actual_cost'][key] += value
                files.write(path,report)
        expected = len(inputs)*(360 if host=='zaps' else len(grid)*(2 if inputs[0]['controls'] else 4))
        files.require(report['actual_cost']['forwards']==expected,'A145 total calls differ')
        report.update(status='complete',group=summarize(report['cases']))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),cost_is_lower_bound=True)
        raise
    finally:
        files.write(path,report)
    print('A145 SUMMARY',host,task,report['group'],flush=True)


def matrix(root):
    rows = []
    for host in HOSTS:
        for task in TASKS:
            reuse = REUSED.get((host,task))
            folder,candidate,zero = reuse or (root/f'{host}_{task}','TEMPORAL_WEAK','ZERO_WEAK')
            path = folder/'summary.json'
            record = files.read(path) if path.is_file() else {'status':'missing'}
            row = dict(host=host,task=task,status=record['status'],source=str(path))
            if record['status']=='complete':
                group = record['group']
                comparisons = {p['reference']:p for p in group['paired'] if p['candidate']==candidate}
                quality = comparisons['SCALAR']
                mechanism = comparisons[zero]
                weak_names = ('weak_mse','missing_mse','nullspace_mse')
                weak_key = next(k for k in weak_names if k in mechanism['region_delta'])
                timing_key = 'seconds' if 'seconds' in group['arms'][candidate] else 'full_seconds'
                row.update(nfe=group['arms'][candidate]['nfe'],input_vjp=group['arms'][candidate]['input_vjp'],
                           delta=quality['delta'],psnr_wins=quality['psnr_wins'],images=quality['images'],
                           passes=quality['passes'],weak_error_key=weak_key,
                           seconds={n:group['arms'][n][timing_key] for n in ('SCALAR',zero,candidate)},
                           versus_zero=mechanism['delta'],
                           weak_mse_delta_vs_zero=mechanism['region_delta'][weak_key],
                           weak_mse_delta_vs_scalar=quality['region_delta'][weak_key])
            elif record.get('error'):
                row['error'] = record['error']
            rows.append(row)
            print('A145 MATRIX',row,flush=True)
    files.write(root/'matrix.json',rows)
    return all(row['status']=='complete' for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=(*HOSTS,'matrix'))
    parser.add_argument('task',choices=TASKS,nargs='?')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--zaps-parent',type=Path,default=Path(zaps.rules.ROOT))
    args = parser.parse_args()
    if args.host=='matrix':
        raise SystemExit(0 if matrix(args.root.resolve()) else 1)
    if args.task is None:
        parser.error('a task is required for a native job')
    files.require((args.host,args.task) not in REUSED,'Reuse this cell via A143/A144')
    execute(args.host,args.task,args.root.resolve()/f'{args.host}_{args.task}',args.zaps_parent.resolve())


if __name__=='__main__':
    main()
