"""A147: independent ten-image confirmation of prequalified temporal cells."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import statistics
import time

from utils import diag_a145_temporal_panel as panel
from utils.a54_session import exclusive

temporal, parent = panel.temporal, panel.parent
audit, bridge, clock, files = panel.audit, panel.bridge, panel.clock, panel.files
history, native, shared = panel.history, panel.native, panel.shared
ROOT = panel.ROOT.with_name('diag_a147_temporal_confirmation')
CELLS = (('dps','inpainting'),('dps','super_resolution'),('pigdm','inpainting'))
ARMS = ('SCALAR','TEMPORAL')
DEVELOPMENT_IDS = {'00226','00443','00963'}
IMAGE_IDS = tuple(f'{i:05d}' for i in range(10))
REGIONS = panel.REGIONS


def source_inputs(host,task,panel_root,matrix_path):
    digest = files.digest(matrix_path)
    qualification = files.read(matrix_path)
    matches = [r for r in qualification if (r['host'],r['task'])==(host,task)]
    files.require(len(matches)==1 and matches[0]['status']=='complete' and matches[0]['passes'],
                  'A147 requires the qualifying A145 result')
    grid,active,warm = panel.schedule(host)
    parent_plan = panel_root/'plan.json'
    source_panel = files.read(parent_plan)
    files.require(source_panel['experiment']==panel.zaps.rules.EXPERIMENT,'Wrong pre-existing FFHQ panel')
    images = {r['image_id']:r for r in source_panel['cases'] if r['task']==task}
    old = [e for e in history.load_sources(host,'ffhq') if e['case']['task']==task]
    files.require(len(old)==3 and {r['case']['image_id'] for r in old}.isdisjoint(IMAGE_IDS) and
                  set(IMAGE_IDS).isdisjoint(DEVELOPMENT_IDS) and set(IMAGE_IDS)<=images.keys(),
                  'Confirmation identities overlap with the development panel')
    original = old[0]['plan']
    checkpoint = original['checkpoint']
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'FFHQ weight bytes differ')
    dev_hashes = {e['plan']['inputs'][e['case']['image']] for e in old}
    cases = []
    for image_id in IMAGE_IDS:
        anchor = images[image_id]
        path = anchor['image']
        sha = files.digest(path)
        files.require(sha==anchor['image_sha256'] and sha not in dev_hashes,'Confirmation image bytes differ')
        age = int(image_id)
        case = dict(key=f'{task}_{image_id}_n050',task=task,image=path,image_id=image_id,image_sha256=sha,
                    sigma=.05,seed=1000+age,measurement_seed=(2000 if host=='pigdm' else 1000)+age,
                    mask_seed=2001000+age,kernel_seed=3000+age)
        cases.append(case)
    files.require(len({c['image_sha256'] for c in cases})==len(IMAGE_IDS),'Duplicate confirmation image bytes')
    config = dict(experiment='A147',host=host,task=task,checkpoint=checkpoint,
                  A145_matrix_sha256=digest,A102_plan_sha256=files.digest(parent_plan),
                  native_plan_sha256=files.digest(old[0]['root']/'plan.json'),
                  cases=cases,arms=list(ARMS),grid=grid,positions=list(active),warm=list(warm),
                  graph_strength=temporal.STRENGTH,ridge=temporal.graph.RIDGE,
                  max_updates=temporal.late.depth.MAX_UPDATES,grad_rtol=temporal.late.depth.GRAD_RTOL,
                  decision=dict(mean_psnr_min=.10,psnr_wins_min=7,mean_ssim_min=0.,mean_lpips_max=0.))
    if host=='dps':
        for key in ('diffusion_config','dps_root'):
            config[key] = original[key]
    return config


def make_operator(host,case,device,counts=None,mask=None):
    if host=='pigdm':
        from utils.pigdm_noisy_runtime import Operator

        return Operator(case['task'],device,seed=case['kernel_seed'],count=counts)
    from utils.shrink_multitask_runtime import make_operator as build

    operator = build(case,device,count=counts)
    if mask is not None:
        operator._mask_cache = mask
    return operator


def observation(folder,host,case,truth,device):
    import torch

    file = folder/'observation.pt'
    meta = folder/'observation.json'
    if not meta.is_file():
        files.require(not file.exists(),'Partial A147 observation retained')
        with torch.no_grad():
            operator = make_operator(host,case,device)
            expected = operator.H(truth.to(device))
            shared.seed(case['measurement_seed'])
            y = expected+case['sigma']*torch.randn_like(expected)
            mask = operator.mask if host=='pigdm' else getattr(operator,'_mask_cache',None)
            torch.save(dict(y=y.cpu(),mask=None if mask is None else mask.cpu()),file)
        files.write(meta,dict(case=case,sha256=files.digest(file)))
    recorded = files.read(meta)
    files.require(recorded['case']==case and files.digest(file)==recorded['sha256'],'Frozen observation changed')
    return file,recorded['sha256']


def sample(config,case,host,arm,model,wave,observation_path,counts,trace):
    import torch

    payload = torch.load(observation_path,map_location=model.device,weights_only=True)
    op = make_operator(host,case,model.device,counts,payload['mask'] if host=='dps' else None)
    if host=='pigdm':
        files.require(torch.equal(op.mask,payload['mask']),'Native PiGDM mask differs')
    runtime = history.native(dict(case=case),host)
    run = runtime.run
    if arm=='TEMPORAL':
        if case['task']=='inpainting':
            remember,refine = temporal.controller(op,wave,'TEMPORAL',case['sigma'],counts,trace)
        else:
            geom = panel.method.Geometry(case['task'],op,torch.zeros((1,3,256,256),device=model.device),counts)
            remember,refine = panel.method.controller(geom,wave,'TEMPORAL_WEAK',case['sigma'],counts,trace)
        run = panel.native_instrument(runtime,host,config['positions'],config['warm'],remember,refine)
    if host=='pigdm':
        return run('A147',model,op,payload['y'],case,'SHRINK',counts)
    return run(model,op,payload['y'],case,'SHRINK',config['diffusion_config'],config['dps_root'],counts)


def measure(root,config,case,host,arm,model,wave,truth,observation_path,observation_sha):
    import torch
    from utils.diag_pigdm_noisy_r1 import fingerprint

    folder = root/'cases'/case['key']/arm
    path = folder/'run.json'
    if path.exists():
        row = files.read(path)
        files.require(row['status']=='complete' and row['observation_sha256']==observation_sha,
                      'A147 incomplete arm preserved: '+str(path))
        for name in ('output.pt','trace.json'):
            audit.qualified_artifact(folder,name,row)
        return row
    folder.mkdir()
    trace,counts,late = [],{},[]
    row = dict(status='running',case=case['key'],arm=arm,observation_sha256=observation_sha)
    files.write(path,row)
    grid = config['grid']
    positions = config['positions']
    entry_call = len(grid)-len(positions)+1
    try:
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)

            def at_entry(_network,args):
                if len(counts['calls'])==entry_call:
                    late.append(fingerprint(args[0]))

            with model.model.register_forward_pre_hook(at_entry):
                (output,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:sample(config,case,host,arm,model,wave,observation_path,counts,trace))
            pairing = history.fingerprint(detail,shared.rng(),counts,host)
        panel.count_budget(counts,len(grid),0,len(grid),grid)
        files.require(len(late)==1 and detail['grid']==grid and
                      [s['t'] for s in trace]==(positions if arm=='TEMPORAL' else []) and
                      [s['t'] for s in detail['trace'] if 'weight' in s]==grid,
                      'A147 native window or prefix missing')
        with torch.no_grad():
            payload = torch.load(observation_path,map_location=model.device,weights_only=True)
            op = make_operator(host,case,model.device,mask=payload['mask'] if host=='dps' else None)
            diagnostic = {}
            native.initialize_counts(diagnostic)
            geometry = panel.method.Geometry(case['task'],op,truth.to(model.device),diagnostic)
            gate = geometry.check(truth.to(model.device))
            y = payload['y']
            if host=='dps' and case['task']=='inpainting':
                y = y[geometry.view.operator.selected]
            regions = panel.evaluate_on_device(output,truth,y,geometry,model.device)
        torch.save(output,folder/'output.pt')
        files.write(folder/'trace.json',trace)
        row.update(status='complete',counts=counts,late_entry=late[0],pairing=pairing,full_seconds=seconds,
                   method_nfe=len(grid),geometry=gate,**regions,
                   artifacts={name:files.digest(folder/name) for name in ('output.pt','trace.json')})
        print('A147 COMPLETE',host,case['key'],arm,row['quality'],f'{seconds:.2f}s',flush=True)
        return row
    except BaseException as error:
        row.update(status='failed',error=repr(error),counts=counts)
        files.write(folder/'partial_trace.json',trace)
        raise
    finally:
        files.write(path,row)


def summarize(cases):
    arm_means = {}
    keys = parent.transfer.METRICS
    for arm in ARMS:
        rows = [case['arms'][arm] for case in cases]
        arm_means[arm] = dict(quality={k:statistics.mean(r['quality'][k] for r in rows) for k in keys},
                              regions={k:statistics.mean(r[k] for r in rows) for k in REGIONS},
                              seconds=statistics.mean(r['full_seconds'] for r in rows),
                              nfe=rows[0]['method_nfe'],input_vjp=rows[0]['counts']['input_vjp_attempts'])
    delta = {k:statistics.mean(case['arms']['TEMPORAL']['quality'][k]-case['arms']['SCALAR']['quality'][k]
                               for case in cases) for k in keys}
    wins = sum(case['arms']['TEMPORAL']['quality']['psnr']>case['arms']['SCALAR']['quality']['psnr'] for case in cases)
    regional = {k:arm_means['TEMPORAL']['regions'][k]-arm_means['SCALAR']['regions'][k] for k in REGIONS}
    return dict(images=len(cases),arms=arm_means,delta=delta,psnr_wins=wins,region_delta=regional,
                passes=len(cases)==10 and wins>=7 and delta['psnr']>=.10 and delta['ssim']>=0 and delta['lpips']<=0)


def execute(host,task,root,panel_root,matrix_path):
    import torch
    from modules.main_single import load_diffusion_model,load_image_as_tensor
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    files.require(not root.is_relative_to(panel_root) and not root.is_relative_to(matrix_path.parent),
                  'A147 needs a separate output root')
    settings = source_inputs(host,task,panel_root,matrix_path)
    card = parent.transfer.PROJECTS/'experiment_cards/A147_temporal_confirmation.md'
    methods = (Path(__file__),Path(temporal.__file__),Path(panel.__file__),Path(panel.method.__file__),card)
    settings['code_sha256'] = {str(p):files.digest(p) for p in methods}
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'worker.lock'):
        plan = root/'resolved_config.json'
        path = root/'summary.json'
        if plan.is_file():
            files.require(settings==files.read(plan),'A147 settings changed')
        else:
            files.write(plan,settings)
        if path.is_file() and files.read(path)['status']=='complete':
            print('A147 SUMMARY',host,task,files.read(path)['group'],flush=True)
            return
        resource = panel.admission.check(torch,root)
        report = dict(experiment='A147',host=host,task=task,status='running',cases=[],resource=resource,
                      planned_runs=2*len(IMAGE_IDS),recorded_cost=dict.fromkeys(panel.COSTS,0),
                      cost_is_lower_bound=False)
        files.write(path,report)
        try:
            started = time.perf_counter()
            checkpoint = settings['checkpoint']
            model = load_diffusion_model('ffhq','cuda:0')
            model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
            model.model.eval()
            wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
            torch.cuda.synchronize()
            report['model_load_seconds'] = time.perf_counter()-started
            scope = native.dps_reference_scope() if host=='dps' else nullcontext()
            with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),scope:
                for index,case in enumerate(settings['cases']):
                    folder = root/'cases'/case['key']
                    folder.mkdir(parents=True,exist_ok=True)
                    loader = shared.load_image if host=='pigdm' else load_image_as_tensor
                    truth = loader(case['image'])
                    observation_path,observation_sha = observation(folder,host,case,truth,model.device)
                    arms = {}
                    for arm in ARMS if index%2==0 else reversed(ARMS):
                        row = measure(root,settings,case,host,arm,model,wave,truth,observation_path,observation_sha)
                        arms[arm] = row
                        n = len(settings['grid'])
                        panel.count_budget(row['counts'],n,0,n,settings['grid'])
                        for key in report['recorded_cost']:
                            report['recorded_cost'][key] += row['counts'].get(key,0)
                    scalar,candidate = (arms[arm] for arm in ARMS)
                    files.require(scalar['pairing']==candidate['pairing'] and
                                  scalar['late_entry']==candidate['late_entry'],'A147 native pairing differs')
                    report['cases'].append(dict(case=case['key'],arms=arms))
                    files.write(path,report)
            expected = report['planned_runs']*len(settings['grid'])
            files.require(report['recorded_cost']['forwards']==report['recorded_cost']['input_vjp_attempts']==expected,
                          'A147 native budget incomplete')
            report.update(status='complete',group=summarize(report['cases']))
        except BaseException as exc:
            report.update(status='failed',error=repr(exc),cost_is_lower_bound=True)
            raise
        finally:
            files.write(path,report)
        print('A147 SUMMARY',host,task,report['group'],flush=True)
        print('A147 REPORT',path,flush=True)


def matrix(root):
    rows = []
    for host,task in CELLS:
        path = root/f'{host}_{task}'/'summary.json'
        data = files.read(path) if path.is_file() else {'status':'missing'}
        row = dict(host=host,task=task,status=data['status'],completed_cases=len(data.get('cases',[])),source=str(path))
        if data['status']=='complete':
            row.update(data['group'])
        elif data.get('error'):
            row['error'] = data['error']
        rows.append(row)
        print('A147 MATRIX',row,flush=True)
    files.write(root/'matrix.json',rows)
    return all(row['status']=='complete' for row in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host',choices=('dps','pigdm','matrix'))
    p.add_argument('task',choices=('inpainting','super_resolution'),nargs='?')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--a102-root',type=Path,default=Path(panel.zaps.rules.ROOT))
    p.add_argument('--a145-matrix',type=Path,default=panel.ROOT/'matrix.json')
    args = p.parse_args()
    root = args.root.resolve()
    if args.host=='matrix':
        raise SystemExit(0 if matrix(root) else 1)
    if (args.host,args.task) not in CELLS:
        p.error('unsupported A147 sampler/task pair')
    execute(args.host,args.task,root/f'{args.host}_{args.task}',args.a102_root.resolve(),args.a145_matrix.resolve())


if __name__=='__main__':
    main()
