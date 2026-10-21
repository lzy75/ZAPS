"""A26 bounded 27-arm scan, then locked A27 multi-image confirmation; no historical edits."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_fixed_count_state_positions as source
from utils import state_weight_scan as rules
from utils import diag_guidance_replication as baseline
from utils.diag_state_trajectory import digest
parent=source.parent;policy=source.policy;old=parent.parent.old;WRITE=source.WRITE
ROOT='/data0/lzy/ZAPS-results/diag_state_weight_scan/A26_A27_20261021'
CARD=PROJECTS/'experiment_cards/A26_A27_multimage_weight_scan_20261021.md'
ROLES=('C','F','S','P','M','STAGE')


def finish(path,r):
    WRITE(path,r);path.with_suffix('.sha256').write_text(digest(path)+'\n')


def check_plan(root):
    p=old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Frozen plan changed')
    for path,sha in p['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Bound worker changed: '+path)
    return p


def history_pool(data,root,n):
    """Select before outcomes; audit image SHA or same-directory image paths, not unrelated IDs."""
    paths=set()
    for base in (PROJECTS/'results',PROJECTS/'server_records',Path('/data0/lzy/ZAPS-results')):
        if base.exists():paths.update(p.resolve() for p in base.rglob('run.json') if not p.resolve().is_relative_to(root))
    used_names=set();used_sha=set();audit=[]
    def visit(v):
        if isinstance(v,dict):
            if isinstance(v.get('image_sha256'),str):used_sha.add(v['image_sha256'])
            image=v.get('image')
            if isinstance(image,str) and Path(image).resolve().parent==data:used_names.add(Path(image).stem)
            for k,item in v.items():
                if k not in ('experiment_card','limits','scope'):visit(item)
        elif isinstance(v,list):
            for item in v:visit(item)
    for path in sorted(paths):
        r=old.read(path);visit(r);audit.append(dict(path=str(path),sha256=digest(path)))
    if not audit:raise ValueError('No accessible historical run records')
    pool=[]
    for path in sorted(data.glob('*.png')):
        if path.stem.isdigit() and 32<=int(path.stem)<=999 and path.stem not in used_names:
            sha=digest(path)
            if sha not in used_sha:pool.append(dict(image=str(path.resolve()),image_id=path.stem,image_sha256=sha))
    # Avoid duplicated images under different filenames within the candidate pool.
    unique={}
    for c in pool:unique.setdefault(c['image_sha256'],c)
    pool=sorted(unique.values(),key=lambda c:c['image_id'])
    if len(pool)<n:raise ValueError(f'Only {len(pool)} unused unique accessible images; need {n}; do not substitute after results')
    selected=sorted([pool[i] for i in np.random.default_rng(20261021).choice(len(pool),n,replace=False)],key=lambda c:c['image_id'])
    cases=[dict(c,key=c['image_id']+'_s'+str(seed),sampling_seed=seed,measurement_seed=1000+int(c['image_id'])) for c in selected for seed in (1000,2000)]
    return dict(pool=pool,history_audit=audit,scope='Accessible server run records only; uniform sample from eligible local files, fixed before development outcomes.'),cases


def prepare(args):
    root=Path(args.root).resolve();a22=Path(args.a22_dir).resolve()
    if root.exists():
        p=check_plan(root)
        if p['source_A22']!=str(a22) or p['validation_images']!=args.validation_images:raise ValueError('Existing plan size/source differs; refuse overwrite')
        for path,sha in p['identity_files'].items():
            if digest(path)!=sha:raise ValueError('Frozen input changed: '+path)
        print('A26/A27 frozen plan reused; validation IDs:',sorted({c['image_id'] for c in p['validation_cases']}),flush=True);return
    if root.is_relative_to(a22) or a22.is_relative_to(root):raise ValueError('Run overlaps historical A22')
    p=old.read(a22/'plan.json')
    for name in ('plan','summary'):
        if digest(a22/(name+'.json'))!=(a22/(name+'.sha256')).read_text().strip():raise ValueError('A22 SHA differs')
    rr=[parent.load_record(a22,s) for s in p['cases']]
    if len(rr)!=12 or any(r is None for r in rr) or parent.summarize(rr)!=old.read(a22/'summary.json'):raise ValueError('Need twelve audited A22 cases')
    if {(s['case']['image_id'],s['case']['sampling_seed']) for s in p['cases']}!={(f'{i:05d}',s) for i in range(20,26) for s in (1000,2000)}:raise ValueError('Calibration cohort differs')
    identities=dict(p['identity_files'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Historical source changed: '+path)
    def bind(path):identities[str(path)]=digest(path)
    for name in ('plan.json','plan.sha256','summary.json','summary.sha256'):bind(a22/name)
    for s in p['cases']:
        for name in ('run.json','run.sha256','outputs.pt'):bind(a22/s['case']['key']/name)
    data=Path(p['cases'][0]['case']['image']).resolve().parent
    audit,fresh=history_pool(data,root,args.validation_images)
    for c in fresh:identities[c['image']]=c['image_sha256']
    records=[old.read(Path(s['source'])/'run.json') for s in p['cases']]
    refs={s['case']['key']:policy.reference(records,s['case']['image_id']) for s in p['cases']}
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'utils/test_state_weight_scan.py',PROJECTS/'run_a26_a27_20261021.sh',PROJECTS/'utils/archive_refinement_confirmation.py',PROJECTS/'utils/diag_guidance_replication.py']
    for path in own:bind(path)
    plan=dict(experiment='A26_A27_state_weight_scan',source_A22=str(a22),calibration_cases=p['cases'],validation_cases=fresh,validation_images=args.validation_images,
        config=p['config'],grid=p['grid'],checkpoint_sha256=p['checkpoint_sha256'],runtime=p['runtime'],native_policy=p['native_policy'],identity_files=identities,
        worker_identity_files={**p['worker_identity_files'],**{str(x):identities[str(x)] for x in own},str(Path(source.__file__)):digest(source.__file__),str(Path(policy.__file__)):digest(policy.__file__)},
        bank=rules.candidates(),references=refs,confirmation_reference=policy.reference(records),validation_history=audit,
        planned_method_nfe=600*args.validation_images,planned_diagnostic_nfe=9720+360*args.validation_images,deployment_nfe_per_case=300,rounds='A26=4/5 scan; A27=5/5 locked confirmation',
        user_authorization='User explicitly reopened bounded parameter scans and selected 6 calibration +12 confirmation images, two seeds each on 2026-10-21.')
    root.parent.mkdir(parents=True,exist_ok=True);parent.coupled.archive.recovery.require_space(root.parent);root.mkdir(parents=True,exist_ok=False)
    WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    print('A26/A27 PREPARE PASS:',len(plan['bank']),'arms; calibration images=6; confirmation IDs=',sorted({c['image_id'] for c in fresh}),'new method/diagnostic=',plan['planned_method_nfe'],plan['planned_diagnostic_nfe'],flush=True)


@contextmanager
def counted(model,path,record,limit):
    start=time.perf_counter();record.update(status='running',calls=[],nfe=0)
    def save():record['seconds']=time.perf_counter()-start;WRITE(path,record)
    def count(_module,inputs):
        t=int(inputs[1].flatten()[0]);record['nfe']+=1;record['calls'].append(t);save()
        if record['nfe']>limit:raise RuntimeError('Actual model budget exceeded')
    save();hook=model.model.register_forward_pre_hook(count)
    try:yield record
    except BaseException as e:record.update(status='failed',error=repr(e));save();raise
    finally:hook.remove()


@contextmanager
def preserved_rng():
    import torch
    numpy_state=np.random.get_state();python_state=random.getstate()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):yield
    finally:np.random.set_state(numpy_state);random.setstate(python_state)


def read_arm(folder,plan_sha,definition):
    path=folder/'run.json'
    if not path.exists():
        if folder.exists():raise ValueError('Interrupted arm retained: '+str(folder))
        return None
    r=old.read(path)
    if r['status'] not in ('sampled','complete') or digest(path)!=path.with_suffix('.sha256').read_text().strip() or r['plan_sha256']!=plan_sha or r['definition']!=definition or r['nfe']!=30 or len(r['calls'])!=30 or r['calls']!=[x['t'] for x in r['trace']] or digest(folder/'output.pt')!=r['output_sha256'] or not r['gates'] or not all(g['passed'] for g in r['gates'].values()):raise ValueError('Invalid/incomplete arm retained: '+str(folder))
    return r


def roll(z,y,state,noises,positions,ref,definition,before,observe=None):
    import torch
    from utils.diag_cached_guidance_geometry import vector_readouts
    params=definition['parameters'];stage=definition.get('stage');x=state['init_noise'].to(y.device);t=999;history=[];rows=[]
    for k in range(30):
        before(k,t);f=source.field(z,y,x,t,positions);u=float(positions.u[t])
        geo=vector_readouts(f['q'],history[-1] if history else None,history[-2] if len(history)>1 else None,u,rows[-1]['u'] if rows else None,rows[-2]['u'] if len(rows)>1 else None)
        signals=policy.scores(f['R'],geo['variation3_per_u'],u,ref);E,gamma=rules.controls(params,signals,t)
        if stage is not None:gamma=stage['gamma'][k]
        s=-1 if k==29 else (stage['times'][k+1] if stage is not None else rules.choose(positions,k+1,E,params['rho']))
        if observe is not None:observe(k,x,f)
        rows.append(dict(step=k,t=t,next_t=s,u=u,R=f['R'],Omega=geo['variation3_per_u'],qRMS=geo['rms'],clip_fraction=f['clip_fraction'],E=E,gamma=gamma,**signals,parameter_weights=f['weights']))
        history.append(f['q']);history=history[-2:]
        # Anchor the arithmetic at the original update, matching the archived A22 pulse.
        base=f['clean']+f['q'] if s<0 else parent.coupled.posterior(x,f['clean'],t,s,z.dm.alphas_cumprod,noises[k])+f['q']
        x=base if gamma==1 else base+(gamma-1)*f['q'];t=s
        if not torch.isfinite(x).all():raise RuntimeError('Nonfinite state')
    return x.cpu(),rows


def training_source(root,p,case,model,device,operator):
    """Only original 300-call optimization; checkpoint it separately from all diagnostic arms."""
    import fcntl,torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    folder=root/'validation'/case['key']/'BASELINE';path=folder/'run.json'
    if path.exists():
        r=old.read(path)
        if r['status']!='complete' or r['case']!=case or r['plan_sha256']!=digest(root/'plan.json') or digest(path)!=path.with_suffix('.sha256').read_text().strip() or r['nfe']!=300 or r['calls']!=list(reversed(p['grid']))*10:raise ValueError('Invalid/interrupted training retained: '+str(folder))
        for name,sha in r['artifacts'].items():
            if digest(folder/name)!=sha:raise ValueError('Training artifact changed')
        if digest(r['measurement']['path'])!=r['measurement']['sha256']:raise ValueError('Observation changed')
        return folder,r
    if folder.exists():raise ValueError('Interrupted training retained')
    folder.mkdir(parents=True);measurements=root/'measurements';measurements.mkdir(exist_ok=True);measurement=measurements/(case['image_id']+'.pt')
    with measurement.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not measurement.exists():
            truth=load_image_as_tensor(case['image']).to(device);set_seed(case['measurement_seed'])
            with torch.no_grad():y=operator.H(truth)+.05*torch.randn_like(operator.H(truth))
            tmp=measurement.with_suffix('.pt.tmp');torch.save(dict(y=y.cpu(),image_sha256=case['image_sha256'],measurement_seed=case['measurement_seed']),tmp);tmp.replace(measurement);del truth
        saved=torch.load(measurement,map_location='cpu',weights_only=True)
    if saved['image_sha256']!=case['image_sha256'] or saved['measurement_seed']!=case['measurement_seed']:raise ValueError('Observation identity differs')
    y=saved['y'].to(device)
    if y.shape!=(1,3,64,64) or y.dtype!=torch.float32 or not torch.isfinite(y).all():raise ValueError('Invalid observation')
    r=dict(kind='original_training',case=case,plan_sha256=digest(root/'plan.json'),measurement=dict(path=str(measurement),sha256=digest(measurement)))
    with counted(model,path,r,300),model_rng_guard(model.model,device):
        z=baseline.snapshot_class()(model,operator,img_size=256,**p['config']);z.tau=torch.tensor(p['grid'],device=device)
        set_seed(case['sampling_seed']);initial=torch.randn(1,3,256,256,device=device)
        for _ in range(290):torch.randn_like(initial)
        expected=rng_state(device);set_seed(case['sampling_seed'])
        r['loss_history']=z.optimize(y,verbose=False,x0_gt=None,scheduler=None)
        if (r['nfe']!=300 or r['calls']!=list(reversed(p['grid']))*10 or len(r['loss_history'])!=10 or not all(math.isfinite(v) for v in r['loss_history']) or z._last_nfe!=300 or
            z.audit_state is None or not torch.equal(z._last_opt_noise,initial) or not same_rng(expected,rng_state(device)) or any(a.grad is not None for a in model.model.parameters())):raise RuntimeError('Original training protocol failed')
        torch.save(z.audit_state,folder/'pre_last_state.pt');torch.save(z._last_opt_x0.detach().cpu(),folder/'last_opt.pt')
        r['artifacts']={n:digest(folder/n) for n in ('pre_last_state.pt','last_opt.pt')};r['gates']=dict(original_training=dict(passed=True),GT_epoch_selection=dict(passed=True,GT_used=False))
    r['status']='complete';finish(path,r);print('A27 BASELINE:',case['key'],'method NFE=300',flush=True);return folder,r


def selection(root,p):
    path=root/'selection.json';r=old.read(path)
    if digest(path)!=(root/'selection.sha256').read_text().strip() or r['plan_sha256']!=digest(root/'plan.json') or r['validation_used']:raise ValueError('Selection changed/leaks validation')
    for path,sha in r['calibration_records'].items():
        if digest(path)!=sha:raise ValueError('Selected development records changed')
    return r


def definitions(root,p,phase):
    if phase=='calibration':return {a:dict(parameters=v) for a,v in p['bank'].items()}
    chosen=selection(root,p);result={a:dict(parameters=p['bank'][a]) for a in ('C','F')}
    result.update({a:dict(parameters=chosen['parameters'][a],selection_sha256=digest(root/'selection.json')) for a in ('S','P')})
    result['M']=dict(parameters=dict(chosen['parameters']['S'],kind='MATCHED'),selection_sha256=digest(root/'selection.json'))
    result['STAGE']=dict(parameters=p['bank']['C'],stage=chosen['stage'],selection_sha256=digest(root/'selection.json'));return result


def case_records(root,p,phase,case,defs,require_complete=True):
    arms={a:read_arm(root/phase/case['key']/a,digest(root/'plan.json'),d) for a,d in defs.items()}
    if any(r is None or (require_complete and r['status']!='complete') for r in arms.values()):return None
    gate_path=root/phase/case['key']/'case_gate.json'
    if not gate_path.exists():return None
    gate=old.read(gate_path)
    if digest(gate_path)!=gate_path.with_suffix('.sha256').read_text().strip() or gate['case']!=case or gate['plan_sha256']!=digest(root/'plan.json') or not gate['passed']:raise ValueError('Case completion gate differs')
    for arm in defs:
        if digest(root/phase/case['key']/arm/'run.json')!=gate['arm_records'][arm]:raise ValueError('Case gate arm hashes differ')
    if any(r['case']!=case for r in arms.values()):raise ValueError('Arm case identity differs')
    if digest(case['image'])!=case['image_sha256']:raise ValueError('Completed image identity changed')
    if phase=='calibration':
        spec=next(s for s in p['calibration_cases'] if s['case']['key']==case['key']);parent.verify_case_sources(p,spec);src=Path(spec['source']);measurement=spec['measurement']
    else:
        src=root/'validation'/case['key']/'BASELINE';path=src/'run.json';base=old.read(path)
        if digest(path)!=path.with_suffix('.sha256').read_text().strip() or base['status']!='complete' or base['case']!=case or base['nfe']!=300 or base['calls']!=list(reversed(p['grid']))*10 or base['plan_sha256']!=digest(root/'plan.json') or not base['gates'] or not all(g['passed'] for g in base['gates'].values()):raise ValueError('Validation baseline changed')
        for name,sha in base['artifacts'].items():
            if digest(src/name)!=sha:raise ValueError('Validation baseline tensor changed')
        measurement=base['measurement']
    if digest(measurement['path'])!=measurement['sha256']:raise ValueError('Observation changed')
    bound=dict(PRE=digest(src/'pre_last_state.pt'),original=digest(src/'last_opt.pt'),measurement=digest(measurement['path']))
    if any(r['inputs']!=bound for r in arms.values()):raise ValueError('Completed arm source fingerprints differ')
    return dict(case=case,branches={a:r['metrics'] for a,r in arms.items()},traces={a:r['trace'] for a,r in arms.items()},loss_floor=arms['C']['loss_floor'])


def work_case(root,p,phase,spec,model,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection,restore_sampling_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    case=spec['case'] if phase=='calibration' else spec;defs=definitions(root,p,phase)
    if digest(case['image'])!=case['image_sha256']:raise ValueError('Image changed')
    if case_records(root,p,phase,case,defs) is not None:return
    external=rng_state(device)
    with preserved_rng():
        operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
        if phase=='calibration':
            parent.verify_case_sources(p,spec);src=Path(spec['source']);native=old.read(src/'run.json');measurement=spec['measurement'];cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True)
            ref=p['references'][case['key']]
        else:
            src,native=training_source(root,p,case,model,device,operator);measurement=native['measurement'];cache=None;ref=p['confirmation_reference']
        state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True);original=torch.load(src/'last_opt.pt',map_location='cpu',weights_only=True)
        y=torch.load(measurement['path'],map_location='cpu',weights_only=True)['y'].to(device)
        z=ZAPS(model,operator,img_size=256,**p['config']);z.tau=state['tau'].to(device)
        with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
        positions=policy.Positions(policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(p['grid'])))
        if state['tau'].tolist()!=p['grid']:raise ValueError('PRE grid differs')
        if phase=='calibration':
            noises,noise_gate=parent.coupled.replay_noises(state,cache,dict(rows=native['rows']),z,y,device,original)
        else:
            mapping=saved_cuda_selection(state,device)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                restore_sampling_rng(state,device,mapping['saved_sampling_index']);noises={k:torch.randn_like(state['init_noise'].to(device)) for k in range(29)}
            noise_gate=dict(passed=True,draws=29,selection=mapping,contract='Raw PRE draws; correctness additionally gated by C endpoint parity')
        bound=dict(PRE=digest(src/'pre_last_state.pt'),original=digest(src/'last_opt.pt'),measurement=digest(measurement['path']))
        for arm,definition in defs.items():
            folder=root/phase/case['key']/arm;r=read_arm(folder,digest(root/'plan.json'),definition)
            if r is not None:
                if r['inputs']!=bound:raise ValueError('Completed arm input fingerprints differ')
                continue
            folder.mkdir(parents=True);path=folder/'run.json';r=dict(case=case,phase=phase,arm=arm,definition=definition,plan_sha256=digest(root/'plan.json'),inputs=bound)
            with counted(model,path,r,30),torch.no_grad(),model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unaccounted sampling noise')):
                errors=[]
                def observe(k,x,f):
                    if arm=='C' and cache is not None:
                        er={n:relative_error(a,b) for n,a,b in (('latent',cache['latent_inputs'][k:k+1],x.cpu()),('clean',cache['clean'][k:k+1],f['clean'].cpu()),('q',cache['guidance'][k:k+1],f['q'].cpu()))}
                        if any(not math.isfinite(v) or v>1e-6 for v in er.values()):raise RuntimeError('Original step parity failed')
                        errors.append(dict(step=k,**er))
                output,trace=roll(z,y,state,noises,positions,ref,definition,lambda k,t:None,observe)
                if r['calls']!=[row['t'] for row in trace] or r['nfe']!=30:raise RuntimeError('Actual times/NFE mismatch')
                error=relative_error(original,output) if arm=='C' else 0.
                if not math.isfinite(error) or error>1e-6:raise RuntimeError('Original endpoint parity failed')
                torch.save(output,folder/'output.pt');r.update(trace=trace,output_sha256=digest(folder/'output.pt'),gates=dict(noise=noise_gate,actual_budget=dict(passed=True),original_C=dict(passed=True,applicable=arm=='C',endpoint_error=error,steps=errors)))
            r['status']='sampled';finish(path,r)
            print('A26' if phase=='calibration' else 'A27','SAMPLED:',case['key'],arm,'30 calls',flush=True)
        # All candidate outputs are immutable before GT evaluation starts.
        if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Original parameters changed')
        gt=load_image_as_tensor(case['image']).cpu()
        def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
        baseline_metrics=metrics(original);floor=1e-7+2e-5*abs(baseline_metrics['loss'])
        for arm,definition in defs.items():
            folder=root/phase/case['key']/arm;path=folder/'run.json';r=read_arm(folder,digest(root/'plan.json'),definition)
            if r['status']=='complete':continue
            r['metrics']=metrics(torch.load(folder/'output.pt',map_location='cpu',weights_only=True));r['loss_floor']=floor
            if not all(math.isfinite(v) for v in r['metrics'].values()):raise RuntimeError('Nonfinite evaluation')
            if arm=='C':
                if abs(r['metrics']['loss']-baseline_metrics['loss'])>floor:raise RuntimeError('C physics parity failed')
                reference=native.get('final_metrics',native.get('baseline',baseline_metrics))
                if any(abs(r['metrics'][m]-reference[m])>tol for m,tol in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('C quality protocol differs')
            if phase=='calibration' and arm=='F':
                a22=parent.load_record(Path(p['source_A22']),spec);saved=torch.load(Path(p['source_A22'])/case['key']/'outputs.pt',map_location='cpu',weights_only=True)['outputs']['T319_G2']
                fixed_error=relative_error(saved,torch.load(folder/'output.pt',map_location='cpu',weights_only=True))
                if fixed_error>1e-6 or any(abs(r['metrics'][m]-a22['branches']['T319_G2'][m])>tol for m,tol in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Strong fixed F replay differs')
                r['gates']['strong_fixed_F']=dict(passed=True,relative_error=fixed_error)
            r['gates'].update(parameters=dict(passed=True),GT_after_all_outputs=dict(passed=True));r['status']='complete';finish(path,r)
        if phase=='calibration':parent.verify_case_sources(p,spec)
        elif digest(measurement['path'])!=native['measurement']['sha256']:raise ValueError('Shared observation changed')
        if bound['PRE']!=digest(src/'pre_last_state.pt') or bound['original']!=digest(src/'last_opt.pt'):raise ValueError('Source tensor changed')
    if not same_rng(external,rng_state(device)) or parent.precision.precision_state(torch)!=p['native_policy']:raise RuntimeError('External RNG/precision changed')
    check_plan(root)
    gate_path=root/phase/case['key']/'case_gate.json'
    finish(gate_path,dict(passed=True,case=case,plan_sha256=digest(root/'plan.json'),arm_records={a:digest(root/phase/case['key']/a/'run.json') for a in defs},scope='All arms complete; original parameters, sources, Torch/Python/NumPy RNG and native precision checked before case completion.'))
    print('CASE COMPLETE:',phase,case['key'],flush=True)


def collect(root,p,phase):
    defs=definitions(root,p,phase);rr=[]
    for spec in p['calibration_cases'] if phase=='calibration' else p['validation_cases']:
        case=spec['case'] if phase=='calibration' else spec
        r=case_records(root,p,phase,case,defs)
        if r is not None:rr.append(r)
    return rr


def lock_selection(root,p):
    from modules.diffusion_model import _precompute_diffusion_params
    rr=collect(root,p,'calibration')
    if len(rr)!=12:raise ValueError('Need all 12 complete calibration cases; do not use validation')
    positions=policy.Positions(policy.noise_u(_precompute_diffusion_params()['alphas_cumprod'].cpu().numpy()),list(reversed(p['grid'])))
    chosen=rules.select(rr,p['bank'],positions);chosen['plan_sha256']=digest(root/'plan.json')
    chosen['calibration_records']={str(path):digest(path) for s in p['calibration_cases'] for path in (root/'calibration'/s['case']['key']).glob('*/run.json')}
    chosen['calibration_records'].update({str(root/'calibration'/s['case']['key']/'case_gate.json'):digest(root/'calibration'/s['case']['key']/'case_gate.json') for s in p['calibration_cases']})
    path=root/'selection.json'
    if path.exists():
        if old.read(path)!=chosen:raise ValueError('Frozen development selection differs')
        selection(root,p)
    else:WRITE(path,chosen);(root/'selection.sha256').write_text(digest(path)+'\n')
    print('A26 LOCKED:',json.dumps(dict(roles=chosen['roles'],parameters=chosen['parameters'],fallback=chosen['development_fallback'],development_effects={r:chosen['development_effects'][a] for r,a in chosen['roles'].items()})),flush=True)


def costs(root):
    method=diagnostic=0;interrupted=False
    for path in root.rglob('run.json'):
        r=old.read(path);n=r.get('nfe',0)
        if r.get('kind')=='original_training':method+=n
        else:diagnostic+=n
        interrupted|=r.get('status') in ('running','failed')
    return dict(method_nfe=method,diagnostic_nfe=diagnostic,total=method+diagnostic,cost_is_lower_bound_if_abruptly_interrupted=interrupted)


def summarize(root,p):
    cost=costs(root);rr=collect(root,p,'calibration')
    out=dict(classification='INCOMPLETE_SCAN_OR_CONFIRMATION',completed_calibration_cases=len(rr),planned_calibration_cases=12,planned_validation_cases=len(p['validation_cases']),new_cost=cost,
        cumulative_queue_cost=dict(method_nfe=7200+cost['method_nfe'],diagnostic_nfe=5408+cost['diagnostic_nfe'],total=12608+cost['total']),deployment_nfe_per_case=300,
        limits='Reopened user-authorized finite hyperparameter scan. Last unroll only; original first 270 calls. GT used for development hyperparameter selection, never online action or output selection. Approximate image-cluster confidence within accessible FFHQ pool; not global efficacy, exact coverage, curvature or NFE-saving proof.')
    if len(rr)!=12 or not(root/'selection.json').exists():return out
    chosen=selection(root,p);out['selection']=dict(roles=chosen['roles'],parameters=chosen['parameters'],development_fallback=chosen['development_fallback'])
    vv=collect(root,p,'validation');out['completed_validation_cases']=len(vv)
    if len(vv)!=len(p['validation_cases']):return out
    if cost['method_nfe']!=p['planned_method_nfe'] or cost['diagnostic_nfe']!=p['planned_diagnostic_nfe']:raise ValueError('Completed total budget differs')
    result=rules.confirmation(vv);out.update(classification=result['classification'],confirmation=result,
        rows=[dict(case=r['case']['key'],branches=r['branches']) for r in vv])
    return out


def worker(args,root,p):
    import fcntl,torch
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or parent.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Native runtime/precision differs; no calls')
        torch.set_num_threads(2);parent.coupled.archive.recovery.require_space(root)
        specs=p['calibration_cases'] if args.phase=='calibration' else p['validation_cases']
        specs=sorted(specs,key=lambda s:(s['case'] if args.phase=='calibration' else s)['key'])
        model=parent.coupled.archive.recovery.load_model(p,args.device)
        for i,s in enumerate(specs):
            if (i//2)%2==args.shard:work_case(root,p,args.phase,s,model,args.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a22-dir',default=parent.ROOT);p.add_argument('--validation-images',type=int,choices=(6,12,24),default=12);p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=('calibration','validation'),default='calibration')
    m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--shard',type=int,choices=(0,1));m.add_argument('--select',action='store_true');m.add_argument('--archive',action='store_true');m.add_argument('--summary',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=check_plan(root)
    if args.shard is not None:worker(args,root,plan);return
    if args.select:lock_selection(root,plan);return
    if args.summary:
        if digest(root/'summary.json')!=(root/'summary.sha256').read_text().strip():raise ValueError('Summary changed')
        full=old.read(root/'summary.json');compact={k:v for k,v in full.items() if k not in ('rows','confirmation')}
        if 'confirmation' in full:
            c=full['confirmation'];compact['confirmation']={k:v for k,v in c.items() if k not in ('comparisons','per_arm_vs_C')}
            compact['confirmation']['PSNR_comparisons']={a:{k:v for k,v in metrics['psnr'].items() if k!='per_image'} for a,metrics in c['comparisons'].items()}
            compact['confirmation']['mean_delta_vs_C']={a:e['mean_delta'] for a,e in c['per_arm_vs_C'].items()}
        print('A26/A27 Summary:',json.dumps(compact),flush=True);print('Records:',root,'; complete per-image data and all confidence bounds in summary.json',flush=True);return
    out=summarize(root,plan);WRITE(root/'summary.json',out);(root/'summary.sha256').write_text(digest(root/'summary.json')+'\n')
    from utils.archive_refinement_confirmation import export_and_commit
    bundle,commit=export_and_commit({'A26_A27':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


if __name__=='__main__':main()
