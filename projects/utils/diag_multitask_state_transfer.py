"""A40 FFHQ/ImageNet x SR/Gaussian fixed-grid state transfer; 11160 planned calls."""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_frozen_response_transfer as base
from utils import diag_common_state_risk as a39
from utils import multitask_state_transfer as rules

source,P,scan=base.source,base.P,base.scan
ROOT='/data0/lzy/ZAPS-results/diag_multitask_state_transfer/A40_20261025'
CARD=PROJECTS/'experiment_cards/A40_multitask_state_transfer_20261025.md'


def model_sources():
    resolved={}
    for name in ('guided_diffusion.script_util','guided_diffusion.unet','guided_diffusion.nn','guided_diffusion.fp16_util'):
        spec=importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).is_file():raise ValueError('Missing model library: '+name)
        path=Path(spec.origin).resolve();resolved[name]=dict(path=str(path),sha256=source.digest(path))
    return resolved


def prepare(args):
    root,old,old39=Path(args.root).resolve(),Path(args.a36_dir).resolve(),Path(args.a39_dir).resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (old,old39)):
        raise ValueError('Output overlaps historical source')
    if root.exists():
        p=source.check_plan(root)
        if p['source_A36']!=str(old) or p['source_A39']!=str(old39):raise ValueError('Frozen sources differ')
        for d,folder in (('ffhq',args.ffhq_dir),('imagenet',args.imagenet_dir)):
            if Path(folder).resolve()!=Path(p['selected_images'][d][0]['image']).parent:raise ValueError('Frozen data directory differs')
        verify_inputs(p);print('A40 frozen plan reused; complete attempts retained.',flush=True);return
    previous=source.check_plan(old);p39=source.check_plan(old39);s39=source.load_signed(old39/'summary.json')
    if previous['experiment']!='A36_frozen_response_transfer' or s39['completed_cases']!=12:
        raise ValueError('Need frozen A36 table and complete A39')
    if p39['source_A36']!=str(old) or p39['grid']!=previous['grid']:
        raise ValueError('A36/A39 source grid differs')
    rules.risk.expected_calls(previous['grid'])
    if len(previous['time_gamma'])!=30 or any(not math.isfinite(v) or not 0<=v<=2 for v in previous['time_gamma']):
        raise ValueError('Invalid frozen source time table')
    source_records=[];identities={}
    for c in p39['cases']:
        item=a39.done(old39,p39,c)
        if item is None:raise ValueError('A39 source incomplete')
        source_records.append(str(item[0]));identities[str(item[0])]=source.digest(item[0])
    for directory in (old,old39):
        for name in ('plan.json','summary.json'):identities[str(directory/name)]=source.digest(directory/name)
    manifest=source.read(PROJECTS/'IMAGENET_BASELINE_MANIFEST_20261013.json')
    checkpoints=dict(ffhq=dict(path=str(PROJECTS/'modules/models/ffhq_10m.pt'),sha256=previous['checkpoint_sha256']),
        imagenet=dict(path=str(PROJECTS/'modules/models/256x256_diffusion_uncond.pt'),
                      sha256=manifest['archived_reference']['checkpoint_sha256_verified_in_prior_upload']))
    for spec in checkpoints.values():
        if source.digest(spec['path'])!=spec['sha256']:raise ValueError('Checkpoint identity differs')
    selected={};selection={};cases=[]
    for dataset,data in (('ffhq',args.ffhq_dir),('imagenet',args.imagenet_dir)):
        images,audit=base.select_images(Path(data).resolve(),root);selected[dataset]=images;selection[dataset]=audit
        for i,c in enumerate(images):
            identities[c['image']]=c['image_sha256']
            for task in rules.TASKS:
                cases.append(dict(c,dataset=dataset,task=task,image_index=i,sigma=rules.SIGMA,sampling_seed=1000,
                    measurement_seed=(1000 if task=='super_resolution' else 101000)+int(c['image_id']),
                    key=f'{dataset}_{task}_{c["image_id"]}_s1000'))
    if len({c['image_sha256'] for images in selected.values() for c in images})!=12:
        raise ValueError('Datasets contain duplicated selected content')
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a40_20261025.sh',
         PROJECTS/'utils/test_multitask_state_transfer.py',PROJECTS/'utils/common_state_risk.py',
         PROJECTS/'utils/trajectory_state_geometry.py',PROJECTS/'utils/cached_policy_opportunity.py',
         PROJECTS/'utils/experiment_gpu_selection.py',PROJECTS/'utils/diag_cached_guidance_geometry.py',
         PROJECTS/'IMAGENET_BASELINE_MANIFEST_20261013.json',PROJECTS/'modules/main_single.py',
         PROJECTS/'modules/degradations.py',PROJECTS/'modules/diffusion_model.py',
         PROJECTS/'modules/wavelet.py',PROJECTS/'modules/dataset_loader.py',
         PROJECTS/'utils/diag_imagenet_gaussian_unit_domain.py',PROJECTS/'utils/diag_common_state_risk.py',
         PROJECTS.parent/'DPS/util/resizer.py']
    workers={**p39['worker_identity_files'],**{str(f):source.digest(f) for f in own}}
    libraries=model_sources();workers.update({v['path']:v['sha256'] for v in libraries.values()})
    configs={d:{t:rules.configuration(previous['config'],d,t) for t in rules.TASKS} for d in rules.DATASETS}
    p=dict(experiment='A40_multitask_state_transfer',source_A36=str(old),source_A39=str(old39),
           source_records=source_records,cases=cases,selected_images=selected,selection_audit=selection,
           grid=previous['grid'],configs=configs,checkpoints=checkpoints,model_library_sources=libraries,runtime=previous['runtime'],
           native_policy=previous['native_policy'],source_time_gamma=previous['time_gamma'],
           identity_files=identities,worker_identity_files=workers,planned_method_nfe=7200,
           planned_diagnostic_nfe=3960,scientific_nfe_per_arm=300,
           observation=dict(domain='unit_[0,1]',sigma=.05,clipped=False),
           operators=dict(super_resolution=dict(scale_factor=4,noise_sigma=0.,adjoint='exact_DPS_Resizer'),
                          gaussian_deblur=dict(kernel_size=61,sigma=3.,noise_sigma=0.,padding='zero',adjoint='symmetric_kernel')),
           scope='Frozen physical response rule upper2/deadband.01, source time table and A39 feature definitions. '
                 'Twelve unused images within accessible history, reused across two tasks; one seed. '
                 'New exact-adjoint/unit-domain matched C in all four groups, not historical legacy SR baseline. '
                 'Original first270; only last30 guidance changes. Target-task time is leave-one-image-out '
                 'unlabeled task calibration; frozen source time is zero-target-calibration. '
                 'No PI online control, regularizer, hyperparameter scan, step/schedule or NFE-saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();source.signed(root/'plan.json',p);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A40 PREPARE PASS:',{d:[v['image_id'] for v in x] for d,x in selected.items()},
          ';24 cases;7200 method +3960 diagnostic =11160 calls; exact-adjoint C.',flush=True)


def verify_inputs(p):
    for path,digest in p['identity_files'].items():
        if source.digest(path)!=digest:raise ValueError('Frozen input changed: '+path)


def attempt(root,case,kind):
    folder=root/case['key']/kind;folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';target.mkdir();return target


def done(root,p,case,kind):
    found=[]
    for path in sorted((root/case['key']/kind).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        if (r['kind']!=kind or r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json')
                or r['calls']!=rules.expected_calls(p['grid'],kind) or r['nfe']!=len(r['calls'])
                or not r['gates'] or not all(g['passed'] for g in r['gates'].values())
                or any(source.digest(path.parent/n)!=digest for n,digest in r['artifacts'].items())
                or source.digest(r['measurement']['path'])!=r['measurement']['sha256']):
            raise ValueError('Invalid completed A40 record: '+str(path))
        if kind!='BASELINE':
            trained=done(root,p,case,'BASELINE')
            if trained is None or r['training_sha256']!=source.digest(trained[0]):raise ValueError('Training provenance differs')
        if kind=='MEASURE':
            expected={'C','STATE','NAIVE'}|{rules.risk.arm(k,g) for k,_ in rules.risk.POINTS for g in rules.risk.GAMMAS}
            if (set(r['metrics'])!=expected or [v['t'] for v in r['rows']]!=list(reversed(p['grid']))
                    or [v['t'] for v in r['state_trace']]!=list(reversed(p['grid']))):raise ValueError('Measured trace/arms differ')
        if kind=='TIME':
            measure=done(root,p,case,'MEASURE')
            if (measure is None or r['measure_sha256']!=source.digest(measure[0]) or
                    r['profiles_sha256']!=source.digest(root/'profiles.json')):raise ValueError('Time provenance differs')
            if (set(r['metrics'])!={'TIME_SOURCE','TIME_TASK'} or set(r['time_traces'])!={'TIME_SOURCE','TIME_TASK'}
                    or any([v['t'] for v in trace]!=list(reversed(p['grid'])) for trace in r['time_traces'].values())):
                raise ValueError('Time trace/arms differ')
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple completed attempts; no output selection')
    return found[0] if found else None


def operator_for(task,device):
    from modules.degradations import get_operator
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    config=dict(scale_factor=4,noise_sigma=0.) if task=='super_resolution' else dict(kernel_size=61,sigma=3.,noise_sigma=0.)
    return unit_operator(get_operator(task,device=device,**config)).to(device)


def operator_gate(operator,device,size=256):
    import torch
    from utils.diag_zaps_trace_audit import relative_error
    with P.precision.ieee(torch),torch.no_grad():
        probe=torch.sin(torch.arange(3*size*size,device=device,dtype=torch.float32)*.013).reshape(1,3,size,size)
        zero=torch.zeros_like(probe);mean=operator.H(probe)
        residual=mean-operator.H(zero)
        adjoint=operator.transpose(residual)
        with torch.enable_grad():
            x=probe.detach().clone().requires_grad_(True)
            actual=torch.autograd.grad((operator.H(x)*residual).sum(),x)[0]
        vjp=relative_error(adjoint,actual)
        left=float(((mean-operator.H(zero)).double()*residual.double()).sum())
        right=float((probe.double()*adjoint.double()).sum())
        inner=abs(left-right)/max(abs(left),abs(right),1e-12)
    if not math.isfinite(vjp+inner) or max(vjp,inner)>2e-5:raise RuntimeError('Exact affine operator adjoint gate failed')
    return dict(passed=True,vjp_relative_error=vjp,increment_adjoint_relative_error=inner,
                tolerance=2e-5,score_calls=0,scope='IEEE operator-only gate; production sampling restores native policy')


def measurement(root,case,operator,device):
    import fcntl
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_learning_rate_ablation import set_seed
    folder=root/'measurements'/case['dataset']/case['task'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/(case['image_id']+'.pt')
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            with scan.preserved_rng(),torch.no_grad():
                mean=operator.H(load_image_as_tensor(case['image']).to(device));set_seed(case['measurement_seed'])
                standard=torch.randn_like(mean)
                base.save_tensor(dict(mean=mean.cpu(),standard_noise=standard.cpu(),image_sha256=case['image_sha256'],
                                      seed=case['measurement_seed']),path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
    shape=(1,3,64,64) if case['task']=='super_resolution' else (1,3,256,256)
    if (saved['image_sha256']!=case['image_sha256'] or saved['seed']!=case['measurement_seed']
            or any(tuple(v.shape)!=shape or v.dtype!=torch.float32 or not torch.isfinite(v).all()
                   for v in (saved['mean'],saved['standard_noise']))):raise ValueError('Observation contract differs')
    return saved['mean'].to(device)+case['sigma']*saved['standard_noise'].to(device),dict(path=str(path),sha256=source.digest(path))


def training(root,p,case,model,device,operator,gate):
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    item=done(root,p,case,'BASELINE')
    if item:return item
    folder=attempt(root,case,'BASELINE');path=folder/'run.json';started=time.perf_counter()
    y,meta=measurement(root,case,operator,device)
    r=dict(kind='BASELINE',case=case,plan_sha256=source.digest(root/'plan.json'),measurement=meta,gates={})
    with scan.counted(model,path,r,300),scan.preserved_rng(),model_rng_guard(model.model,device):
        z=base.baseline.snapshot_class()(model,operator,img_size=256,**p['configs'][case['dataset']][case['task']])
        z.tau=torch.tensor(p['grid'],device=device)
        set_seed(case['sampling_seed']);initial=torch.randn(1,3,256,256,device=device)
        for _ in range(290):torch.randn_like(initial)
        expected=rng_state(device);set_seed(case['sampling_seed']);torch.cuda.synchronize(device);now=time.perf_counter()
        r['loss_history']=z.optimize(y,verbose=False,x0_gt=None,scheduler=None)
        torch.cuda.synchronize(device);r['sampling_optimization_seconds']=time.perf_counter()-now
        if (r['calls']!=rules.expected_calls(p['grid'],'BASELINE') or z._last_nfe!=300 or z.audit_state is None
                or not torch.equal(z._last_opt_noise,initial) or not same_rng(expected,rng_state(device))
                or len(r['loss_history'])!=10 or any(not math.isfinite(v) for v in r['loss_history'])
                or any(v.grad is not None for v in model.model.parameters())):raise RuntimeError('Original optimization gate failed')
        base.save_tensor(z.audit_state,folder/'pre_last_state.pt');base.save_tensor(z._last_opt_x0.detach().cpu(),folder/'last_opt.pt')
        r['artifacts']={n:source.digest(folder/n) for n in ('pre_last_state.pt','last_opt.pt')}
        r['gates']=dict(operator=gate,budget_rng=dict(passed=True),GT_not_used=dict(passed=True))
    r.update(status='complete',seconds=time.perf_counter()-started);source.signed(path,r)
    print('A40 BASELINE:',case['key'],'300 calls complete.',flush=True);return path,r


def setup_sampling(root,p,case,model,device,operator):
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection,restore_sampling_rng
    trained=done(root,p,case,'BASELINE')
    if trained is None:raise ValueError('Missing baseline')
    state=torch.load(trained[0].parent/'pre_last_state.pt',map_location='cpu',weights_only=True)
    original=torch.load(trained[0].parent/'last_opt.pt',map_location='cpu',weights_only=True)
    y,meta=measurement(root,case,operator,device)
    if meta!=trained[1]['measurement']:raise ValueError('Paired observation differs')
    z=ZAPS(model,operator,img_size=256,**p['configs'][case['dataset']][case['task']]);z.tau=state['tau'].to(device)
    with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
    if z.tau.tolist()!=p['grid']:raise ValueError('Grid differs')
    positions=source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(p['grid'])))
    with scan.preserved_rng():
        mapping=saved_cuda_selection(state,device);restore_sampling_rng(state,device,mapping['saved_sampling_index'])
        noises=[torch.randn_like(state['init_noise'].to(device)) for _ in range(29)]
    return trained,state,original,y,meta,z,positions,noises


def sample(root,p,case,model,device,operator,kind):
    import torch
    from unittest.mock import patch
    from modules.main_single import load_image_as_tensor
    from utils.diag_cached_guidance_geometry import physical_descent
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    if done(root,p,case,kind):return
    trained,state,original,y,meta,z,positions,noises=setup_sampling(root,p,case,model,device,operator)
    folder=attempt(root,case,kind);path=folder/'run.json';started=time.perf_counter()
    r=dict(kind=kind,case=case,plan_sha256=source.digest(root/'plan.json'),training_sha256=source.digest(trained[0]),
           measurement=meta,gates={},arm_seconds={},sensor_seconds=0.)
    profiles=None
    if kind=='TIME':
        profiles=source.load_signed(root/'profiles.json');measure=done(root,p,case,'MEASURE')
        if measure is None:raise ValueError('Missing measurement phase')
        r.update(measure_sha256=source.digest(measure[0]),profiles_sha256=source.digest(root/'profiles.json'))
    external=rng_state(device);outputs={};anchors={};history=rules.risk.History();parity=None
    limit=rules.MEASURE_CALLS if kind=='MEASURE' else rules.TIME_CALLS
    with scan.counted(model,path,r,limit):
        with torch.no_grad(),model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
            if kind=='MEASURE':
                x=state['init_noise'].to(device);r['rows']=[];torch.cuda.synchronize(device);now=time.perf_counter()
                for k,t in enumerate(positions.nominal):
                    f=source.fields.field(z,y,x,t,positions);sensor_started=time.perf_counter()
                    readout=rules.response.sensor(z,y,f['clean'],f['q'],t,case['sigma'])
                    gradient,loss=physical_descent(z.A,f['clean'],y,sigma=case['sigma'])
                    observed=history.append(float(positions.u[t]),f['clean'][0].cpu().numpy(),f['q'][0].cpu().numpy(),
                                            gradient[0].cpu().numpy(),readout,f['clip_fraction'])
                    r['sensor_seconds']+=time.perf_counter()-sensor_started
                    r['rows'].append(dict(step=k,t=t,u=float(positions.u[t]),state=observed,physical_loss=loss))
                    if k in {v[0] for v in rules.risk.POINTS}:
                        anchors[k]=(x.clone(),{n:f[n].clone() for n in ('clean','q')})
                    x=rules.risk.advance(z,x,f,positions,k,noises)
                torch.cuda.synchronize(device);r['arm_seconds']['C']=time.perf_counter()-now;outputs['C']=x.cpu()
                parity=relative_error(outputs['C'],original)
                if not math.isfinite(parity) or parity>1e-6:raise RuntimeError('Original C replay differs')
                for k,_ in rules.risk.POINTS:
                    for gamma in rules.risk.GAMMAS:
                        name=rules.risk.arm(k,gamma);torch.cuda.synchronize(device);now=time.perf_counter()
                        center,f=anchors[k];x=rules.risk.advance(z,center,f,positions,k,noises,gamma)
                        for j in range(k+1,30):
                            f=source.fields.field(z,y,x,positions.nominal[j],positions)
                            x=rules.risk.advance(z,x,f,positions,j,noises)
                        torch.cuda.synchronize(device);r['arm_seconds'][name]=time.perf_counter()-now;outputs[name]=x.cpu()
                control_arms=('STATE',)
            else:control_arms=('TIME_SOURCE','TIME_TASK')
            for arm in control_arms:
                x=state['init_noise'].to(device);trace=[];sensor_total=0.
                torch.cuda.synchronize(device);now=time.perf_counter()
                for k,t in enumerate(positions.nominal):
                    f=source.fields.field(z,y,x,t,positions)
                    if arm=='STATE':
                        torch.cuda.synchronize(device);sensor_started=time.perf_counter()
                        readout=rules.response.sensor(z,y,f['clean'],f['q'],t,case['sigma'])
                        sensor_total+=time.perf_counter()-sensor_started;gamma=readout['gamma']
                    else:
                        readout={};gamma=(p['source_time_gamma'][k] if arm=='TIME_SOURCE' else profiles['values'][case['key']]['gamma'][k])
                    trace.append(dict(readout,step=k,t=t,gamma=gamma))
                    x=rules.risk.advance(z,x,f,positions,k,noises,gamma)
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite controlled state; no replacement')
                torch.cuda.synchronize(device);r['arm_seconds'][arm]=time.perf_counter()-now;outputs[arm]=x.cpu()
                if arm=='STATE':r.update(state_trace=trace,state_sensor_seconds=sensor_total)
                else:r.setdefault('time_traces',{})[arm]=trace
            if kind=='MEASURE':
                naive=2*y-1
                if case['task']=='super_resolution':
                    naive=torch.nn.functional.interpolate(naive,size=(256,256),mode='bicubic',align_corners=False)
                outputs['NAIVE']=naive.cpu()
        if (r['calls']!=rules.expected_calls(p['grid'],kind) or not same_rng(external,rng_state(device))
                or P.precision.precision_state(torch)!=p['native_policy']
                or any(v.grad is not None for v in model.model.parameters())
                or any(not torch.isfinite(out).all() for out in outputs.values())
                or not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('Budget/RNG/native precision/parameters gate failed')
        base.save_tensor(outputs,folder/'outputs.pt');r['artifacts']={'outputs.pt':source.digest(folder/'outputs.pt')}
        r['status']='sampled';source.WRITE(path,r)
        gt=load_image_as_tensor(case['image']).cpu()
        metrics={a:dict(loss=float((y-z.A.H(out.to(device))).double().square().mean()/(2*case['sigma']**2)),
                        **compute_all_metrics(out,gt,lpips_net='alex')) for a,out in outputs.items()}
        if any(not math.isfinite(v) for m in metrics.values() for v in m.values()):raise RuntimeError('Nonfinite metric')
        r.update(metrics=metrics,gates=dict(budget_noise_rng=dict(passed=True),parameters_native_precision=dict(passed=True),
                                          GT_after_outputs=dict(passed=True),model_grads_absent=dict(passed=True)))
        if kind=='MEASURE':r['gates']['original_C']=dict(passed=True,relative_error=parity)
    r.update(status='complete',seconds=time.perf_counter()-started);source.signed(path,r)
    print('A40',kind+':',case['key'],r['nfe'],'calls complete.',flush=True)


def make_profiles(args):
    root=Path(args.root).resolve();p=source.check_plan(root);records=[];identities={}
    for case in p['cases']:
        item=done(root,p,case,'MEASURE')
        if item is None:raise ValueError('Need all state trajectories before task-time calibration')
        records.append(item[1]);identities[str(item[0])]=source.digest(item[0])
    result=dict(experiment='A40_task_time_profiles',plan_sha256=source.digest(root/'plan.json'),
                values=rules.profiles(records,p['grid']),source_records=identities,GT_used=False,
                calibration='Five OTHER images of same task/dataset, no quality labels; target-task calibration, not zero-shot.')
    path=root/'profiles.json'
    if path.exists():
        if source.load_signed(path)!=result:raise ValueError('Frozen profiles differ')
    else:source.signed(path,result)
    print('A40 PROFILES PASS:24 leave-one-image-out task profiles, no GT quality labels; new NFE0.',flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);verify_inputs(p)
    if model_sources()!=p['model_library_sources']:raise ValueError('Resolved model library differs')
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        kind='MEASURE' if args.phase=='measure' else 'TIME'
        cases=[c for c in p['cases'] if c['image_index']%2==args.shard]
        if all(done(root,p,c,kind) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                     gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                     TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Native runtime differs')
        if kind=='TIME':
            profile=source.load_signed(root/'profiles.json')
            if profile['plan_sha256']!=source.digest(root/'plan.json'):raise ValueError('Profiles refer to other plan')
            for path,digest in profile['source_records'].items():
                if source.digest(path)!=digest:raise ValueError('Calibration trajectory changed')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        for dataset in rules.DATASETS:
            current=[c for c in cases if c['dataset']==dataset]
            if all(done(root,p,c,kind) for c in current):continue
            with scan.preserved_rng():
                model=load_diffusion_model(dataset,args.device);model.model.eval()
                if (source.digest(model.ckpt_path)!=p['checkpoints'][dataset]['sha256'] or model.num_steps!=1000
                        or any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Model identity/dtype differs')
                for task in rules.TASKS:
                    operator=operator_for(task,args.device);gate=operator_gate(operator,args.device)
                    for case in (c for c in current if c['task']==task):
                        if kind=='MEASURE':training(root,p,case,model,args.device,operator,gate)
                        sample(root,p,case,model,args.device,operator,kind)
                    del operator
                del model
            gc.collect();torch.cuda.empty_cache()
        source.check_plan(root)


def pictures(path,r,time_item):
    import torch
    from PIL import Image,ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    outputs=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
    outputs.update(torch.load(time_item[0].parent/'outputs.pt',map_location='cpu',weights_only=True))
    names=('GT',)+rules.ARMS;sheet=Image.new('RGB',(20+276*len(names),680),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,10),r['case']['key']+' | fixed grid; each scientific arm300 NFE',fill='black',font=font(18))
    for j,name in enumerate(names):
        raw=Image.fromarray(_to_uint8_numpy(load_image_as_tensor(r['case']['image']) if name=='GT' else outputs[name]));full=raw.copy();x=20+276*j
        ImageDraw.Draw(full).rectangle((64,48,192,176),outline='#f8bd46',width=2)
        draw.text((x,45),name,fill='black',font=font(17))
        if name!='GT':
            m=r['metrics'][name];draw.text((x,70),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}',fill='black',font=font(14))
            draw.text((x,90),f'LPIPS {m["lpips"]:.4f}',fill='black',font=font(14))
        sheet.paste(full,(x,115));sheet.paste(raw.crop((64,48,192,176)).resize((256,256),Image.Resampling.NEAREST),(x,390))
    draw.text((20,660),'TIME_TASK additionally uses five other-image calibration trajectories; acquisition costs reported separately.',
              fill='black',font=font(12))
    sheet.save(path.parent/'policy_comparison.png')


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);records=[];items=[]
    for case in p['cases']:
        measured=done(root,p,case,'MEASURE');timed=done(root,p,case,'TIME')
        if measured and timed:
            r=dict(measured[1],metrics={**measured[1]['metrics'],**timed[1]['metrics']},time_traces=timed[1]['time_traces'])
            records.append(r);items.append((measured[0],r,timed))
    old=[]
    for path in p['source_records']:
        if source.digest(path)!=p['identity_files'][path]:raise ValueError('Frozen A39 source differs')
        old.append(source.load_signed(Path(path)))
    result=rules.summarize(records,old);attempts=[source.read(f) for f in root.glob('*/*/attempt_*/run.json')]
    result.update(experiment=p['experiment'],completed_cases=len(records),planned_cases=24,
                  method_nfe=sum(r['nfe'] for r in attempts if r['kind']=='BASELINE'),
                  diagnostic_nfe=sum(r['nfe'] for r in attempts if r['kind']!='BASELINE'),
                  planned_method_nfe=7200,planned_diagnostic_nfe=3960,scientific_nfe_per_arm=300,
                  partial_attempts=sum(r['status']!='complete' for r in attempts),
                  cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] in ('running','sampled') for r in attempts),
                  source_scope=p['scope'],
                  task_time_calibration='Other five images/condition require state trajectories; not a free standalone300-NFE method. '
                                        'All acquisition/replay costs above; no end-to-end speed claim.')
    source.signed(root/'summary.json',result)
    if args.archive:
        for item in items:pictures(*item)
    compact={k:v for k,v in result.items() if k!='groups'}
    compact['groups']=[]
    for g in result['groups']:
        out={k:v for k,v in g.items() if k not in ('contrasts','frozen_indicator_analysis','single_pulse_opportunity')}
        if 'baseline_context' in out:
            out['baseline_context']={**out['baseline_context'],'C_minus_naive':
                {k:v for k,v in out['baseline_context']['C_minus_naive'].items() if k!='per_image'}}
        out['contrasts']={n:{k:v for k,v in c.items() if k!='per_image'} for n,c in g.get('contrasts',{}).items()}
        if 'frozen_indicator_analysis' in g:
            out['indicator_lpips']={mode:targets['lpips'] for mode,targets in g['frozen_indicator_analysis'].items()}
            out['single_pulse_opportunity']=[dict(source_t=v['source_t'],best_fixed=v['best_fixed_feasible'],
                oracle_headroom=v['oracle_headroom_over_best_fixed']) for v in g['single_pulse_opportunity']]
        compact['groups'].append(out)
    print('A40 Summary:',json.dumps(compact,ensure_ascii=False),flush=True);print('Records:',root,flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A40':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)
    if len(records)!=24:raise SystemExit('A40 incomplete; finished attempts retained. Read phase logs.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a36-dir',default=base.ROOT);p.add_argument('--a39-dir',default=a39.ROOT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256')
    p.add_argument('--prepare',action='store_true');p.add_argument('--profiles',action='store_true')
    p.add_argument('--phase',choices=('measure','time'),default='measure');p.add_argument('--shard',type=int,choices=(0,1))
    p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true');args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.profiles:make_profiles(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
