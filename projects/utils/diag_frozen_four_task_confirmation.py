"""A46 frozen four-task rule on six new images, original baselines + paired interventions."""
import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_balanced_state_calibration as development
from utils import frozen_four_task_confirmation as rules

parent=development.parent
source,a42,old,P,scan=parent.source,parent.a42,parent.old,parent.P,parent.scan
a40=old.parent
sensor,control,mix=parent.sensor,parent.control,development.rules
write,signed=parent.write,parent.signed
invariant=parent.invariant
ROOT='/data0/lzy/ZAPS-results/diag_frozen_four_task/A46_20261025'
CARD=PROJECTS/'experiment_cards/A46_frozen_four_task_confirmation_20261025.md'
BRANCH='results/ffhq-frozen-four-task-20261025'


def prepare(args):
    root,previous=Path(args.root).resolve(),Path(args.a45_dir).resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):raise ValueError('Source/output overlap')
    if root.exists():
        p=source.check_plan(root);a42.verify(p)
        if p['source_A45']!=str(previous):raise ValueError('Frozen source differs')
        print('A46 frozen new-image plan reused; no reselection.',flush=True);return
    before=source.check_plan(previous);a42.verify(before);summary=source.load_signed(previous/'summary.json')
    items=[development.done(previous,before,c) for c in before['cases']]
    if not all(items) or len(items)!=20:raise ValueError('Need all20 completed A45 development cases')
    weights={};matched={};identities={str(previous/'plan.json'):source.digest(previous/'plan.json'),
        str(previous/'summary.json'):source.digest(previous/'summary.json')}
    for task in rules.WEIGHTS:
        rows=[r for _,r in items if r['case']['task']==task];choice=mix.select_weight(rows)['chosen_lambda']
        published=next(g for g in summary['groups'] if g['task']==task)['development']['chosen_lambda']
        if choice!=published or choice!=rules.WEIGHTS[task]:raise ValueError('A45 weight lock differs: '+task)
        weights[task]=choice;gammas=[]
        for path,r in [v for v in items if v[1]['case']['task']==task]:
            identities[str(path)]=source.digest(path)
            if choice==.5:gammas.append(r['gamma'])
            else:
                finish=Path(before['a44_records'][r['case']['key']]['finish']);src=source.load_signed(finish)
                identities[str(finish)]=source.digest(finish)
                gammas.append(src['gamma']['PHYSICAL' if choice==0. else 'STATE'])
        matched[task]=float(statistics.median(gammas))
        if abs(matched[task]-rules.MATCHED[task])>1e-9:raise ValueError('A45 median multiplier differs')
    a41=Path(before['source_A41']);p41=source.check_plan(a41)
    identities[str(a41/'plan.json')]=source.digest(a41/'plan.json')
    identities[str(a41/'motion_kernel.pt')]=source.digest(a41/'motion_kernel.pt')
    offsets={}
    for task in rules.WEIGHTS:
        values={c['measurement_seed']-int(c['image_id']) for c in p41['cases'] if c['task']==task}
        if len(values)!=1:raise ValueError('Ambiguous source measurement seeding')
        offsets[task]=next(iter(values))
    data=Path(before['cases'][0]['image']).resolve().parent
    selection,pool=scan.history_pool(data,root,6)
    selected={c['image_id']:{n:c[n] for n in ('image','image_id','image_sha256')} for c in pool}
    if len(selected)!=6:raise ValueError('Need six unique unused images')
    cases=[dict(image,dataset='ffhq',task=task,image_index=i,sigma=.05,sampling_seed=1000,
        measurement_seed=offsets[task]+int(image['image_id']),mask_seed=2001000+int(image['image_id']),
        key=f'ffhq_{task}_{image["image_id"]}_s1000') for i,image in enumerate(sorted(selected.values(),key=lambda c:c['image_id']))
        for task in rules.WEIGHTS]
    for c in cases:identities[c['image']]=c['image_sha256']
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a46_20261025.sh',PROJECTS/'utils/test_frozen_four_task_confirmation.py']
    p={k:before[k] for k in ('runtime','native_policy','checkpoint','model_library_sources','grid')}
    p.update(experiment='A46_frozen_four_task_confirmation',source_A45=str(previous),cases=cases,
        configs=p41['configs'],selected_images=sorted(selected),selection_scope=selection['scope'],
        weights_frozen=weights,matched_fixed=matched,fixed_reference=before['fixed_reference'],measurement_offsets=offsets,
        step=24,t=319,next_t=218,identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=7200,planned_diagnostic_nfe_max=2040,planned_total_nfe_max=9240,
        limits='Six images unused in accessible experimental histories, one seed, four related task conditions. '
               'Frozen t319 single pulse. No retuning on these images, broad generalization, full online loop or NFE saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();shutil.copyfile(a41/'motion_kernel.pt',root/'motion_kernel.pt')
    p['identity_files'][str(root/'motion_kernel.pt')]=source.digest(root/'motion_kernel.pt')
    signed(root/'selection_audit.json',selection);p['identity_files'][str(root/'selection_audit.json')]=source.digest(root/'selection_audit.json')
    signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    shutil.copyfile(previous/'summary.json',snapshot/'A45_summary.json')
    for c in before['cases']:
        source_path=Path(before['a44_records'][c['key']]['finish']);shutil.copyfile(source_path,snapshot/('A44_'+c['key']+'.json'))
    print('A46 PREPARE PASS: new image IDs=',p['selected_images'],'; weights=',weights,'; matched fixed=',matched,
        '; NEW7200 method +<=2040 diagnostic =<=9240 calls.',flush=True)


def phase_done(root,p,case,phase):
    found=[];times=list(reversed(p['grid']));s=p['next_t']
    for path in sorted((root/case['key']).glob(phase+'_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path);trained=a40.done(root,p,case,'BASELINE')
        expected=(times+[s]*(6+4*r['rank']+len(r['new_query_gammas'])) if phase=='choose' else
            [t for _ in r['new_suffix_gammas'] for t in times[25:]])
        if (trained is None or r['training_sha256']!=source.digest(trained[0]) or r['case']!=case or r['phase']!=phase or
                r['plan_sha256']!=source.digest(root/'plan.json') or r['calls']!=expected or r['nfe']!=len(expected) or
                r['nfe']+r['prior_recorded_nfe']>rules.cap(phase,case['task']) or
                not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):raise ValueError('Invalid A46 phase: '+str(path))
        if phase=='finish' and r['allocation_sha256']!=source.digest(root/'allocation.json'):raise ValueError('Allocation differs')
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple completed phases; no output selection')
    return found[0] if found else None


def attempt(root,p,case,phase):
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    prior=sum(source.read(f)['nfe'] for f in folder.glob(phase+'_*/run.json'))
    out=folder/f'{phase}_{len(list(folder.glob(phase+"_*"))):03d}';out.mkdir()
    trained=a40.done(root,p,case,'BASELINE')
    r=dict(case=case,phase=phase,prior_recorded_nfe=prior,plan_sha256=source.digest(root/'plan.json'),
        training_sha256=source.digest(trained[0]),gates={})
    return out,r


@contextmanager
def counted(model,path,r):
    started=time.perf_counter();r.update(status='running',calls=[],nfe=0)
    def save():r['seconds']=time.perf_counter()-started;write(path,r)
    def hook(_m,inputs):
        if r['nfe']+r['prior_recorded_nfe']>=rules.cap(r['phase'],r['case']['task']):raise RuntimeError('A46 phase budget exhausted')
        r['calls'].append(int(inputs[1].flatten()[0]));r['nfe']+=1;save()
    save();handle=model.model.register_forward_pre_hook(hook)
    try:yield
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:handle.remove()


@contextmanager
def baseline_counted(model,path,r,limit):
    if limit!=300:raise ValueError('Original baseline budget differs')
    r['phase']='BASELINE';r['prior_recorded_nfe']=sum(source.read(f)['nfe'] for f in path.parent.parent.glob('attempt_*/run.json'))
    with counted(model,path,r):yield


def setup(root,p,case,model,device):
    operator,mask=old.make_operator(root,case,device);gate=a40.operator_gate(operator,device)
    _,state,original,y,_,z,positions,noises=a40.setup_sampling(root,p,case,model,device,operator)
    return operator,mask,gate,state,original,y,z,positions,noises


# Same A44 sampler/target arithmetic; new baselines and selected-objective branch.
def choose(root,p,case,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    operator,mask,opgate,state,original,y,z,positions,noises=setup(root,p,case,model,device)
    folder,r=attempt(root,p,case,'choose');path=folder/'run.json';external=rng_state(device)
    arr=lambda x:x.detach().cpu().numpy()
    m=arr(mask) if mask is not None else None;obs=sensor.active(arr(y),m);k=p['step'];t=p['t'];s=p['next_t']
    if positions.nominal[k]!=t or positions.nominal[k+1]!=s:raise ValueError('Frozen time differs')
    r['new_query_gammas']=[]
    with counted(model,path,r),torch.no_grad(),model_rng_guard(model.model,device),\
            patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
        x=state['init_noise'].to(device);center=f=None
        for j,tj in enumerate(positions.nominal):
            field=source.fields.field(z,y,x,tj,positions)
            if j==k:center=x.clone();f={n:field[n].clone() for n in ('clean','q')}
            x=old.parent.rules.risk.advance(z,x,field,positions,j,noises)
        parity=relative_error(x.cpu(),original)
        if not math.isfinite(parity) or parity>1e-6:raise RuntimeError('C replay parity failed')
        with P.precision.ieee(torch):
            cleans={};observations={};raw0=reference=None
            for g in sensor.GAMMAS:
                xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                raw,clean=a42.raw_field(z,xx,s);cleans[g]=arr(clean);observations[g]=arr(operator.H(clean))
                if g==0.:raw0,reference=raw,xx
            repeated,_=a42.raw_field(z,reference,s);repeat_error=relative_error(raw0,repeated)
            if not math.isfinite(repeat_error) or repeat_error>1e-7:raise RuntimeError('Raw repeat failed')
            raw_obs=arr(operator.H(raw0));gradient=operator.transpose(y-operator.H(raw0))
            V=sensor.basis([cleans[2.]-cleans[0.],arr(gradient)]);r['rank']=V.shape[1]
            matrices,derivatives,span=a42.covariance_probes(lambda xx:a42.raw_field(z,xx,s)[0],reference,V,z.dm.alphas_cumprod[s])
            cov,B=sensor.covariance_gate(matrices,span);target={};state_readout={};W=[]
            for j in range(V.shape[1]):
                v=torch.from_numpy(V[:,j].copy()).reshape(raw0.shape).to(device=device,dtype=raw0.dtype)
                W.append(sensor.active(arr(.5*operator.base.H(v)),m))
            W=np.stack(W,axis=1) if W else np.empty((len(obs),0))
            if cov['usable']:
                for name,b in (('ISO',np.eye(len(B))*float(np.trace(B))/len(B)),('STATE',B)):
                    target[name],state_readout[name]=sensor.gaussian_target(arr(raw0),V,b,W,obs-sensor.active(raw_obs,m),case['sigma'])
            def readout(clean,observation):
                a=np.asarray(clean,dtype=float).reshape(-1)
                return dict(loss=float(np.mean((obs-sensor.active(observation,m))**2)/(2*case['sigma']**2)),
                    risks={name:float(np.mean((a-v)**2)) for name,v in target.items()})
            values={g:readout(cleans[g],observations[g]) for g in sensor.GAMMAS}
            budget=control.noise_budget(float(z.dm.alphas_cumprod[t]),float(z.dm.alphas_cumprod[s]),float(f['q'].double().square().mean().sqrt()))
            def evaluate(g):
                if g not in values:
                    r['new_query_gammas'].append(g);xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                    try:
                        _,clean=a42.raw_field(z,xx,s);values[g]=readout(arr(clean),arr(operator.H(clean)));cleans[g]=arr(clean)
                    except RuntimeError as error:
                        if str(error)!='Nonfinite raw denoiser':raise
                        values[g]=dict(loss=None,risks={},error=str(error))
                return values[g]
            choices={}
            for name in ('PHYSICAL','ISO','STATE'):
                curve={g:values[g]['loss'] if name=='PHYSICAL' else values[g]['risks'].get(name,0.) for g in sensor.GAMMAS}
                def objective(g,name=name):
                    value=evaluate(g);return dict(loss=value['loss'],objective=value['loss'] if name=='PHYSICAL' else value['risks'].get(name))
                choices[name]=control.continuous_choice(curve,budget['cap'],objective,
                    usable=budget['q_resolved'] and (name=='PHYSICAL' or name in target))
            weight=p['weights_frozen'][case['task']]
            if weight in (0.,1.):
                choices['SELECTED']=dict(choices['PHYSICAL' if weight==0. else 'STATE'])
            elif cov['usable']:
                norm=mix.normalization(values)
                curve={g:mix.objective(values[g],norm,weight) for g in sensor.GAMMAS}
                def mixed_objective(g):
                    value=evaluate(g)
                    return dict(loss=value['loss'],objective=mix.objective(value,norm,weight))
                choices['SELECTED']=control.continuous_choice(curve,budget['cap'],mixed_objective,usable=budget['q_resolved'])
            else:choices['SELECTED']=dict(gamma=1.,reason='SOURCE_COVARIANCE_UNRESOLVED',attempts=[])
        r.update(covariance=cov,budget=budget,state_readout=state_readout,choices=choices,
            gamma={name:v['gamma'] for name,v in choices.items()},query_readouts={str(g):v for g,v in values.items()})
        expected_calls=list(positions.nominal)+[s]*(6+4*r['rank']+len(r['new_query_gammas']))
        invariant(model,z,state,device,external,p)
        if r['calls']!=expected_calls:raise RuntimeError('Choose call trace differs')
        cache=dict(source_x=center.cpu(),source_clean=f['clean'].cpu(),source_q=f['q'].cpu(),reference=reference.cpu(),
            raw_mean=raw0.cpu(),basis=torch.from_numpy(V),B=torch.from_numpy(B),W=torch.from_numpy(W),
            matrices=[torch.from_numpy(v) for v in matrices],derivatives=[torch.from_numpy(v) for v in derivatives],
            candidate_clean={str(g):torch.from_numpy(v.copy()) for g,v in cleans.items()})
        old.parent.base.save_tensor(cache,folder/'probes.pt')
        signed(folder/'decisions_before_labels.json',dict(case=case,gamma=r['gamma'],choices=choices,plan_sha256=r['plan_sha256']))
        r['artifacts']={n:source.digest(folder/n) for n in ('probes.pt','decisions_before_labels.json')}
        r['gates']=dict(operator=opgate,C_replay=dict(passed=True,relative_error=parity),
            raw_repeat=dict(passed=True,relative_error=repeat_error),training_bound=dict(passed=True),
            calls_rng_parameters=dict(passed=True),no_FINAL_labels_used=dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A46 CHOOSE:',case['key'],'gamma=',r['gamma'],'new calls=',r['nfe'],flush=True)



def finish(root,p,case,model,device):
    import torch
    from unittest.mock import patch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    allocation=source.load_signed(root/'allocation.json')
    for path,sha in allocation['choices'].items():
        if source.digest(path)!=sha:raise ValueError('Frozen choice record differs')
    chosen=phase_done(root,p,case,'choose');cache=torch.load(chosen[0].parent/'probes.pt',map_location='cpu',weights_only=True)
    operator,mask,opgate,state,original,y,z,positions,noises=setup(root,p,case,model,device)
    folder,r=attempt(root,p,case,'finish');path=folder/'run.json';external=rng_state(device)
    assigned=allocation['assignments'][case['key']];r.update(assigned,allocation_sha256=source.digest(root/'allocation.json'),
        new_suffix_gammas=[],evaluation={})
    center=cache['source_x'].to(device);f={n:cache['source_'+n].to(device) for n in ('clean','q')};k=p['step']
    outputs={};metrics={};arr=lambda x:x.detach().cpu().numpy();m=arr(mask) if mask is not None else None
    obs=sensor.active(arr(y),m);gt=None
    with counted(model,path,r),torch.no_grad(),model_rng_guard(model.model,device),\
            patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
        reference=old.parent.rules.risk.advance(z,center,f,positions,k,noises,0.)
        parity=relative_error(reference.cpu(),cache['reference'])
        if not math.isfinite(parity) or parity>1e-7:raise RuntimeError('Choose/finish anchor differs')
        gt=load_image_as_tensor(case['image']).cpu();residual_C=(y-operator.H(original.to(device))).double()
        with scan.preserved_rng():baseline=dict(loss=float(residual_C.square().mean()/(2*case['sigma']**2)),
            **compute_all_metrics(original,gt,lpips_net='alex'))
        r['C_metrics']=baseline
        for name in rules.POLICIES:
            g=r['gamma'][name];key=f'G{g:g}'
            if key not in outputs:
                if g==1.:
                    outputs[key]=original;metrics[key]=baseline
                else:
                    r['new_suffix_gammas'].append(g);xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                    for j in range(k+1,len(positions.nominal)):
                        field=source.fields.field(z,y,xx,positions.nominal[j],positions)
                        xx=old.parent.rules.risk.advance(z,xx,field,positions,j,noises)
                    if not torch.isfinite(xx).all():raise RuntimeError('Nonfinite final output')
                    outputs[key]=xx.cpu()
                    if gt is None:gt=load_image_as_tensor(case['image']).cpu()
                    residual=(y-operator.H(xx)).double()
                    with scan.preserved_rng():metrics[key]=dict(loss=float(residual.square().mean()/(2*case['sigma']**2)),
                        **compute_all_metrics(outputs[key],gt,lpips_net='alex'))
            metric=metrics[key];res=obs-sensor.active(arr(operator.H(outputs[key].to(device))),m)
            r['evaluation'][name]=dict(output=key,metrics=metric,informative_loss=float(np.mean(res**2)/(2*case['sigma']**2)),
                delta={n:metric[n]-baseline[n] for n in rules.METRICS})
        invariant(model,z,state,device,external,p)
        if r['calls']!=[t for _ in r['new_suffix_gammas'] for t in positions.nominal[25:]]:raise RuntimeError('Finish trace differs')
        if any(not math.isfinite(v) for ev in r['evaluation'].values() for v in ev['metrics'].values()):raise RuntimeError('Nonfinite quality')
        old.parent.base.save_tensor(outputs,folder/'outputs.pt');r['artifacts']={'outputs.pt':source.digest(folder/'outputs.pt')}
        r['gates']=dict(operator=opgate,anchor=dict(passed=True,relative_error=parity),calls_rng_parameters=dict(passed=True),
            all24_decisions_frozen_before_evaluation=dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A46 FINISH:',case['key'],'dPSNR=',{n:round(r['evaluation'][n]['delta']['psnr'],4) for n in rules.POLICIES},
          'new calls=',r['nfe'],flush=True)


def freeze(root,p):
    items=[phase_done(root,p,c,'choose') for c in p['cases']]
    if not all(items):raise ValueError('Need all24 completed choices before final evaluation')
    choices={r['case']['key']:r['gamma'] for _,r in items}
    value=dict(plan_sha256=source.digest(root/'plan.json'),choices={str(path):source.digest(path) for path,_ in items},
        assignments=rules.allocation(p['cases'],choices,p['fixed_reference'],p['matched_fixed']),GT_used=False)
    path=root/'allocation.json'
    if path.exists():
        if source.load_signed(path)!=value:raise ValueError('Frozen allocation differs')
    else:signed(path,value)
    print('A46 FREEZE PASS: all24 selected multipliers, matched fixed references and cyclic reassignment frozen.',flush=True)


def worker(args):
    import fcntl
    import torch
    from unittest.mock import patch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(phase_done(root,p,c,args.phase) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Runtime changed')
        if a40.model_sources()!=p['model_library_sources']:raise ValueError('Model library changed')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoint']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Checkpoint/dtype changed')
            for case in cases:
                if phase_done(root,p,case,args.phase):continue
                if args.phase=='choose':
                    operator,_=old.make_operator(root,case,args.device);gate=a40.operator_gate(operator,args.device)
                    # Original training arithmetic, with an additional cumulative-attempt cap.
                    with patch.object(scan,'counted',baseline_counted):a40.training(root,p,case,model,args.device,operator,gate)
                    choose(root,p,case,model,args.device)
                else:finish(root,p,case,model,args.device)
                gc.collect();torch.cuda.empty_cache()


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    items=[item for c in p['cases'] if (item:=phase_done(root,p,c,'finish')) is not None]
    groups=rules.summarize([r for _,r in items]);complete=len(items)==24
    baseline_attempts=[source.read(f) for f in root.glob('*/BASELINE/attempt_*/run.json')]
    diagnostic_attempts=[source.read(f) for f in root.glob('*/*_*/run.json')]
    method=sum(r['nfe'] for r in baseline_attempts);diagnostic=sum(r['nfe'] for r in diagnostic_attempts)
    quality=sum(g['quality_candidate'] for g in groups);matching=sum(g['state_matching_candidate'] for g in groups)
    result=dict(experiment=p['experiment'],classification=('FOUR_TASK_FROZEN_QUALITY_TRANSFER_CANDIDATE' if quality==4 else
        'TASK_SPECIFIC_OR_MIXED_FROZEN_QUALITY_TRANSFER') if complete else 'INCOMPLETE_PILOT',
        state_matching_classification=('MATCHING_SIGNAL_IN_'+str(matching)+'_TASKS') if complete else 'INCOMPLETE_PILOT',
        completed_cases=len(items),planned_cases=24,image_clusters=6,new_image_ids=p['selected_images'],
        method_nfe=method,diagnostic_nfe=diagnostic,total_new_nfe=method+diagnostic,
        planned_method_nfe=7200,planned_diagnostic_nfe_max=2040,planned_total_nfe_max=9240,
        partial_attempts=sum(r['status']!='complete' for r in baseline_attempts+diagnostic_attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in baseline_attempts+diagnostic_attempts),
        quality_candidate_tasks=quality,state_matching_candidate_tasks=matching,groups=groups,
        weights_frozen=p['weights_frozen'],matched_fixed=p['matched_fixed'],limits=p['limits']+
        ' Quality transfer and state matching are separate. Scientific sampling300; paid selected sensor<=317, fixed300; '
        'offline reassignment is not a deployable extra-free policy. No statistical significance claim from six image clusters.')
    signed(root/'summary.json',result);print('A46 Summary:',json.dumps(sensor.plain(result),ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True)
    if not complete:raise SystemExit('A46 incomplete; all attempts retained')
    if args.archive:
        render(items)
        from unittest.mock import patch
        with patch.object(parent.parent,'BRANCH',BRANCH):bundle,commit=parent.parent.archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def render(items):
    import torch
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    torch.set_num_threads(2)
    for path,r in items:
        outputs=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        sheet=Image.new('RGB',(24+276*(1+len(rules.POLICIES)),382),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,5),r['case']['key']+' | NEW image, frozen t319, all predefined arms',fill='black',font=font(16))
        for j,name in enumerate(('GT',)+rules.POLICIES):
            x=12+276*j
            if name=='GT':
                im=Image.open(r['case']['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS);label='GT'
            else:
                ev=r['evaluation'][name];m=ev['metrics'];a=outputs[ev['output']][0].numpy().transpose(1,2,0)
                im=Image.fromarray(((np.clip(a,-1,1)+1)*127.5).astype(np.uint8));label=f'{name} x{r["gamma"][name]:g}'
                draw.text((x,62),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
            draw.text((x,34),label,fill='black',font=font(14));sheet.paste(im,(x,96))
        sheet.save(path.parent/'policy_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a45-dir',default=development.ROOT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--phase',choices=('choose','finish'))
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.freeze:
        root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p);freeze(root,p)
    elif args.shard is not None:
        if args.phase is None:parser.error('--phase required with --shard')
        worker(args)
    else:summarize(args)


if __name__=='__main__':main()
