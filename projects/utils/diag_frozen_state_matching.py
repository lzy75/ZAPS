"""A44 frozen continuous PROJECTED feedback: five other images, four tasks, state reassignment."""
import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_continuous_uncertainty_feedback as parent
from utils import frozen_state_matching as rules

a42,old,source,P,scan=parent.parent,parent.old,parent.source,parent.P,parent.scan
sensor,control=a42.rules,parent.rules
ROOT='/data0/lzy/ZAPS-results/diag_frozen_state_matching/A44_20261025'
CARD=PROJECTS/'experiment_cards/A44_frozen_state_matching_20261025.md'
BRANCH='results/ffhq-state-matching-20261025'
write,signed=parent.write,parent.signed
CAP={'choose':53,'finish':30}


def prepare(args):
    root,previous=Path(args.root).resolve(),Path(args.a43_dir).resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):raise ValueError('Source/output overlap')
    if root.exists():
        p=source.check_plan(root);a42.verify(p)
        if p['source_A43']!=str(previous):raise ValueError('Frozen parent differs')
        print('A44 frozen plan reused; completed phases retained.',flush=True);return
    before=source.check_plan(previous);a42.verify(before)
    a41=Path(before['source_A41']);p41=source.check_plan(a41);old.verify(p41)
    identities=dict(before['identity_files']);identities[str(previous/'plan.json')]=source.digest(previous/'plan.json')
    fixed={}
    for c in before['cases']:
        item=parent.done(previous,before,c)
        if item is None:raise ValueError('A43 incomplete; no new model will start')
        path,r=item;identities[str(path)]=source.digest(path)
        for n,sha in r['artifacts'].items():identities[str(path.parent/n)]=sha
        fixed[c['task']]=next(v for v in r['points'] if v['step']==24)['gamma']['PROJECTED']
    if set(fixed)!=set(rules.REFERENCE) or any(abs(fixed[t]-rules.REFERENCE[t])>1e-9 for t in fixed):
        raise ValueError('A43 PROJECTED reference differs')
    cases=[c for c in p41['cases'] if c['image_id'] in rules.IMAGES];refs={}
    if len(cases)!=20:raise ValueError('Need all five other A41 images x four tasks')
    identities[str(a41/'plan.json')]=source.digest(a41/'plan.json')
    identities[str(a41/'motion_kernel.pt')]=source.digest(a41/'motion_kernel.pt')
    for c in cases:
        if c['sigma']!=.05 or c['sampling_seed']!=1000:raise ValueError('Source protocol differs')
        item=old.done(a41,p41,c)
        if item is None:raise ValueError('A41 source incomplete: '+c['key'])
        path,r=item;refs[c['key']]=str(path);identities[str(path)]=source.digest(path)
        for n,sha in r['artifacts'].items():identities[str(path.parent/n)]=sha
        trainroot,trainplan=old.training_context(a41,p41,c)
        trained=old.parent.done(trainroot,trainplan,c,'BASELINE')
        if trained is None:raise ValueError('Source baseline incomplete')
        identities[str(trainroot/'plan.json')]=source.digest(trainroot/'plan.json')
        identities[str(trained[0])]=source.digest(trained[0])
        for n,sha in trained[1]['artifacts'].items():identities[str(trained[0].parent/n)]=sha
        identities[r['measurement']['path']]=r['measurement']['sha256'];identities[c['image']]=c['image_sha256']
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a44_20261025.sh',
         PROJECTS/'utils/test_frozen_state_matching.py']
    p={k:before[k] for k in ('runtime','native_policy','checkpoint','model_library_sources','grid')}
    p.update(experiment='A44_frozen_state_matching',source_A43=str(previous),source_A41=str(a41),source_records=refs,
        cases=cases,fixed_reference=fixed,step=24,t=319,next_t=218,phase_caps=CAP,
        identity_files=identities,worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=0,planned_diagnostic_nfe_max=1660,
        limits='Five reused development image clusters held out of A42/A43, not new blind confirmation. '
               'Frozen t319 single pulse only; no state-based timing, full online loop, NFE saving or equal-cost claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    shutil.copyfile(a41/'plan.json',snapshot/'A41_plan.json');shutil.copyfile(previous/'plan.json',snapshot/'A43_plan.json')
    for key,path in refs.items():shutil.copyfile(path,snapshot/('A41_'+key+'.json'))
    for c in before['cases']:
        item=parent.done(previous,before,c);shutil.copyfile(item[0],snapshot/('A43_'+c['key']+'.json'))
    print('A44 PREPARE PASS: five other reused images x four tasks; NEW training0, diagnostic<=1660; '
          'frozen rules, strong task-fixed and cross-image reassignment controls.',flush=True)


def phase_done(root,p,case,phase):
    found=[];times=list(reversed(p['grid']));s=times[25]
    for path in sorted((root/case['key']).glob(phase+'_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        expected=(times+[s]*(6+4*r['rank']+len(r['new_query_gammas'])) if phase=='choose' else
                  [t for _ in r['new_suffix_gammas'] for t in times[25:]])
        if (r['case']!=case or r['phase']!=phase or r['plan_sha256']!=source.digest(root/'plan.json') or
                r['source_record_sha256']!=p['identity_files'][p['source_records'][case['key']]] or
                r['calls']!=expected or r['nfe']!=len(expected) or r['nfe']+r['prior_recorded_nfe']>CAP[phase] or
                not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):
            raise ValueError('Invalid A44 phase: '+str(path))
        if phase=='finish' and r['allocation_sha256']!=source.digest(root/'allocation.json'):
            raise ValueError('Final phase decision binding differs')
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple completed phases; no output selection')
    return found[0] if found else None


def attempt(root,p,case,phase):
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    prior=sum(source.read(f)['nfe'] for f in folder.glob(phase+'_*/run.json'))
    out=folder/f'{phase}_{len(list(folder.glob(phase+"_*"))):03d}';out.mkdir()
    r=dict(case=case,phase=phase,prior_recorded_nfe=prior,plan_sha256=source.digest(root/'plan.json'),
        source_record_sha256=p['identity_files'][p['source_records'][case['key']]],gates={})
    return out,r


@contextmanager
def counted(model,path,r):
    started=time.perf_counter();r.update(status='running',calls=[],nfe=0)
    def save():r['seconds']=time.perf_counter()-started;write(path,r)
    def count(_m,inputs):
        if r['nfe']+r['prior_recorded_nfe']>=CAP[r['phase']]:raise RuntimeError('A44 phase budget exhausted')
        r['calls'].append(int(inputs[1].flatten()[0]));r['nfe']+=1;save()
    save();hook=model.model.register_forward_pre_hook(count)
    try:yield
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def setup(p,case,model,device):
    a41=Path(p['source_A41']);p41=source.check_plan(a41)
    operator,mask=old.make_operator(a41,case,device);gate=old.parent.operator_gate(operator,device)
    trainroot,trainplan=old.training_context(a41,p41,case)
    _,state,original,y,meta,z,positions,noises=old.parent.setup_sampling(trainroot,trainplan,case,model,device,operator)
    if meta!=source.load_signed(Path(p['source_records'][case['key']]))['measurement']:raise ValueError('Observation differs')
    return operator,mask,gate,state,original,y,z,positions,noises


def invariant(model,z,state,device,external,p):
    import torch
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    if (not same_rng(external,rng_state(device)) or P.precision.precision_state(torch)!=p['native_policy'] or
            any(v.grad is not None for v in model.model.parameters()) or
            not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
        raise RuntimeError('A44 RNG/precision/parameter gate failed')


def choose(root,p,case,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    operator,mask,opgate,state,original,y,z,positions,noises=setup(p,case,model,device)
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
            before=source.load_signed(Path(p['source_records'][case['key']]))
            expected=next(v for v in before['points'] if v['step']==k)['response']['loss']
            loss_error=max(abs(values[g]['loss']-expected[str(g)])/max(abs(expected[str(g)]),1e-12) for g in sensor.GAMMAS)
            if loss_error>1e-6:raise RuntimeError('A41 IEEE loss replay differs')
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
            raw_repeat=dict(passed=True,relative_error=repeat_error),A41_readouts=dict(passed=True,relative_error=loss_error),
            calls_rng_parameters=dict(passed=True),no_FINAL_labels_used=dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A44 CHOOSE:',case['key'],'gamma=',r['gamma'],'new calls=',r['nfe'],flush=True)


def freeze(root,p):
    items=[phase_done(root,p,c,'choose') for c in p['cases']]
    if not all(items):raise ValueError('Need ALL20 decisions before any final evaluation')
    choices={r['case']['key']:r['gamma'] for _,r in items}
    value=dict(plan_sha256=source.digest(root/'plan.json'),choices={str(path):source.digest(path) for path,_ in items},
        assignments=rules.allocation(p['cases'],choices,p['fixed_reference']),GT_used=False)
    file=root/'allocation.json'
    if file.exists():
        if source.load_signed(file)!=value:raise ValueError('Frozen allocation differs')
    else:signed(file,value)
    print('A44 FREEZE PASS: all20 own decisions and cyclic reassignment frozen; no final quality used.',flush=True)


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
    operator,mask,opgate,state,_,y,z,positions,noises=setup(p,case,model,device)
    folder,r=attempt(root,p,case,'finish');path=folder/'run.json';external=rng_state(device)
    assigned=allocation['assignments'][case['key']];r.update(assigned,allocation_sha256=source.digest(root/'allocation.json'),
        new_suffix_gammas=[],evaluation={})
    center=cache['source_x'].to(device);f={n:cache['source_'+n].to(device) for n in ('clean','q')};k=p['step']
    before=source.load_signed(Path(p['source_records'][case['key']]))
    bank=torch.load(Path(p['source_records'][case['key']]).parent/'outputs.pt',map_location='cpu',weights_only=True)
    outputs={};metrics={};arr=lambda x:x.detach().cpu().numpy();m=arr(mask) if mask is not None else None
    obs=sensor.active(arr(y),m);gt=None
    with counted(model,path,r),torch.no_grad(),model_rng_guard(model.model,device),\
            patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
        reference=old.parent.rules.risk.advance(z,center,f,positions,k,noises,0.)
        parity=relative_error(reference.cpu(),cache['reference'])
        if not math.isfinite(parity) or parity>1e-7:raise RuntimeError('Choose/finish anchor differs')
        for name in rules.POLICIES:
            g=r['gamma'][name];key=f'G{g:g}'
            if key not in outputs:
                if g in sensor.GAMMAS:
                    source_arm='C' if g==1. else old.rules.arm(k,g);outputs[key]=bank[source_arm];metrics[key]=before['metrics'][source_arm]
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
                delta={n:metric[n]-before['metrics']['C'][n] for n in rules.METRICS})
        invariant(model,z,state,device,external,p)
        if r['calls']!=[t for _ in r['new_suffix_gammas'] for t in positions.nominal[25:]]:raise RuntimeError('Finish trace differs')
        if any(not math.isfinite(v) for ev in r['evaluation'].values() for v in ev['metrics'].values()):raise RuntimeError('Nonfinite quality')
        old.parent.base.save_tensor(outputs,folder/'outputs.pt');r['artifacts']={'outputs.pt':source.digest(folder/'outputs.pt')}
        r['gates']=dict(operator=opgate,anchor=dict(passed=True,relative_error=parity),calls_rng_parameters=dict(passed=True),
            all20_decisions_frozen_before_evaluation=dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A44 FINISH:',case['key'],'dPSNR=',{n:round(r['evaluation'][n]['delta']['psnr'],4) for n in rules.POLICIES},
          'new calls=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(phase_done(root,p,c,args.phase) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Native runtime differs')
        if old.parent.model_sources()!=p['model_library_sources']:raise ValueError('Model library differs')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoint']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Checkpoint/dtype differs')
            for c in cases:
                if not phase_done(root,p,c,args.phase):(choose if args.phase=='choose' else finish)(root,p,c,model,args.device)
                gc.collect();torch.cuda.empty_cache()


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    items=[item for c in p['cases'] if (item:=phase_done(root,p,c,'finish')) is not None]
    attempts=[source.read(f) for f in root.glob('*/*_*/run.json')];groups=rules.task_summary([r for _,r in items])
    complete=len(items)==20;passed=sum(v['state_matching_candidate'] for v in groups)
    result=dict(experiment=p['experiment'],classification=('FOUR_TASK_STATE_MATCHING_CANDIDATE' if passed==4 else
        'TASK_SPECIFIC_STATE_MATCHING_SIGNAL' if passed else 'NO_FROZEN_STATE_MATCHING_SUPPORT') if complete else 'INCOMPLETE_PILOT',
        completed_cases=len(items),planned_cases=20,image_clusters=5,method_nfe=0,diagnostic_nfe=sum(v['nfe'] for v in attempts),
        planned_diagnostic_nfe_max=1660,partial_attempts=sum(v['status']!='complete' for v in attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(v['status']=='running' for v in attempts),
        tasks_with_matching_candidate=passed,groups=groups,fixed_reference=p['fixed_reference'],limits=p['limits']+
        ' Paid full sensor deployment<=317 vs fixed300; no blind generalization or statistical significance claim.')
    signed(root/'summary.json',result);print('A44 Summary:',json.dumps(sensor.plain(result),ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True)
    if not complete:raise SystemExit('A44 incomplete; failed/partial attempts retained')
    if args.archive:
        render(items)
        from unittest.mock import patch
        with patch.object(parent,'BRANCH',BRANCH):bundle,commit=parent.archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def render(items):
    import torch
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    torch.set_num_threads(2)
    for path,r in items:
        out=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        sheet=Image.new('RGB',(24+276*(1+len(rules.POLICIES)),382),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,5),r['case']['key']+' | frozen t319 one-pulse; all predefined arms',fill='black',font=font(16))
        for j,name in enumerate(('GT',)+rules.POLICIES):
            x=12+276*j
            if name=='GT':
                im=Image.open(r['case']['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS);label='GT'
            else:
                ev=r['evaluation'][name];m=ev['metrics'];a=out[ev['output']][0].numpy().transpose(1,2,0)
                im=Image.fromarray(((np.clip(a,-1,1)+1)*127.5).astype(np.uint8));label=f'{name} x{r["gamma"][name]:g}'
                draw.text((x,62),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
            draw.text((x,34),label,fill='black',font=font(14));sheet.paste(im,(x,96))
        sheet.save(path.parent/'policy_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a43-dir',default=parent.ROOT);parser.add_argument('--prepare',action='store_true')
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
