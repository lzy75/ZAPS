"""A48: identical local probes, with/without model-agreement feedback; old development only."""
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
from utils import diag_balanced_state_calibration as development
from utils import response_reliability as rules
from utils.frozen_four_task_confirmation import MATCHED

parent=development.parent
source,a42,old,P,scan=parent.source,parent.a42,parent.old,parent.P,parent.scan
sensor,control=parent.sensor,parent.control
write,signed=parent.write,parent.signed
ROOT='/data0/lzy/ZAPS-results/diag_response_reliability/A48_20261026'
CARD=PROJECTS/'experiment_cards/A48_response_reliability_20261026.md'
BRANCH='results/ffhq-response-reliability-20261026'


def prepare(args):
    root,previous=Path(args.root).resolve(),Path(args.a45_dir).resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):raise ValueError('Source/output overlap')
    if root.exists():
        p=source.check_plan(root);a42.verify(p)
        if p['source_A45']!=str(previous):raise ValueError('Frozen parent differs')
        print('A48 frozen plan reused; completed cases retained.',flush=True);return
    before=source.check_plan(previous);a42.verify(before)
    summary=source.load_signed(previous/'summary.json');refs={};identities=dict(before['identity_files'])
    for task,weight in rules.WEIGHTS.items():
        published=next(g for g in summary['groups'] if g['task']==task)['development']['chosen_lambda']
        if published!=weight:raise ValueError('A45 weight lock differs')
    for case in before['cases']:
        item=development.done(previous,before,case)
        if item is None:raise ValueError('Need all20 completed A45 cases')
        path,r=item;refs[case['key']]=str(path);identities[str(path)]=source.digest(path)
        for name,sha in r['artifacts'].items():identities[str(path.parent/name)]=sha
    if len(refs)!=20 or {c['image_id'] for c in before['cases']}!={'00446','00559','00608','00873','00951'}:
        raise ValueError('Expected five OLD development images only')
    for file in (previous/'plan.json',previous/'summary.json'):identities[str(file)]=source.digest(file)
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'utils/test_response_reliability.py',
        PROJECTS/'run_a48_20261026.sh',PROJECTS/'utils/frozen_four_task_confirmation.py']
    p=dict(before);p.update(experiment='A48_response_reliability',source_A45=str(previous),a45_records=refs,
        identity_files=identities,worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        weights_frozen=rules.WEIGHTS,matched_fixed=MATCHED,rho_min=rules.RHO_MIN,fractions=list(rules.FRACTIONS),
        case_cap=rules.CASE_CAP,planned_method_nfe=0,planned_diagnostic_nfe_max=400,
        limits='Five REUSED development image clusters, original sigma=.05, fixed t319. Post-A46-inspired engineering gate. '
        'Same probes LOCAL/TRUST; final labels excluded from decisions. No independent confirmation, full trajectory curvature, '
        'state-based timing, NFE saving or guarantee of better quality. Deployment conservative<=322 versus fixed300.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent);root.mkdir()
    signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snapshot=root/'source_snapshot';snapshot.mkdir();shutil.copyfile(previous/'summary.json',snapshot/'A45_summary.json')
    print('A48 PREPARE PASS: five old development images x four tasks; identical LOCAL/TRUST probes; '
        'rho>=.75; NEW training0 + diagnostic<=400 calls. A47 not started.',flush=True)


def done(root,p,case):
    found=[];times=list(reversed(p['grid']))[25:]
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        expected=[p['next_t']]*len(r['probe_gammas'])+times*len(r['new_suffix_gammas'])
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json') or r['calls']!=expected or
                r['nfe']!=len(expected) or r['nfe']+r['prior_recorded_nfe']>rules.CASE_CAP or
                not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):
            raise ValueError('Invalid A48 case: '+str(path))
        decision=source.load_signed(path.parent/'decision_before_labels.json')
        if decision['gamma']!=r['gamma'] or decision['plan_sha256']!=r['plan_sha256']:raise ValueError('Decision binding differs')
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple completed attempts; no label selection')
    return found[0] if found else None


@contextmanager
def counted(model,path,r):
    started=time.perf_counter();r.update(status='running',calls=[],nfe=0)
    def save():r['seconds']=time.perf_counter()-started;write(path,r)
    def hook(_m,inputs):
        if r['nfe']+r['prior_recorded_nfe']>=rules.CASE_CAP:raise RuntimeError('A48 cumulative case budget exhausted')
        r['calls'].append(int(inputs[1].flatten()[0]));r['nfe']+=1;save()
    save();handle=model.model.register_forward_pre_hook(hook)
    try:yield
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:handle.remove()


def run_case(root,p,case,model,device):
    import torch
    from unittest.mock import patch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    source_path=Path(p['a44_records'][case['key']]['choose']);before=source.load_signed(source_path)
    prior45path=Path(p['a45_records'][case['key']]);prior45=source.load_signed(prior45path)
    cache=torch.load(source_path.parent/'probes.pt',map_location='cpu',weights_only=True)
    operator,mask,opgate,state,_,y,z,positions,noises=parent.setup(p,case,model,device)
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    prior=sum(source.read(f)['nfe'] for f in folder.glob('attempt_*/run.json'))
    out=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';out.mkdir();path=out/'run.json'
    r=dict(case=case,plan_sha256=source.digest(root/'plan.json'),prior_recorded_nfe=prior,
        probe_gammas=[],new_suffix_gammas=[],gates={})
    external=rng_state(device);arr=lambda x:x.detach().cpu().numpy()
    m=arr(mask) if mask is not None else None;obs=sensor.active(arr(y),m);k,t,s=p['step'],p['t'],p['next_t']
    if positions.nominal[k]!=t or positions.nominal[k+1]!=s:raise ValueError('Frozen time differs')
    center=cache['source_x'].to(device);f={n:cache['source_'+n].to(device) for n in ('clean','q')}
    weight=p['weights_frozen'][case['task']]
    legacy=prior45['gamma'] if weight==.5 else before['choices']['PHYSICAL' if weight==0. else 'STATE']['gamma']
    values={float(g):v for g,v in prior45['query_readouts'].items()}
    clean={float(g):v.numpy() for g,v in cache['candidate_clean'].items()};target=None
    with counted(model,path,r),torch.no_grad(),model_rng_guard(model.model,device),\
            patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
        with P.precision.ieee(torch):
            reference=old.parent.rules.risk.advance(z,center,f,positions,k,noises,0.)
            parity=relative_error(reference.cpu(),cache['reference'])
            if not math.isfinite(parity) or parity>1e-7:raise RuntimeError('Cached anchor differs')
            risk_error=None
            if before['covariance']['usable']:
                V,B,W=(cache[n].numpy() for n in ('basis','B','W'))
                residual=obs-sensor.active(arr(operator.H(cache['raw_mean'].to(device))),m)
                target,_=sensor.gaussian_target(cache['raw_mean'].numpy(),V,B,W,residual,case['sigma'])
                errors=[abs(float(np.mean((clean[g].reshape(-1).astype(float)-target)**2))-values[g]['risks']['STATE'])/
                    max(abs(values[g]['risks']['STATE']),1e-12) for g in sensor.GAMMAS]
                risk_error=max(errors)
                if not math.isfinite(risk_error) or risk_error>1e-6:raise RuntimeError('Cached risk target differs')
            def probe(g):
                r['probe_gammas'].append(g)
                xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                try:
                    _,c=a42.raw_field(z,xx,s);a=arr(c);res=obs-sensor.active(arr(operator.H(c)),m)
                    value=dict(loss=float(np.mean(res**2)/(2*case['sigma']**2)),
                        risks={'STATE':float(np.mean((a.reshape(-1).astype(float)-target)**2))} if target is not None else {})
                    return a,value
                except RuntimeError as error:
                    if str(error)!='Nonfinite raw denoiser':raise
                    return None,dict(loss=None,risks={},error=str(error))
            repeated,_=probe(1.)
            repeat_error=relative_error(torch.from_numpy(repeated),cache['candidate_clean']['1.0']) if repeated is not None else math.inf
            if not math.isfinite(repeat_error) or repeat_error>1e-7:raise RuntimeError('Native/IEEE or source next-clean replay differs')
            if legacy!=1. and legacy not in clean:
                a,value=probe(legacy)
                if a is None:raise RuntimeError('Previously accepted LEGACY now nonfinite')
                oldvalue=values[legacy]
                err=abs(value['loss']-oldvalue['loss'])/max(abs(oldvalue['loss']),1e-12)
                if err>1e-6:raise RuntimeError('Previously accepted LEGACY readout differs')
                clean[legacy]=a;values[legacy]=value
            for g in rules.candidates(legacy):
                if g in values:continue
                a,value=probe(g);values[g]=value
                if a is not None:clean[g]=a
            usable=bool(legacy==1. or weight==0. or target is not None)
            decision=rules.choose(values,legacy,weight,usable)
            geometry=rules.geometry(rules.candidates(legacy),clean,float(f['q'].double().square().mean().sqrt()))
        r.update(gamma=dict(C=1.,LEGACY=float(legacy),MATCHED_FIXED=p['matched_fixed'][case['task']],**decision['gamma']),
            decision=decision,geometry=geometry,query_readouts={str(g):v for g,v in values.items()})
        signed(out/'decision_before_labels.json',dict(case=case,gamma=r['gamma'],decision=decision,geometry=geometry,
            plan_sha256=r['plan_sha256'],GT_used=False))
        # Final labels/old final tensors are accessed only after both policy choices are signed.
        finished_path=Path(p['a44_records'][case['key']]['finish']);finished=source.load_signed(finished_path)
        bank=torch.load(finished_path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        reuse={float(finished['gamma'][n]):(bank[e['output']],e['metrics']) for n,e in finished['evaluation'].items()}
        mixed=torch.load(prior45path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        reuse[float(prior45['gamma'])]=(mixed['MIX'],prior45['evaluation']['MIX']['metrics'])
        outputs={};metrics={};evaluation={};baseline=finished['evaluation']['C']['metrics'];gt=None
        for name in rules.POLICIES:
            g=r['gamma'][name];key=f'G{g:.3f}'
            if key not in outputs:
                if g in reuse:outputs[key],metrics[key]=reuse[g]
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
            evaluation[name]=dict(output=key,metrics=metric,informative_loss=float(np.mean(res**2)/(2*case['sigma']**2)),
                delta={n:metric[n]-baseline[n] for n in rules.METRICS})
        r['evaluation']=evaluation;parent.invariant(model,z,state,device,external,p)
        expected=[s]*len(r['probe_gammas'])+list(positions.nominal[25:])*len(r['new_suffix_gammas'])
        if r['calls']!=expected:raise RuntimeError('A48 call trace differs')
        if any(not math.isfinite(v) for ev in evaluation.values() for v in ev['metrics'].values()):raise RuntimeError('Nonfinite quality')
        old.parent.base.save_tensor(outputs,out/'outputs.pt')
        old.parent.base.save_tensor({str(g):torch.from_numpy(clean[g]) for g in rules.candidates(legacy) if g in clean},out/'probes.pt')
        r['artifacts']={n:source.digest(out/n) for n in ('outputs.pt','probes.pt','decision_before_labels.json')}
        r['gates']=dict(operator=opgate,anchor=dict(passed=True,relative_error=parity),
            repeated_next_clean=dict(passed=True,relative_error=repeat_error),risk_replay=dict(passed=True,relative_error=risk_error),
            calls_rng_parameters=dict(passed=True),decision_before_labels=dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A48 COMPLETE:',case['key'],'gamma=',r['gamma'],'TRUST-C dPSNR=',round(evaluation['TRUST']['delta']['psnr'],5),
        'new calls=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Runtime changed')
        if old.parent.model_sources()!=p['model_library_sources']:raise ValueError('Model library changed')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoint']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Checkpoint/dtype changed')
            for case in cases:
                if not done(root,p,case):run_case(root,p,case,model,args.device)
                gc.collect();torch.cuda.empty_cache()


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    items=[v for c in p['cases'] if (v:=done(root,p,c)) is not None]
    groups=rules.summarize([r for _,r in items]);complete=len(items)==20
    attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')]
    positive=sum(g['reliability_increment_development_candidate'] for g in groups)
    result=dict(experiment=p['experiment'],classification=('RELIABILITY_INCREMENT_IN_'+str(positive)+'_TASKS_DEVELOPMENT_ONLY') if complete else 'INCOMPLETE_PILOT',
        completed_cases=len(items),planned_cases=20,image_clusters=5,method_nfe=0,
        diagnostic_nfe=sum(r['nfe'] for r in attempts),planned_diagnostic_nfe_max=400,
        partial_attempts=sum(r['status']!='complete' for r in attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in attempts),
        quality_vs_C_tasks=sum(g['quality_vs_C'] for g in groups),groups=groups,limits=p['limits'])
    signed(root/'summary.json',result)
    compact={k:v for k,v in result.items() if k!='groups'}
    compact['groups']=[{k:v for k,v in g.items() if k!='per_image'} for g in groups]
    print('A48 Summary:',json.dumps(sensor.plain(compact),ensure_ascii=False,allow_nan=False),flush=True);print('Records:',root,flush=True)
    if not complete:raise SystemExit('A48 incomplete; attempts retained')
    if args.archive:
        from unittest.mock import patch
        render(items)
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
        draw.text((12,5),r['case']['key']+' | OLD development image, identical paid probes, fixed t319',fill='black',font=font(16))
        for j,name in enumerate(('GT',)+rules.POLICIES):
            x=12+276*j
            if name=='GT':
                im=Image.open(r['case']['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS);label='GT'
            else:
                ev=r['evaluation'][name];metric=ev['metrics'];a=outputs[ev['output']][0].numpy().transpose(1,2,0)
                im=Image.fromarray(((np.clip(a,-1,1)+1)*127.5).astype(np.uint8));label=f'{name} x{r["gamma"][name]:g}'
                draw.text((x,62),f'P {metric["psnr"]:.3f} S {metric["ssim"]:.4f} L {metric["lpips"]:.4f}',fill='black',font=font(12))
            draw.text((x,34),label,fill='black',font=font(14));sheet.paste(im,(x,96))
        sheet.save(path.parent/'policy_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a45-dir',default=development.ROOT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0');parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
