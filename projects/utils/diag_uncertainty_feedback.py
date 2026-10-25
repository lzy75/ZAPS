"""A42 four-task one-image projected-uncertainty action pilot, <=232 new forwards."""
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
from utils import diag_ffhq_action_geometry as old
from utils import uncertainty_feedback as rules

source,P,scan=old.source,old.P,old.scan
ROOT='/data0/lzy/ZAPS-results/diag_uncertainty_feedback/A42_20261025'
CARD=PROJECTS/'experiment_cards/A42_uncertainty_feedback_pilot_20261025.md'
BRANCH='results/ffhq-uncertainty-feedback-20261025'


def write(path,value):source.WRITE(path,rules.plain(value))


def signed(path,value):
    write(path,value);Path(path).with_suffix('.sha256').write_text(source.digest(path)+'\n')


def prepare(args):
    root,parent=Path(args.root).resolve(),Path(args.a41_dir).resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):raise ValueError('Parent/output overlap')
    if root.exists():
        p=source.check_plan(root)
        if p['source_A41']!=str(parent):raise ValueError('Frozen source differs')
        verify(p);print('A42 frozen pilot reused; completed attempts retained.',flush=True);return
    previous=source.check_plan(parent);old.verify(previous)
    if previous['experiment']!='A41_ffhq_action_geometry':raise ValueError('Expected A41')
    image=min(c['image_id'] for c in previous['cases'])
    cases=[c for c in previous['cases'] if c['image_id']==image]
    if len(cases)!=4 or {c['task'] for c in cases}!=set(old.rules.TASKS):raise ValueError('Four matched source tasks required')
    refs={};identities={str(parent/'plan.json'):source.digest(parent/'plan.json'),
        str(parent/'motion_kernel.pt'):source.digest(parent/'motion_kernel.pt')}
    compatibility={}
    if (parent/'serialization_compat_v1.json').exists():
        compat=source.load_signed(parent/'serialization_compat_v1.json')
        if compat['plan_sha256']!=source.digest(parent/'plan.json'):raise ValueError('A41 compatibility plan differs')
        for path,sha in compat['files'].items():
            if source.digest(path)!=sha:raise ValueError('A41 compatibility source differs: '+path)
        compatibility=compat['files'];identities[str(parent/'serialization_compat_v1.json')]=source.digest(parent/'serialization_compat_v1.json')
    for c in cases:
        item=old.done(parent,previous,c)
        if item is None:raise ValueError('Source case incomplete: '+c['key'])
        path,r=item;refs[c['key']]=str(path);identities[str(path)]=source.digest(path)
        if 'serialization_compat_sha256' in r and r['serialization_compat_sha256']!=identities.get(str(parent/'serialization_compat_v1.json')):
            raise ValueError('A41 completed record compatibility differs')
        for n,sha in r['artifacts'].items():identities[str(path.parent/n)]=sha
        trainroot,trainplan=old.training_context(parent,previous,c)
        identities[str(trainroot/'plan.json')]=source.digest(trainroot/'plan.json')
        train=old.parent.done(trainroot,trainplan,c,'BASELINE')
        if train is None:raise ValueError('Missing paired C')
        identities[str(train[0])]=source.digest(train[0])
        for n,sha in train[1]['artifacts'].items():identities[str(train[0].parent/n)]=sha
        identities[r['measurement']['path']]=r['measurement']['sha256']
        identities[c['image']]=c['image_sha256']
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a42_20261025.sh',
         PROJECTS/'utils/test_uncertainty_feedback.py']
    p=dict(experiment='A42_uncertainty_feedback_pilot',source_A41=str(parent),source_records=refs,
        cases=cases,image_id=image,grid=previous['grid'],runtime=previous['runtime'],native_policy=previous['native_policy'],
        baseline_configs=previous['configs']['ffhq'],
        checkpoint=previous['checkpoints']['ffhq'],model_library_sources=previous['model_library_sources'],
        points=[list(v) for v in rules.POINTS],gammas=list(rules.GAMMAS),etas=list(rules.ETAS),
        identity_files=identities,worker_identity_files={**previous['worker_identity_files'],**compatibility,**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=0,planned_diagnostic_nfe_max=232,diagnostic_nfe_per_case_max=58,
        one_pulse_deployment_nfe_upper=314,
        scope='One reused image, four tasks, two separate single-pulse decisions. No GT policy fitting, '
              'step/grid changes, original-core edits, independent confirmation, free-NFE or online benefit claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);(root/'experiment_card.md').write_text(CARD.read_text())
    snapshot=root/'source_snapshot';snapshot.mkdir()
    shutil.copyfile(parent/'plan.json',snapshot/'A41_plan.json')
    for c in cases:
        dst=snapshot/(c['key']+'.json');shutil.copyfile(refs[c['key']],dst)
    print('A42 PREPARE PASS:image',image,'x four tasks x two states; new method0, diagnostic<=232. '
          'Parent completed data reused, no guaranteed final-quality gain.',flush=True)


def verify(p):
    for path,sha in p['identity_files'].items():
        if source.digest(path)!=sha:raise ValueError('Frozen input changed: '+path)


@contextmanager
def counted(model,path,r):
    start=time.perf_counter();r.update(status='running',calls=[],nfe=0)
    def save():r['seconds']=time.perf_counter()-start;write(path,r)
    def count(_m,inputs):
        r['calls'].append(int(inputs[1].flatten()[0]));r['nfe']+=1;save()
        if r['nfe']>58:raise RuntimeError('A42 per-case cap exceeded')
    save();hook=model.model.register_forward_pre_hook(count)
    try:yield
    except BaseException as e:r.update(status='failed',error=repr(e));save();raise
    finally:hook.remove()


def raw_field(z,x,t):
    import torch
    batch=torch.full((x.shape[0],),t,device=x.device,dtype=torch.long)
    eps=z.dm._predict_eps(x,batch);alpha=z.dm.alphas_cumprod[t]
    raw=(x-(1-alpha).sqrt()*eps.detach())/alpha.sqrt().clamp(min=1e-8)
    if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite raw denoiser')
    return raw,raw.clamp(-1,1)


def covariance_probes(evaluate,center,V,alpha):
    """IEEE caller. Central differences along unit columns; <=8 actual model calls."""
    import torch
    raw_matrices=[];derivatives=[];span_error=0.
    n=center.numel();base=max(float(center.double().square().mean().sqrt()),math.sqrt(1-float(alpha)))
    factor=(1-float(alpha))/math.sqrt(float(alpha))
    for eta in rules.ETAS:
        columns=[];vectors=[];h=eta*base*math.sqrt(n)
        for j in range(V.shape[1]):
            v=torch.from_numpy(V[:,j].copy()).reshape(center.shape).to(device=center.device,dtype=center.dtype)
            plus,minus=center+h*v,center-h*v
            error=abs(float(((plus-minus).double()*v.double()).sum())/(2*h)-1.)
            span_error=max(span_error,error)
            a,b=evaluate(plus),evaluate(minus)
            derivative=((a.double()-b.double())/(2*h)).cpu().numpy().reshape(-1)
            columns.append(factor*(V.T@derivative));vectors.append(derivative)
        raw_matrices.append(np.stack(columns,axis=1) if columns else np.empty((0,0)))
        derivatives.append(np.stack(vectors,axis=1) if vectors else np.empty((n,0)))
    return raw_matrices,derivatives,span_error


def done(root,p,case):
    found=[]
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path);ranks=[v['rank'] for v in r['points']]
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json') or len(r['points'])!=2 or
                r['calls']!=rules.calls(p['grid'],ranks) or r['nfe']!=len(r['calls']) or r['nfe']>58 or
                r['source_record_sha256']!=p['identity_files'][p['source_records'][case['key']]] or
                not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):
            raise ValueError('Invalid completed A42: '+str(path))
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple complete attempts; no output selection')
    return found[0] if found else None


def run_case(root,p,parent,previous,case,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    if done(root,p,case):return
    operator,mask=old.make_operator(parent,case,device);op_gate=old.parent.operator_gate(operator,device)
    trainroot,trainplan=old.training_context(parent,previous,case)
    _,state,original,y,meta,z,positions,noises=old.parent.setup_sampling(trainroot,trainplan,case,model,device,operator)
    source_item=old.done(parent,previous,case)
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir();path=attempt/'run.json'
    r=dict(case=case,plan_sha256=source.digest(root/'plan.json'),source_record_sha256=source.digest(source_item[0]),
        measurement=meta,points=[],gates={})
    external=rng_state(device);anchors={};cache={}
    def array(v):return v.detach().cpu().numpy()
    with counted(model,path,r),torch.no_grad(),model_rng_guard(model.model,device),\
            patch('torch.randn_like',side_effect=RuntimeError('Unexpected random draw')):
        x=state['init_noise'].to(device)
        for k,t in enumerate(positions.nominal):
            f=source.fields.field(z,y,x,t,positions)
            if k in {v[0] for v in rules.POINTS}:anchors[k]=(x.clone(),{n:f[n].clone() for n in ('clean','q')})
            x=old.parent.rules.risk.advance(z,x,f,positions,k,noises)
        parity=relative_error(x.cpu(),original)
        if not math.isfinite(parity) or parity>1e-6:raise RuntimeError('Original C replay parity failed')
        for k,t in rules.POINTS:
            s=positions.nominal[k+1];center,f=anchors[k];torch.cuda.synchronize(device);started=time.perf_counter()
            with P.precision.ieee(torch):
                cleans={};observations={};raw0=reference=None
                for g in rules.GAMMAS:
                    xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                    raw,clean=raw_field(z,xx,s);cleans[g]=array(clean);observations[g]=array(operator.H(clean))
                    if g==0.:reference=xx;raw0=raw
                repeated,_=raw_field(z,reference,s);repeat_error=relative_error(raw0,repeated)
                if not math.isfinite(repeat_error) or repeat_error>1e-7:raise RuntimeError('IEEE raw center repeat failed')
                raw_observation=operator.H(raw0);gradient=operator.transpose(y-raw_observation)
                V=rules.basis([cleans[2.]-cleans[0.],array(gradient)])
                matrices,derivatives,input_error=covariance_probes(lambda xx:raw_field(z,xx,s)[0],
                    reference,V,z.dm.alphas_cumprod[s])
                cov_gate,B=rules.covariance_gate(matrices,input_error)
                W=[]
                for j in range(V.shape[1]):
                    vector=torch.from_numpy(V[:,j].copy()).reshape(raw0.shape).to(device=device,dtype=raw0.dtype)
                    W.append(rules.active(array(.5*operator.base.H(vector)),array(mask) if mask is not None else None))
                W=np.stack(W,axis=1) if W else np.empty((len(rules.active(array(y),array(mask) if mask is not None else None)),0))
                decision=rules.choose(cleans,observations,array(y),array(mask) if mask is not None else None,
                    case['sigma'],array(raw0),V,B,W,cov_gate['usable'],array(raw_observation))
                source_point=next(v for v in source_item[1]['points'] if v['step']==k)
                expected=source_point['response']['loss']
                measurement_replay=max(abs(decision['local_measurement_loss'][g]-expected[str(g)])/
                    max(abs(expected[str(g)]),1e-12) for g in rules.GAMMAS)
                if measurement_replay>1e-6:raise RuntimeError('A41 fixed-state IEEE observation replay differs')
            torch.cuda.synchronize(device)
            r['points'].append(dict(step=k,t=t,next_t=s,rank=V.shape[1],covariance=cov_gate,decision=decision,
                sensor_nfe=6+4*V.shape[1],sensor_seconds=time.perf_counter()-started,
                reference_alpha_bar=float(z.dm.alphas_cumprod[s]),reference_input_RMS=float(reference.double().square().mean().sqrt()),
                IEEE_repeat_relative_error=repeat_error,A41_measurement_relative_error=measurement_replay,
                raw_center_clip_fraction=float((raw0.abs()>1).float().mean())))
            cache[str(k)]=dict(source_x=center.cpu(),source_clean=f['clean'].cpu(),source_q=f['q'].cpu(),
                reference=reference.cpu(),raw_mean=raw0.cpu(),basis=torch.from_numpy(V),
                candidate_clean={g:torch.from_numpy(v.copy()) for g,v in cleans.items()},
                matrices=[torch.from_numpy(v) for v in matrices],derivatives=[torch.from_numpy(v) for v in derivatives],
                B=torch.from_numpy(B),W=torch.from_numpy(W))
            print('A42 POINT:',case['task'],'t=',t,'covariance=',cov_gate['reason'],
                  'gamma=',decision['gamma'],'calls=',r['nfe'],flush=True)
        if (r['calls']!=rules.calls(p['grid'],[v['rank'] for v in r['points']]) or
                not same_rng(external,rng_state(device)) or P.precision.precision_state(torch)!=p['native_policy'] or
                any(v.grad is not None for v in model.model.parameters()) or
                not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('Cost/RNG/precision/parameters gate failed')
        old.parent.base.save_tensor(cache,attempt/'probes.pt');r['artifacts']={'probes.pt':source.digest(attempt/'probes.pt')}
        # Persist decisions before any FINAL quality labels are accessed.
        signed(attempt/'decisions_before_labels.json',dict(case=case,points=r['points'],plan_sha256=r['plan_sha256']))
        r['artifacts']['decisions_before_labels.json']=source.digest(attempt/'decisions_before_labels.json')
        labels=source_item[1]['metrics'];r['C_metrics']=labels['C']
        for point in r['points']:
            point['evaluation']={}
            for policy,g in point['decision']['gamma'].items():
                name='C' if g==1. else old.rules.arm(point['step'],g);metric=labels[name]
                point['evaluation'][policy]=dict(arm=name,metrics=metric,
                    delta={m:metric[m]-labels['C'][m] for m in old.rules.METRICS})
        r['gates']=dict(operator=op_gate,original_C=dict(passed=True,relative_error=parity),
            calls_rng_precision=dict(passed=True),parameters_fixed=dict(passed=True),decisions_before_FINAL_labels=dict(passed=True))
    r['status']='complete';signed(path,r);print('A42 COMPLETE:',case['task'],'diagnostic=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);verify(p)
    parent=Path(p['source_A41']);previous=source.check_plan(parent)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Native runtime changed')
        if old.parent.model_sources()!=p['model_library_sources']:raise ValueError('Model library changed')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoint']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Checkpoint/dtype changed')
            for c in cases:run_case(root,p,parent,previous,c,model,args.device);gc.collect();torch.cuda.empty_cache()


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);verify(p)
    items=[item for c in p['cases'] if (item:=done(root,p,c)) is not None]
    attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')];rows=[]
    for _,r in items:
        for point in r['points']:
            e=point['evaluation'];full=e['FULL']['delta'];relative={}
            for name in ('PHYSICAL','ISO','FIXED2'):
                relative[name]={m:full[m]-e[name]['delta'][m] for m in old.rules.METRICS}
            guard=full['ssim']>=-.002 and full['lpips']<=.005
            increment=(point['covariance']['usable'] and full['psnr']>=.05 and guard and
                all(relative[name]['psnr']>=.02 and relative[name]['ssim']>=-.002 and relative[name]['lpips']<=.005 for name in ('PHYSICAL','ISO')))
            rows.append(dict(task=r['case']['task'],t=point['t'],covariance=point['covariance'],gamma=point['decision']['gamma'],
                deltas={name:e[name]['delta'] for name in rules.POLICIES},FULL_minus=relative,
                FULL_quality_guard=guard,state_increment_candidate=bool(increment)))
    complete=len(items)==4;late=[r for r in rows if r['t']==319]
    positive=complete and all(r['covariance']['usable'] and r['deltas']['FULL']['psnr']>1e-6 and r['FULL_quality_guard'] for r in late)
    result=dict(experiment=p['experiment'],classification=('FOUR_TASK_LOCAL_POSITIVE_ACTION_CANDIDATE' if positive else
        'NO_FOUR_TASK_LOCAL_POSITIVE_ACTION_SUPPORT') if complete else 'INCOMPLETE_PILOT',
        completed_cases=len(items),planned_cases=4,image_clusters=1,method_nfe=0,
        diagnostic_nfe=sum(r['nfe'] for r in attempts),planned_diagnostic_nfe_max=232,
        partial_attempts=sum(r['status']!='complete' for r in attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in attempts),
        usable_states=sum(r['covariance']['usable'] for r in rows),total_states=len(rows),rows=rows,
        limits='One reused image; separate pulses, approximate rank<=2 covariance. Paid sensor <=14 extra calls '
               'per single decision, original scientific arm300. No guaranteed final gain, online loop, new-image '
               'transfer, true full covariance, step changes or NFE saving. Local positive actions are not state increment.')
    signed(root/'summary.json',result)
    print('A42 Summary:',json.dumps(rules.plain(result),ensure_ascii=False,allow_nan=False),flush=True);print('Records:',root,flush=True)
    if not complete:raise SystemExit('A42 incomplete; existing attempts retained; see shard logs')
    if args.archive:
        render(root,p,items);bundle,commit=archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def render(root,p,items):
    import torch
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    torch.set_num_threads(2)
    for path,r in items:
        src=Path(p['source_records'][r['case']['key']]);outputs=torch.load(src.parent/'outputs.pt',map_location='cpu',weights_only=True)
        names=('GT',)+rules.POLICIES;sheet=Image.new('RGB',(24+276*len(names),760),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),r['case']['task']+' | ONE reused image; rows are SEPARATE one-pulse experiments',fill='black',font=font(16))
        for row,point in enumerate(r['points']):
            top=42+352*row
            for j,name in enumerate(names):
                x=12+276*j
                if name=='GT':
                    im=Image.open(r['case']['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS);label='GT'
                else:
                    e=point['evaluation'][name];m=e['metrics'];tensor=outputs[e['arm']][0].cpu().numpy().transpose(1,2,0)
                    im=Image.fromarray(((np.clip(tensor,-1,1)+1)*127.5).astype(np.uint8))
                    label=f'{name} x{point["decision"]["gamma"][name]:g} t{point["t"]}'
                    draw.text((x,top+24),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
                draw.text((x,top),label,fill='black',font=font(15));sheet.paste(im,(x,top+60))
        sheet.save(path.parent/'policy_comparison.png')


def archive(root):
    import hashlib
    import tempfile
    from utils.archive_refinement_confirmation import inventory
    from utils.archive_state_indicator_stage import ALLOWED,commit_results
    files=inventory(root);fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    parent=PROJECTS/'server_records';parent.mkdir(exist_ok=True);bundle=parent/('uncertainty_feedback_'+fingerprint[:16])
    if not bundle.exists():
        with tempfile.TemporaryDirectory(prefix='.a42_export_',dir=parent) as temp:
            work=Path(temp)/'bundle';work.mkdir();(work/'.gitattributes').write_text('* -text\n')
            manifest=dict(source=str(root),fingerprint=fingerprint,files=[],excluded=[])
            for name,meta in files.items():
                item=dict(path=name,**meta)
                if Path(name).suffix.lower() not in ALLOWED|{'.sha256'}:manifest['excluded'].append(item);continue
                target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
                if source.digest(target)!=meta['sha256']:raise ValueError('Archive copy differs')
                manifest['files'].append(item)
            if inventory(root)!=files:raise ValueError('A42 changed during archive')
            write(work/'manifest.json',manifest);work.rename(bundle)
    return str(bundle),commit_results(bundle,BRANCH,message='Archive A42 uncertainty feedback mechanism pilot\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a41-dir',default=old.ROOT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
