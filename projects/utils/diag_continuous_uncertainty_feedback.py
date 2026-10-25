"""A43 continuous, measured action verification and full-column covariance completion."""
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
from utils import diag_uncertainty_feedback as parent
from utils import continuous_uncertainty_feedback as rules

source,old,P,scan=parent.source,parent.old,parent.P,parent.scan
ROOT='/data0/lzy/ZAPS-results/diag_continuous_uncertainty/A43_20261025'
CARD=PROJECTS/'experiment_cards/A43_continuous_completed_uncertainty_20261025.md'
BRANCH='results/ffhq-continuous-uncertainty-20261025'
write,signed=parent.write,parent.signed


def prepare(args):
    root,previous=Path(args.root).resolve(),Path(args.a42_dir).resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):raise ValueError('Source/output overlap')
    if root.exists():
        p=source.check_plan(root);parent.verify(p)
        if p['source_A42']!=str(previous):raise ValueError('Frozen parent differs')
        print('A43 frozen plan reused; completed cases retained.',flush=True);return
    before=source.check_plan(previous);parent.verify(before)
    if before['experiment']!='A42_uncertainty_feedback_pilot' or source.digest(previous/'plan.json')!=\
            '73d68f849ba8bd835df26f5a95afeea55e3cfa061ead5219d12c2b9fea571e63':
        raise ValueError('Need the completed A42 source diagnosed in A42b')
    identities=dict(before['identity_files']);identities[str(previous/'plan.json')]=source.digest(previous/'plan.json')
    refs={}
    for c in before['cases']:
        item=parent.done(previous,before,c)
        if item is None:raise ValueError('A42 incomplete; no new model will start')
        path,r=item;refs[c['key']]=str(path);identities[str(path)]=source.digest(path)
        for n,sha in r['artifacts'].items():identities[str(path.parent/n)]=sha
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a43_20261025.sh',
         PROJECTS/'utils/test_continuous_uncertainty_feedback.py',PROJECTS/'utils/recover_a42_archive.py']
    p=dict(experiment='A43_continuous_completed_uncertainty',source_A42=str(previous),
        source_A41=before['source_A41'],source_records=refs,A41_records=before['source_records'],
        cases=before['cases'],image_id=before['image_id'],grid=before['grid'],runtime=before['runtime'],
        native_policy=before['native_policy'],checkpoint=before['checkpoint'],
        model_library_sources=before['model_library_sources'],identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=0,planned_diagnostic_nfe_max=516,diagnostic_nfe_per_case_max=129,
        single_pulse_sampling_nfe=300,conservative_deployment_nfe_upper=317,
        fit_relative_error_limit=rules.FIT_TOLERANCE,completion_closure_limit=.10,posterior_scale_spread_limit=.10,
        gamma_quantization=.001,verification_attempts_max=3,
        scope='One reused image, four tasks, two separate one-pulse trials. Matched continuous controls; '
              'not an online loop, independent confirmation, free sensor, equal-300-NFE or generalization claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A43 PREPARE PASS: one reused image x four tasks x two states; <=516 NEW diagnostic calls; '
          'continuous PHYSICAL/ISO/PROJECTED/NYSTROM + fixed4/8 and noise-budget controls.',flush=True)


def expected_trace(r,grid):
    times=list(reversed(grid));result=[]
    for job in r['query_jobs']:
        result.append(dict(kind='lookahead',step=job['step'],gamma=job['gamma'],t=times[job['step']+1]))
    for job in r['suffix_jobs']:
        result.extend(dict(kind='suffix',step=job['step'],gamma=job['gamma'],t=t) for t in times[job['step']+1:])
    return result


def done(root,p,case):
    found=[]
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json') or len(r['points'])!=2 or
                r['calls']!=expected_trace(r,p['grid']) or r['nfe']!=len(r['calls']) or
                r['nfe']+r['prior_recorded_nfe']>129 or
                r['source_record_sha256']!=p['identity_files'][p['source_records'][case['key']]] or
                not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):
            raise ValueError('Invalid A43 completed record: '+str(path))
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple completed attempts; refuse output selection')
    return found[0] if found else None


@contextmanager
def counted(model,path,r,context):
    start=time.perf_counter();r.update(status='running',calls=[],nfe=0)
    def save():r['seconds']=time.perf_counter()-start;write(path,r)
    def count(_m,inputs):
        if r['nfe']+r.get('prior_recorded_nfe',0)>=129:raise RuntimeError('A43 case budget exhausted before next forward')
        if set(context)!={'kind','step','gamma'}:raise RuntimeError('Unclassified model forward')
        r['calls'].append(dict(context,t=int(inputs[1].flatten()[0])));r['nfe']+=1;save()
    save();hook=model.model.register_forward_pre_hook(count)
    try:yield
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def targets(cache,point,operator,y,mask,sigma,device):
    """No UNet: reconstruct old targets exactly, then complete the same measured sketch."""
    import torch
    arr=lambda x:x.detach().cpu().numpy()
    V,B,W=(cache[n].numpy().astype(np.float64) for n in ('basis','B','W'))
    mean=cache['raw_mean'].numpy();m=mask.detach().cpu().numpy() if mask is not None else None
    raw_obs=arr(operator.H(cache['raw_mean'].to(device)))
    residual=parent.rules.active(arr(y),m)-parent.rules.active(raw_obs,m)
    target,state={},{};gate=dict(usable=False,reason='SOURCE_COVARIANCE_UNRESOLVED')
    if not point['covariance']['usable']:return target,state,gate
    for name,cov in (('ISO',np.eye(len(B))*float(np.trace(B))/len(B)),('PROJECTED',B)):
        target[name],state[name]=parent.rules.gaussian_target(mean,V,cov,W,residual,sigma)
        original=point['decision']['predicted_risk']['FULL' if name=='PROJECTED' else name]
        error=max(abs(float(np.mean((cache['candidate_clean'][g].numpy().reshape(-1).astype(float)-target[name])**2))-
            original[str(g)])/max(abs(original[str(g)]),1e-12) for g in parent.rules.GAMMAS)
        if error>1e-6:raise RuntimeError('Cached Gaussian target replay differs: '+name)
        state[name]['cached_risk_relative_error']=error
    alpha=point['reference_alpha_bar'];factor=(1-alpha)/math.sqrt(alpha);factors=[];diagnostics=[];shifts=[]
    for derivative,matrix in zip(cache['derivatives'],cache['matrices']):
        F,info=rules.nystrom_factor(V,factor*derivative.numpy(),matrix.numpy());WF=[]
        for j in range(F.shape[1]):
            v=torch.from_numpy(F[:,j].copy()).reshape(cache['raw_mean'].shape).to(device=device,dtype=torch.float32)
            WF.append(parent.rules.active(arr(.5*operator.base.H(v)),m))
        WF=np.stack(WF,axis=1)
        candidate,readout=parent.rules.gaussian_target(mean,F,np.eye(F.shape[1]),WF,residual,sigma)
        # With a nonorthogonal factor, the eigenvalues are those of F^T F, not I.
        readout.pop('projected_prior_eigenvalues');readout.update(info)
        factors.append(candidate);diagnostics.append(readout);shifts.append(candidate-mean.reshape(-1))
    spread=float(np.linalg.norm(shifts[0]-shifts[1]))/max(float(np.linalg.norm(v)) for v in shifts+[np.array([1e-20])])
    usable=all(v['column_closure_relative_error']<=.10 for v in diagnostics) and spread<=.10
    gate=dict(usable=bool(usable),reason='COMPLETED_SKETCH_USABLE' if usable else 'COMPLETION_METROLOGY_UNRESOLVED',
              scales=diagnostics,posterior_shift_relative_scale_spread=spread)
    state['NYSTROM']=diagnostics[0]
    if usable:target['NYSTROM']=factors[0]
    return target,state,gate


def run_case(root,p,case,model,device):
    import torch
    from unittest.mock import patch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    if done(root,p,case):return
    a41=Path(p['source_A41']);p41=source.check_plan(a41)
    operator,mask=old.make_operator(a41,case,device);op_gate=old.parent.operator_gate(operator,device)
    trainroot,trainplan=old.training_context(a41,p41,case)
    _,state,_,y,meta,z,positions,noises=old.parent.setup_sampling(trainroot,trainplan,case,model,device,operator)
    src=Path(p['source_records'][case['key']]);before=source.load_signed(src)
    if meta!=before['measurement']:raise ValueError('Source measurement identity differs')
    caches=torch.load(src.parent/'probes.pt',map_location='cpu',weights_only=True)
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir();path=attempt/'run.json'
    prior_nfe=sum(source.read(f)['nfe'] for f in folder.glob('attempt_*/run.json'))
    r=dict(case=case,plan_sha256=source.digest(root/'plan.json'),source_record_sha256=source.digest(src),prior_recorded_nfe=prior_nfe,
        measurement=meta,points=[],query_jobs=[],suffix_jobs=[],gates={})
    external=rng_state(device);context={};outputs={};probe_tensors={};anchors={}
    arr=lambda x:x.detach().cpu().numpy()
    m=arr(mask) if mask is not None else None;obs=parent.rules.active(arr(y),m)
    with counted(model,path,r,context),torch.no_grad(),model_rng_guard(model.model,device),\
            patch('torch.randn_like',side_effect=RuntimeError('Unexpected random draw')):
        for point in before['points']:
            k,t=point['step'],point['t'];cache=caches[str(k)];s=positions.nominal[k+1]
            if positions.nominal[k]!=t or float(z.dm.alphas_cumprod[s])!=point['reference_alpha_bar']:
                raise ValueError('Source time/noise coordinate differs')
            center=cache['source_x'].to(device);f={n:cache['source_'+n].to(device) for n in ('clean','q')}
            anchors[k]=(center,f)
            with P.precision.ieee(torch):
                reference=old.parent.rules.risk.advance(z,center,f,positions,k,noises,0.)
                error=relative_error(reference.cpu(),cache['reference'])
                if not math.isfinite(error) or error>1e-7:raise RuntimeError('Cached native anchor/noise differs')
                target,state_readout,completion=targets(cache,point,operator,y,mask,case['sigma'],device)
                values={};tensors={}
                def readout(clean):
                    loss=float(np.mean((obs-parent.rules.active(arr(operator.H(clean)),m))**2)/(2*case['sigma']**2))
                    a=arr(clean).astype(float).reshape(-1)
                    return dict(loss=loss,risks={name:float(np.mean((a-v)**2)) for name,v in target.items()})
                for g,clean in cache['candidate_clean'].items():
                    g=float(g);values[g]=readout(clean.to(device));tensors[g]=clean
                loss_error=max(abs(values[g]['loss']-point['decision']['local_measurement_loss'][str(g)])/
                    max(abs(point['decision']['local_measurement_loss'][str(g)]),1e-12) for g in parent.rules.GAMMAS)
                if loss_error>1e-6:raise RuntimeError('Cached measurement replay differs')
                q_rms=float(f['q'].double().square().mean().sqrt())
                budget=rules.noise_budget(float(z.dm.alphas_cumprod[t]),float(z.dm.alphas_cumprod[s]),q_rms)
                choices={};gammas=dict(C=1.,FIXED2=2.,FIXED4=4.,FIXED8=8.,NOISE_BUDGET=budget['quantized_cap'])
                def evaluate(g):
                    if g not in values:
                        job=dict(step=k,gamma=g);r['query_jobs'].append(job);context.clear();context.update(kind='lookahead',**job)
                        xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                        try:
                            _,clean=parent.raw_field(z,xx,s);values[g]=readout(clean);tensors[g]=clean.cpu()
                        except RuntimeError as error:
                            if str(error)!='Nonfinite raw denoiser':raise
                            values[g]=dict(loss=None,risks={},error=str(error))
                    return values[g]
                for name in rules.CONTROLLERS:
                    curve={g:values[g]['loss'] if name=='PHYSICAL' else values[g]['risks'].get(name,0.) for g in parent.rules.GAMMAS}
                    def objective(g,name=name):
                        value=evaluate(g)
                        return dict(loss=value['loss'],objective=value['loss'] if name=='PHYSICAL' else value['risks'].get(name))
                    choices[name]=rules.continuous_choice(curve,budget['cap'],objective,
                        usable=budget['q_resolved'] and (name=='PHYSICAL' or name in target))
                    gammas[name]=choices[name]['gamma']
                r['points'].append(dict(step=k,t=t,next_t=s,budget=budget,completion=completion,
                    state=state_readout,choices=choices,gamma=gammas,query_readouts={str(g):v for g,v in values.items()},
                    cached_anchor_relative_error=error,cached_loss_relative_error=loss_error))
                probe_tensors[str(k)]={str(g):v for g,v in tensors.items()}
            print('A43 CHOICES:',case['task'],'t=',t,'gamma=',gammas,'new calls=',r['nfe'],flush=True)
        # Freeze ALL choices for this case before reading any final quality labels.
        signed(attempt/'decisions_before_labels.json',dict(case=case,plan_sha256=r['plan_sha256'],points=r['points']))
        label_path=Path(p['A41_records'][case['key']]);labels=source.load_signed(label_path)
        original_outputs=torch.load(label_path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        suffix_cache={};metrics={};gt=None
        for point in r['points']:
            k=point['step'];center,f=anchors[k];point['evaluation']={}
            for name in rules.POLICIES:
                g=point['gamma'][name];key=f'K{k}_G{g:g}';outkey=(k,g)
                if outkey not in suffix_cache:
                    if g in parent.rules.GAMMAS:
                        bank='C' if g==1. else old.rules.arm(k,g)
                        suffix_cache[outkey]=original_outputs[bank];metrics[outkey]=labels['metrics'][bank]
                    else:
                        job=dict(step=k,gamma=g);r['suffix_jobs'].append(job);context.clear();context.update(kind='suffix',**job)
                        xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                        torch.cuda.synchronize(device);started=time.perf_counter()
                        for j in range(k+1,len(positions.nominal)):
                            field=source.fields.field(z,y,xx,positions.nominal[j],positions)
                            xx=old.parent.rules.risk.advance(z,xx,field,positions,j,noises)
                        torch.cuda.synchronize(device);suffix_cache[outkey]=xx.cpu()
                        if not torch.isfinite(xx).all():raise RuntimeError('Nonfinite final output')
                        if gt is None:gt=load_image_as_tensor(case['image']).cpu()
                        residual=(y-operator.H(xx)).double()
                        with scan.preserved_rng():
                            metrics[outkey]=dict(loss=float(residual.square().mean()/(2*case['sigma']**2)),
                                **compute_all_metrics(suffix_cache[outkey],gt,lpips_net='alex'))
                        job['seconds']=time.perf_counter()-started
                    outputs[key]=suffix_cache[outkey]
                metric=metrics[outkey];baseline=labels['metrics']['C']
                active_residual=obs-parent.rules.active(arr(operator.H(suffix_cache[outkey].to(device))),m)
                point['evaluation'][name]=dict(output=key,metrics=metric,
                    informative_loss=float(np.mean(active_residual**2)/(2*case['sigma']**2)),
                    delta={n:metric[n]-baseline[n] for n in old.rules.METRICS})
                print('A43 ARM:',case['task'],'t=',point['t'],name,'gamma=',g,
                      'dPSNR=',round(point['evaluation'][name]['delta']['psnr'],5),'calls=',r['nfe'],flush=True)
        if (r['calls']!=expected_trace(r,p['grid']) or not same_rng(external,rng_state(device)) or
                P.precision.precision_state(torch)!=p['native_policy'] or any(v.grad is not None for v in model.model.parameters()) or
                not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('Cost/RNG/precision/parameters gate failed')
        if any(not math.isfinite(v) for pnt in r['points'] for ev in pnt['evaluation'].values() for v in ev['metrics'].values()):
            raise RuntimeError('Nonfinite final metrics')
        old.parent.base.save_tensor(outputs,attempt/'outputs.pt');old.parent.base.save_tensor(probe_tensors,attempt/'queries.pt')
        r['artifacts']={n:source.digest(attempt/n) for n in ('outputs.pt','queries.pt','decisions_before_labels.json')}
        r['gates']=dict(operator=op_gate,cached_anchor_and_readouts=dict(passed=True),calls_rng_precision=dict(passed=True),
            parameters_fixed=dict(passed=True),decisions_before_FINAL_labels=dict(passed=True))
    r['status']='complete';signed(path,r);print('A43 COMPLETE:',case['task'],'NEW diagnostic=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);parent.verify(p)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(done(root,p,c) for c in cases):return
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
            for case in cases:run_case(root,p,case,model,args.device);gc.collect();torch.cuda.empty_cache()


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);parent.verify(p)
    items=[item for case in p['cases'] if (item:=done(root,p,case)) is not None]
    attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')];rows=[]
    for _,r in items:
        for point in r['points']:
            deltas={name:point['evaluation'][name]['delta'] for name in rules.POLICIES}
            full=deltas['NYSTROM'];rows.append(dict(task=r['case']['task'],t=point['t'],gamma=point['gamma'],
                budget=point['budget'],completion=point['completion'],deltas=deltas,
                NYSTROM_minus={name:{m:full[m]-deltas[name][m] for m in old.rules.METRICS}
                    for name in rules.POLICIES if name!='NYSTROM'},
                action_positive=bool(full['psnr']>=.05 and rules.quality_guard(full)),
                state_increment_candidate=bool(point['completion']['usable'] and rules.state_increment(deltas))))
    complete=len(items)==4;increments=sum(r['state_increment_candidate'] for r in rows);positives=sum(r['action_positive'] for r in rows)
    classification=('LOCAL_STATE_INCREMENT_CANDIDATE_ONLY' if increments else
        'ACTION_GAIN_ONLY_NO_STATE_INCREMENT' if positives else 'NO_LOCAL_STATE_GAIN_SUPPORT') if complete else 'INCOMPLETE_PILOT'
    result=dict(experiment=p['experiment'],classification=classification,completed_cases=len(items),planned_cases=4,image_clusters=1,
        method_nfe=0,diagnostic_nfe=sum(v['nfe'] for v in attempts),planned_diagnostic_nfe_max=516,
        partial_attempts=sum(v['status']!='complete' for v in attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(v['status']=='running' for v in attempts),
        action_positive_states=positives,state_increment_candidate_states=increments,rows=rows,
        limits=p['scope']+' Scientific sampling300; complete sensor deployment<=317 per pulse; '
            'fixed controls300, PHYSICAL can use a cheaper sensor. Cached old calls are not new or free online calls. '
            'Positive fixed/continuous actions alone are not covariance/state benefit. Original A42 gates retained.')
    signed(root/'summary.json',result);print('A43 Summary:',json.dumps(parent.rules.plain(result),ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True)
    if not complete:raise SystemExit('A43 incomplete; attempts retained, inspect shard logs')
    if args.archive:
        render(items);bundle,commit=archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def render(items):
    import torch
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    torch.set_num_threads(2)
    for path,r in items:
        outputs=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        names=('GT',)+rules.POLICIES;sheet=Image.new('RGB',(24+276*len(names),760),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),r['case']['task']+' | ONE reused image; rows are SEPARATE one-pulse experiments',fill='black',font=font(16))
        for row,point in enumerate(r['points']):
            top=42+352*row
            for j,name in enumerate(names):
                x=12+276*j
                if name=='GT':
                    im=Image.open(r['case']['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS);label='GT'
                else:
                    ev=point['evaluation'][name];metric=ev['metrics'];tensor=outputs[ev['output']][0].numpy().transpose(1,2,0)
                    im=Image.fromarray(((np.clip(tensor,-1,1)+1)*127.5).astype(np.uint8))
                    label=f'{name} x{point["gamma"][name]:g} t{point["t"]}'
                    draw.text((x,top+24),f'P {metric["psnr"]:.3f} S {metric["ssim"]:.4f} L {metric["lpips"]:.4f}',fill='black',font=font(12))
                draw.text((x,top),label,fill='black',font=font(14));sheet.paste(im,(x,top+60))
        sheet.save(path.parent/'policy_comparison.png')


def archive(root):
    from unittest.mock import patch
    from utils import archive_state_indicator_stage
    from utils.recover_a42_archive import commit_bytes,external_objects,ArchiveGitError,disk_full
    def publish(bundle,_branch,repo=PROJECTS.parent,message=None):
        try:return commit_bytes(bundle,BRANCH,repo,root.parent/'git_recovery_work')
        except ArchiveGitError as error:
            print(str(error),flush=True)
            if not disk_full(error):raise
            store,objects=external_objects(repo,root.parent/'git_objects');prior=os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES')
            alternatives=str(objects)+(os.pathsep+prior if prior else '')
            with patch.dict(os.environ,{'GIT_ALTERNATE_OBJECT_DIRECTORIES':alternatives}):
                return commit_bytes(bundle,BRANCH,repo,root.parent/'git_recovery_work',store)
    with patch.object(archive_state_indicator_stage,'commit_results',publish):return parent.archive(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a42-dir',default=parent.ROOT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
