"""A45 bounded development calibration: only one new midpoint objective, <=160 forwards."""
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
from utils import diag_frozen_state_matching as parent
from utils import balanced_state_calibration as rules

source,a42,old,P,scan=parent.source,parent.a42,parent.old,parent.P,parent.scan
sensor,control=parent.sensor,parent.control
ROOT='/data0/lzy/ZAPS-results/diag_balanced_state_calibration/A45_20261025'
CARD=PROJECTS/'experiment_cards/A45_balanced_state_calibration_20261025.md'
BRANCH='results/ffhq-balanced-state-20261025'
write,signed=parent.write,parent.signed


def prepare(args):
    root,previous=Path(args.root).resolve(),Path(args.a44_dir).resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):raise ValueError('Source/output overlap')
    if root.exists():
        p=source.check_plan(root);a42.verify(p)
        if p['source_A44']!=str(previous):raise ValueError('Frozen parent differs')
        print('A45 frozen finite calibration reused.',flush=True);return
    before=source.check_plan(previous);a42.verify(before);refs={};identities=dict(before['identity_files'])
    identities[str(previous/'plan.json')]=source.digest(previous/'plan.json')
    identities[str(previous/'allocation.json')]=source.digest(previous/'allocation.json')
    for c in before['cases']:
        refs[c['key']]={}
        for phase in ('choose','finish'):
            item=parent.phase_done(previous,before,c,phase)
            if item is None:raise ValueError('Need ALL20 completed A44 choose/finish cases')
            path,r=item;refs[c['key']][phase]=str(path);identities[str(path)]=source.digest(path)
            for n,sha in r['artifacts'].items():identities[str(path.parent/n)]=sha
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a45_20261025.sh',PROJECTS/'utils/test_balanced_state_calibration.py']
    p=dict(before);p.update(experiment='A45_balanced_state_calibration',source_A44=str(previous),a44_records=refs,
        identity_files=identities,worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=0,planned_diagnostic_nfe_max=160,diagnostic_nfe_per_case_max=8,lambda_grid=list(rules.LAMBDAS),
        limits='Finite DEVELOPMENT calibration on the same five old images; not independent confirmation or '
               'proof of per-image state necessity. Fixed t319, original grid. Original sensors are paid; full deployment<=317, fixed300.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for c in before['cases']:
        shutil.copyfile(refs[c['key']]['finish'],snapshot/(c['key']+'.json'))
    print('A45 PREPARE PASS: all20 A44 completed cases reused; only midpoint lambda=.5 is new; <=160 diagnostic, no training.',flush=True)


def done(root,p,case):
    found=[]
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        expected=[p['next_t']]*len(r['new_query_gammas'])+list(reversed(p['grid']))[25:]*int(r['new_suffix'])
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json') or r['calls']!=expected or
                r['nfe']!=len(expected) or r['nfe']+r['prior_recorded_nfe']>8 or
                not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):raise ValueError('Invalid A45: '+str(path))
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple completed attempts')
    return found[0] if found else None


@contextmanager
def counted(model,path,r):
    started=time.perf_counter();r.update(status='running',calls=[],nfe=0)
    def save():r['seconds']=time.perf_counter()-started;write(path,r)
    def hook(_m,inputs):
        if r['nfe']+r['prior_recorded_nfe']>=8:raise RuntimeError('A45 case budget exhausted')
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
    cache=torch.load(source_path.parent/'probes.pt',map_location='cpu',weights_only=True)
    operator,mask,opgate,state,_,y,z,positions,noises=parent.setup(p,case,model,device)
    folder=root/case['key'];folder.mkdir(exist_ok=True);prior=sum(source.read(f)['nfe'] for f in folder.glob('attempt_*/run.json'))
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir();path=attempt/'run.json'
    r=dict(case=case,plan_sha256=source.digest(root/'plan.json'),prior_recorded_nfe=prior,new_query_gammas=[],new_suffix=False,gates={})
    external=rng_state(device);arr=lambda x:x.detach().cpu().numpy();m=arr(mask) if mask is not None else None
    obs=sensor.active(arr(y),m);k,t,s=p['step'],p['t'],p['next_t']
    center=cache['source_x'].to(device);f={n:cache['source_'+n].to(device) for n in ('clean','q')}
    values={float(g):value for g,value in before['query_readouts'].items()};norm=None;target=None
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
                risk_error=max(abs(float(np.mean((cache['candidate_clean'][str(g)].numpy().reshape(-1).astype(float)-target)**2))-
                    values[g]['risks']['STATE'])/max(abs(values[g]['risks']['STATE']),1e-12) for g in sensor.GAMMAS)
                if risk_error>1e-6:raise RuntimeError('Cached STATE target differs')
                norm=rules.normalization(values)
                curve={g:rules.objective(values[g],norm) for g in sensor.GAMMAS}
                def evaluate(g):
                    if g not in values:
                        r['new_query_gammas'].append(g);xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                        try:
                            _,clean=a42.raw_field(z,xx,s);a=arr(clean).reshape(-1).astype(float)
                            values[g]=dict(loss=float(np.mean((obs-sensor.active(arr(operator.H(clean)),m))**2)/(2*case['sigma']**2)),
                                risks={'STATE':float(np.mean((a-target)**2))})
                        except RuntimeError as error:
                            if str(error)!='Nonfinite raw denoiser':raise
                            values[g]=dict(loss=None,risks={},error=str(error))
                    return dict(loss=values[g]['loss'],objective=rules.objective(values[g],norm))
                choice=control.continuous_choice(curve,before['budget']['cap'],evaluate,usable=before['budget']['q_resolved'])
            else:choice=dict(gamma=1.,reason='SOURCE_COVARIANCE_UNRESOLVED',attempts=[])
            geometry=rules.response_geometry(*(cache['candidate_clean'][str(g)].numpy() for g in (0.,1.,2.)))
        g=choice['gamma'];r.update(choice=choice,gamma=g,normalization=norm,response_geometry=geometry,
            source_state_readout=before['state_readout'],query_readouts={str(v):d for v,d in values.items()})
        if 'fit' in choice and choice['fit']['usable'] and g!=1.:
            coeff=choice['fit']['coefficients'];predicted=float(np.polyval(coeff,1.)-np.polyval(coeff,g))
            actual=-rules.objective(values[g],norm)
            r['model_agreement']=dict(predicted_decrease=predicted,actual_decrease=actual,
                actual_over_predicted=float(actual/predicted) if predicted>1e-12 else None)
        signed(attempt/'decision_before_labels.json',dict(case=case,gamma=g,choice=choice,normalization=norm,plan_sha256=r['plan_sha256']))
        # Old endpoint labels and all new final quality are used only after this decision is frozen.
        finished_path=Path(p['a44_records'][case['key']]['finish']);finished=source.load_signed(finished_path)
        aliases=[name for name,value in finished['gamma'].items() if value==g]
        if aliases:
            label=aliases[0];outputs=torch.load(finished_path.parent/'outputs.pt',map_location='cpu',weights_only=True)
            out=outputs[finished['evaluation'][label]['output']];evaluation=finished['evaluation'][label]
        elif g in sensor.GAMMAS:
            a41path=Path(p['source_records'][case['key']]);a41=source.load_signed(a41path)
            bank=torch.load(a41path.parent/'outputs.pt',map_location='cpu',weights_only=True);name='C' if g==1. else old.rules.arm(k,g)
            out=bank[name];metric=a41['metrics'][name];evaluation=dict(metrics=metric,
                delta={n:metric[n]-a41['metrics']['C'][n] for n in parent.rules.METRICS})
        else:
            r['new_suffix']=True;xx=old.parent.rules.risk.advance(z,center,f,positions,k,noises,g)
            for j in range(k+1,len(positions.nominal)):
                field=source.fields.field(z,y,xx,positions.nominal[j],positions)
                xx=old.parent.rules.risk.advance(z,xx,field,positions,j,noises)
            if not torch.isfinite(xx).all():raise RuntimeError('Nonfinite final output')
            out=xx.cpu();residual=(y-operator.H(xx)).double();gt=load_image_as_tensor(case['image']).cpu()
            with scan.preserved_rng():metric=dict(loss=float(residual.square().mean()/(2*case['sigma']**2)),**compute_all_metrics(out,gt,lpips_net='alex'))
            evaluation=dict(metrics=metric,delta={n:metric[n]-finished['evaluation']['C']['metrics'][n] for n in parent.rules.METRICS})
        r['evaluation']=dict(finished['evaluation'],MIX=evaluation)
        res=obs-sensor.active(arr(operator.H(out.to(device))),m)
        r['evaluation']['MIX']=dict(evaluation,output='MIX',informative_loss=float(np.mean(res**2)/(2*case['sigma']**2)))
        parent.invariant(model,z,state,device,external,p)
        if r['calls']!=[s]*len(r['new_query_gammas'])+list(positions.nominal[25:])*int(r['new_suffix']):raise RuntimeError('A45 call trace differs')
        if any(not math.isfinite(v) for ev in r['evaluation'].values() for v in ev['metrics'].values()):raise RuntimeError('Nonfinite quality')
        old.parent.base.save_tensor({'MIX':out},attempt/'outputs.pt')
        r['artifacts']={n:source.digest(attempt/n) for n in ('outputs.pt','decision_before_labels.json')}
        r['gates']=dict(operator=opgate,anchor=dict(passed=True,relative_error=parity),
            risk_replay=dict(passed=True,relative_error=risk_error),calls_rng_parameters=dict(passed=True),decision_before_labels=dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A45 COMPLETE:',case['key'],'lambda=.5 gamma=',g,'dPSNR=',round(r['evaluation']['MIX']['delta']['psnr'],5),'NEW calls=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);a42.verify(p)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
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
    items=[item for c in p['cases'] if (item:=done(root,p,c)) is not None];groups=[]
    for task in parent.rules.REFERENCE:
        rows=[r for _,r in items if r['case']['task']==task]
        if len(rows)!=5:continue
        selected=rules.select_weight(rows);endpoints=rules.select_weight(rows,grid=(0.,1.));loio=[]
        for held in rows:
            chosen=rules.select_weight([r for r in rows if r['case']['image_id']!=held['case']['image_id']])['chosen_lambda']
            loio.append(dict(image_id=held['case']['image_id'],chosen_lambda=chosen,
                evaluation=held['evaluation'][rules.NAMES[chosen]] if chosen is not None else None))
        improvement=None
        if selected['chosen_lambda'] is not None and endpoints['chosen_lambda'] is not None:
            improvement=(selected['scores'][str(selected['chosen_lambda'])]['mean_delta_vs_C']['psnr']-
                endpoints['scores'][str(endpoints['chosen_lambda'])]['mean_delta_vs_C']['psnr'])
        groups.append(dict(task=task,development=selected,already_observed_endpoint_calibration=endpoints,
            midpoint_selected=selected['chosen_lambda']==.5,new_calibration_PSNR_over_best_feasible_endpoint=improvement,LOIO_development_only=loio,
            MIX_mean_delta_vs_C={n:float(np.mean([r['evaluation']['MIX']['delta'][n] for r in rows])) for n in parent.rules.METRICS},
            MIX_gammas={r['case']['image_id']:r['gamma'] for r in rows}))
    attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')];complete=len(items)==20
    feasible=complete and all(g['development']['chosen_lambda'] is not None for g in groups)
    result=dict(experiment=p['experiment'],classification=('FOUR_TASK_DEVELOPMENT_CALIBRATION_CANDIDATE_ONLY' if feasible else
        'NO_FOUR_TASK_FINITE_CALIBRATION_SUPPORT') if complete else 'INCOMPLETE_PILOT',
        completed_cases=len(items),planned_cases=20,image_clusters=5,method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in attempts),
        planned_diagnostic_nfe_max=160,partial_attempts=sum(r['status']!='complete' for r in attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in attempts),groups=groups,limits=p['limits'])
    signed(root/'summary.json',result);print('A45 Summary:',json.dumps(sensor.plain(result),ensure_ascii=False,allow_nan=False),flush=True);print('Records:',root,flush=True)
    if not complete:raise SystemExit('A45 incomplete; attempts retained')
    if args.archive:
        from unittest.mock import patch
        with patch.object(parent.parent,'BRANCH',BRANCH):bundle,commit=parent.parent.archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--a44-dir',default=parent.ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
