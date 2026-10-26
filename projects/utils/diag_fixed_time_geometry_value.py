"""A59 paid fixed-time field geometry versus fixed-grid final dose response."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
from statistics import mean
import sys

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_no_learning_replay as support
from utils import fixed_time_geometry_value as rules
from utils import angular_feedback_control as current
previous,parent,source,engine=support.previous,support.parent,support.source,support.engine
signed=previous.signed
ROOT='/data0/lzy/ZAPS-results/diag_fixed_time_geometry_value/A59_20261027'
CARD=PROJECTS/'experiment_cards/A59_fixed_time_geometry_value_20261027.md'
BRANCH='results/fixed-time-geometry-value-20261027'


def require(value,message):
    if not value:raise ValueError(message)


def check(root):
    p=source.check_plan(root);require(p['experiment']=='A59_fixed_time_geometry_value','Wrong plan')
    parent.a42.verify(p);return p


def prepare(args):
    root,prior=Path(args.root).resolve(),Path(args.a55r2_dir).resolve()
    require(not(root.is_relative_to(prior) or prior.is_relative_to(root)),'Source/output overlap')
    if root.exists():
        p=check(root);require(p['source_A55r2']==str(prior),'Frozen source differs')
        print('A59 frozen plan reused; complete work retained.',flush=True);return
    before=previous.check(prior);report=source.load_signed(prior/'summary.json')
    require(report['completed_cases']==report['planned_cases']==10,'Need all completed A55r2 cases')
    times=list(reversed(before['grid']));require(tuple(times[k] for k in rules.STEPS)==rules.TIMES,'Frozen source grid differs')
    identities=dict(before['identity_files'])
    for n in ('plan.json','summary.json','evaluation.json'):identities[str(prior/n)]=source.digest(prior/n)
    for c in before['cases']:require(previous.item(prior,before,c,'control') is not None,'Missing A55r2 control')
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'utils/test_fixed_time_geometry_value.py',
         PROJECTS/'run_a59_20261027.sh',PROJECTS/'utils/a59_session.py',Path(support.__file__)]
    p=dict(before);p.update(experiment='A59_fixed_time_geometry_value',source_A55r2=str(prior),phases=['control'],
        identity_files=identities,worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=0,planned_diagnostic_nfe=rules.budget()['total'],planned_backward_attempts_max=760,
        budget=rules.budget(),probe_steps=list(rules.STEPS),probe_scales=list(rules.SCALES),gammas=list(rules.GAMMAS),
        limits='Five OLD FFHQ images x SR/IP, one seed and fixed30 grid. Paid replay/probes/continuations. '
               'Same-t finite direction-field response, not stochastic trajectory curvature or a global Hessian bound. '
               'Image-held-out1NN diagnosis only; all images reused and final labels excluded until all outputs frozen. '
               'No online action, final quality gain from a controller, new-image or step/NFE-saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A59 PREPARE PASS:10 OLD cases,30 same-t centers;<=760 new diagnostic forwards/input VJPs; no online policy.',flush=True)


def item(root,p,c,phase):
    paths=sorted((root/c['key']).glob('control_*/run.json'))
    if not paths:return None
    require(len(paths)==1,'Multiple attempts; no selection')
    f=paths[0];r=source.read(f)
    if r['status']!='complete':return None
    r=source.load_signed(f)
    require(r['case']==c and r['plan_sha256']==source.digest(root/'plan.json') and
        r['calls']==r['expected_calls'] and r['nfe']==len(r['expected_calls'])<=76 and
        r['backward_attempts']==r['backward_completed']<=r['nfe'] and
        len(r['points'])==3 and all(g['passed'] for g in r['gates'].values()),'A59 control contract differs')
    require(all(source.digest(f.parent/n)==sha for n,sha in r['artifacts'].items()),'Artifacts changed')
    return f,r,'U_BASE'


def observed_field(z,y,x,t,s,positions,sigma,mask,backward):
    """Tap the exact source VJP output; never evaluate an extra hidden derivative."""
    from unittest.mock import patch
    actual=current.algebra.input_vjp;captured={}
    def tap(*args,**kwargs):
        value=actual(*args,**kwargs);captured['u']=value;return value
    with patch.object(current.algebra,'input_vjp',side_effect=tap):
        field,gamma,row=current.sensed_field(z,y,x,t,s,positions,sigma,1.,mask,backward)
    require(gamma==1.,'Source U gamma changed')
    return field,row,captured.get('u')


def sample(root,p,c,phase,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error
    if item(root,p,c,phase):return
    operator,mask,opgate,state,original,y,z,positions,noises=parent.setup(p,c,model,device)
    folder,r=engine.old.previous.new_attempt(root,p,c,phase);path=folder/'run.json'
    external=rng_state(device);r.update(points=[],expected_calls=[],traces=dict(REPLAY=[]),continuations={})
    centers={};outputs={}
    cpu=lambda x:x.detach().cpu()
    advance=parent.old.parent.rules.risk.advance
    with engine.old.previous.counted(model,path,r,76) as backward, current.base.input_vjp_parameters(model.model), \
         model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling draw')):
        with parent.P.precision.ieee(torch):
            x=state['init_noise'].to(device)
            for k,t in enumerate(positions.nominal):
                s=positions.nominal[k+1] if k<29 else -1;r['expected_calls'].append(t)
                field,row,u=observed_field(z,y,x,t,s,positions,c['sigma'],mask,backward)
                r['backward_completed']+=row['backward_calls'];r['traces']['REPLAY'].append(dict(step=k,t=t,readout=row))
                if k in rules.STEPS:
                    centers[k]=dict(x=cpu(x),field={n:cpu(field[n]) for n in ('clean','q')},u=cpu(u) if u is not None else None,readout=row)
                with torch.no_grad():x=advance(z,x,field,positions,k,noises)
                require(bool(torch.isfinite(x).all()),'Nonfinite U replay')
            outputs['U_BASE']=cpu(x)
            ref=Path(p['norm_sources'][c['key']]);bank=torch.load(ref.parent/'outputs.pt',map_location='cpu',weights_only=True)
            parity=relative_error(outputs['U_BASE'],bank['NORM_ONLY']);require(math.isfinite(parity) and parity<=1e-6,'U replay parity failed')
            for k in rules.STEPS:
                center=centers[k];t=positions.nominal[k];s=positions.nominal[k+1]
                point=dict(step=k,t=t,center_readout=center['readout'],geometry=dict(usable=False,reason='UNRESOLVED_CENTER'),probes=[])
                if center['readout'].get('resolved') and center['u'] is not None:
                    u0=center['u'].to(device);direction=u0/center['readout']['u_RMS']
                    magnitude=max(1.,float(center['x'].double().square().mean().sqrt()));pairs=[];complete=True
                    for scale in rules.SCALES:
                        epsilon=scale*magnitude;vectors=[]
                        for sign in (1.,-1.):
                            xx=center['x'].to(device)+sign*epsilon*direction;r['expected_calls'].append(t)
                            f,reading,uu=observed_field(z,y,xx,t,s,positions,c['sigma'],mask,backward)
                            r['backward_completed']+=reading['backward_calls'];point['probes'].append(dict(scale=scale,epsilon=epsilon,sign=sign,readout=reading))
                            vectors.append(cpu(uu).numpy() if uu is not None and reading.get('resolved') else None)
                        if any(v is None for v in vectors):complete=False
                        pairs.append((epsilon,*vectors))
                    if complete:
                        point['geometry']=rules.finite_geometry(center['u'].numpy(),pairs)
                    else:point['geometry']=dict(usable=False,reason='UNRESOLVED_PERTURBATION')
                r['points'].append(point)
                for gamma in rules.GAMMAS:
                    name=f'K{k}_G{gamma:g}';r['continuations'][name]=[]
                    field={n:v.to(device) for n,v in center['field'].items()}
                    with torch.no_grad():xx=advance(z,center['x'].to(device),field,positions,k,noises,gamma)
                    for j in range(k+1,30):
                        tj=positions.nominal[j];sj=positions.nominal[j+1] if j<29 else -1;r['expected_calls'].append(tj)
                        field,reading,_=observed_field(z,y,xx,tj,sj,positions,c['sigma'],mask,backward)
                        r['backward_completed']+=reading['backward_calls'];r['continuations'][name].append(dict(step=j,t=tj,readout=reading))
                        with torch.no_grad():xx=advance(z,xx,field,positions,j,noises)
                        require(bool(torch.isfinite(xx).all()),'Nonfinite continuation:'+name)
                    outputs[name]=cpu(xx)
                print('A59 point:',c['key'],'t=',t,'geometry usable=',point['geometry']['usable'],'F=',r['nfe'],flush=True)
        parent.invariant(model,z,state,device,external,p)
        require(r['calls']==r['expected_calls'] and r['nfe']<=76,'Unexpected paid call trace')
        parent.old.parent.base.save_tensor(outputs,folder/'outputs.pt');parent.old.parent.base.save_tensor(centers,folder/'centers.pt')
        signed(folder/'decisions_before_labels.json',dict(case=c,points=r['points'],GT_used=False,plan_sha256=r['plan_sha256']))
        r['artifacts']={n:source.digest(folder/n) for n in ('outputs.pt','centers.pt','decisions_before_labels.json')}
        r['gates']=dict(operator=opgate,calls_rng_parameters=dict(passed=True),U_replay=dict(passed=True,relative_error=parity))
    r['status']='complete';signed(path,r)


@contextmanager
def executor_bindings():
    from unittest.mock import patch
    with patch.object(engine,'check',check),patch.object(engine,'item',item),patch.object(engine,'sample',sample):yield


def summarize(root,p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    pairs=[(c,item(root,p,c,'control')) for c in p['cases']];require(all(v for c,v in pairs),'Need all10 outputs frozen before GT')
    labels=[];cases=[];values=[]
    with parent.scan.preserved_rng(),torch.no_grad():
        for c,(f,r,_) in pairs:
            outputs=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True);gt=load_image_as_tensor(c['image']).cpu()
            op,mask=parent.old.make_operator(Path(p['source_A41']),c,'cpu')
            a41=Path(p['source_A41']);p41=source.load_signed(a41/'plan.json');trainroot,trainplan=parent.old.training_context(a41,p41,c)
            _,train=parent.old.parent.done(trainroot,trainplan,c,'BASELINE')
            observation=torch.load(train['measurement']['path'],map_location='cpu',weights_only=True)
            y=observation['mean']+c['sigma']*observation['standard_noise'];metrics={}
            for n,x in outputs.items():
                m=compute_all_metrics(x,gt,lpips_net='alex');rr=current.base.informative_rms(y-op.H(x),mask)
                m['loss']=.5*(rr/c['sigma'])**2;metrics[n]=m
            for point in r['points']:
                k=point['step'];contrast={n:metrics[f'K{k}_G1.5'][n]-metrics[f'K{k}_G0.5'][n] for n in ('psnr','ssim','lpips','loss')}
                labels.append(dict(image_id=c['image_id'],task=c['task'],step=k,t=point['t'],center_readout=point['center_readout'],geometry=point['geometry'],contrast=contrast))
            cases.append(dict(case=c,metrics=metrics));values.append(r)
    analysis=rules.analyze(labels)
    groups=[]
    for task in current.TASKS:
        subset=[v for v in cases if v['case']['task']==task]
        groups.append(dict(task=task,mean_delta_vs_U={f'K{k}_G{g:g}':{m:mean(v['metrics'][f'K{k}_G{g:g}'][m]-v['metrics']['U_BASE'][m] for v in subset)
                      for m in ('psnr','ssim','lpips','loss')} for k in rules.STEPS for g in rules.GAMMAS}))
    result=dict(experiment=p['experiment'],classification=analysis['classification'],completed_cases=10,planned_cases=10,
        image_clusters=5,completed_states=30,method_nfe=0,diagnostic_nfe=sum(v['nfe'] for v in values),
        input_backward_attempts=sum(v['backward_attempts'] for v in values),planned_diagnostic_nfe=760,
        analysis=analysis,groups=groups,per_image=cases,limits=p['limits'])
    signed(root/'evaluation.json',dict(rows=labels,outputs_frozen_before_GT=True));signed(root/'summary.json',result)
    compact={n:v for n,v in result.items() if n not in ('per_image','analysis')}
    compact['analysis']={n:v for n,v in analysis.items() if n!='predictions'}
    print('A59 Summary:',json.dumps(compact,ensure_ascii=False,allow_nan=False),flush=True);print('Records:',root,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a55r2-dir',default=previous.ROOT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--phase',choices=('control',),default='control');parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args);return
    if args.shard is not None:
        with executor_bindings():engine.worker(args)
        return
    root=Path(args.root).resolve();p=check(root);summarize(root,p)
    if args.archive:
        from unittest.mock import patch
        with patch.object(engine.old.previous,'BRANCH',BRANCH):engine.old.previous.archive(root)

if __name__=='__main__':main()
