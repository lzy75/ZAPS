"""A60 same-run current/lagged direction at original30 points and current physical length."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
from statistics import mean,median
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_no_learning_replay as support
from utils import direction_refresh_control as rules
from utils import diag_fixed_time_geometry_value as observer
previous,parent,source,engine=support.previous,support.parent,support.source,support.engine
signed=previous.signed
ROOT='/data0/lzy/ZAPS-results/diag_direction_refresh_value/A60_20261027'
CARD=PROJECTS/'experiment_cards/A60_direction_refresh_20261027.md'
BRANCH='results/direction-refresh-value-20261027'


def require(value,message):
    if not value:raise ValueError(message)


def check(root):
    p=source.check_plan(root);require(p['experiment']=='A60_direction_refresh','Wrong plan')
    parent.a42.verify(p);return p


def prepare(args):
    root,prior=Path(args.root).resolve(),Path(args.a55r2_dir).resolve()
    require(not(root.is_relative_to(prior) or prior.is_relative_to(root)),'Source/output overlap')
    if root.exists():
        p=check(root);require(p['source_A55r2']==str(prior),'Frozen source differs')
        print('A60 frozen plan reused; completed work retained.',flush=True);return
    before=previous.check(prior);report=source.load_signed(prior/'summary.json')
    require(report['completed_cases']==report['planned_cases']==10,'Need all10 completed A55r2')
    identities=dict(before['identity_files'])
    for n in ('plan.json','summary.json','evaluation.json'):identities[str(prior/n)]=source.digest(prior/n)
    for c in before['cases']:require(previous.item(prior,before,c,'control') is not None,'Incomplete parent control')
    own=[Path(__file__),Path(rules.__file__),Path(observer.__file__),CARD,PROJECTS/'utils/test_direction_refresh_control.py',
         PROJECTS/'utils/test_fixed_time_geometry_value.py',PROJECTS/'run_a60_20261027.sh',PROJECTS/'utils/a60_session.py',Path(support.__file__)]
    p=dict(before);p.update(experiment='A60_direction_refresh',source_A55r2=str(prior),source_evaluation=str(prior/'evaluation.json'),
        phases=['control'],arms=list(rules.ARMS),intervals=list(rules.INTERVALS),
        refresh_steps={str(i):list(rules.refresh_steps(i)) for i in rules.INTERVALS},identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=0,planned_diagnostic_nfe=900,planned_backward_attempts_max=550,
        limits='Five OLD FFHQ development images x SR/IP, one seed, fixed30 grid. Current v and K||v|| at every step; '
               'only direction refresh interval1/2/4 changes. First/last mandatory. Stale direction contains old x,t AND v, '
               'so this is direction lag, not an exact frozen Jacobian on current v. All forwards retain graph-enabled arithmetic. '
               'Input VJP saving is not forward-NFE saving. Same-run fixed-order timing includes diagnostic writes and shared load; '
               'no production walltime guarantee, independent transfer or curvature-control claim.')
    root.parent.mkdir(parents=True,exist_ok=True);parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A60 PREPARE PASS:10 OLD cases x3 arms x30=900 new diagnostic forwards;<=550 input VJPs; no step change.',flush=True)


def item(root,p,c,phase):
    paths=sorted((root/c['key']).glob('control_*/run.json'))
    if not paths:return None
    require(len(paths)==1,'Multiple attempts; no output selection')
    f=paths[0];r=source.read(f)
    if r['status']!='complete':return None
    r=source.load_signed(f)
    require(r['case']==c and r['plan_sha256']==source.digest(root/'plan.json') and
        r['calls']==list(reversed(p['grid']))*3 and r['nfe']==90 and
        r['backward_attempts']==r['backward_completed']<=55 and set(r['traces'])==set(rules.ARMS) and
        all(g['passed'] for g in r['gates'].values()),'Direction refresh contract differs')
    count=0
    for mode,interval in zip(rules.ARMS,rules.INTERVALS):
        require(len(r['traces'][mode])==30,'Incomplete direction trajectory')
        for k,row in enumerate(r['traces'][mode]):
            z=row['readout'];due=k in rules.refresh_steps(interval);count+=z['backward_calls']
            require(z['refresh_due']==due and (due or z['backward_calls']==0),'Hidden current-u query on stale step')
            if z['resolved']:require(z['applied_norm_relative_error']<=2e-6,'Current physical norm changed')
        require(r['arm_backward_attempts'][mode]<=len(rules.refresh_steps(interval)),'Refresh budget exceeded')
    require(count==r['backward_completed'],'VJP trace counter mismatch')
    require(all(source.digest(f.parent/n)==sha for n,sha in r['artifacts'].items()),'Artifacts changed')
    return f,r,'CURRENT'


def sample(root,p,c,phase,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error
    if item(root,p,c,phase):return
    operator,mask,opgate,state,original,y,z,positions,noises=parent.setup(p,c,model,device)
    folder,r=engine.old.previous.new_attempt(root,p,c,phase);path=folder/'run.json'
    external=rng_state(device);outputs={};r['arm_backward_attempts']={}
    with engine.old.previous.counted(model,path,r,90) as backward, rules.current.base.input_vjp_parameters(model.model), \
         model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling RNG')):
        for mode,interval in zip(rules.ARMS,rules.INTERVALS):
            x=state['init_noise'].to(device);cache=None;r['traces'][mode]=[];b0=r['backward_attempts']
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
            with parent.P.precision.ieee(torch):
                for k,t in enumerate(positions.nominal):
                    s=positions.nominal[k+1] if k<29 else -1
                    field,gamma,row,cache=rules.sensed_field(z,y,x,t,s,positions,c['sigma'],k,interval,cache,mask,backward)
                    r['backward_completed']+=row['backward_calls']
                    if row.get('resolved'):
                        actual=float(field['q'].detach().double().square().mean().sqrt());expected=row['coefficient']*row['v_RMS']
                        err=abs(actual-expected)/max(expected,1e-30);require(math.isfinite(err) and err<=2e-6,'Applied physical norm mismatch')
                        row['applied_norm_relative_error']=err
                    require(row['refresh_due'] or row['backward_calls']==0,'Forbidden VJP on stale step')
                    r['traces'][mode].append(dict(step=k,t=t,readout=row))
                    with torch.no_grad():x=parent.old.parent.rules.risk.advance(z,x,field,positions,k,noises,gamma)
                    require(bool(torch.isfinite(x).all()),'Nonfinite direction trajectory:'+mode)
            torch.cuda.synchronize(device);r['arm_seconds'][mode]=time.perf_counter()-started
            r['arm_peak_memory_bytes'][mode]=torch.cuda.max_memory_allocated(device)
            r['arm_backward_attempts'][mode]=r['backward_attempts']-b0;outputs[mode]=x.detach().cpu()
            print('A60 arm:',c['key'],mode,'inputB=',r['arm_backward_attempts'][mode],'seconds=',round(r['arm_seconds'][mode],2),flush=True)
        parent.invariant(model,z,state,device,external,p)
        f=Path(p['norm_sources'][c['key']]);cached=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)['NORM_ONLY']
        parity=relative_error(outputs['CURRENT'],cached);require(math.isfinite(parity) and parity<=1e-6,'CURRENT U replay parity failed')
        parent.old.parent.base.save_tensor(outputs,folder/'outputs.pt')
        signed(folder/'decisions_before_labels.json',dict(case=c,traces=r['traces'],GT_used=False,plan_sha256=r['plan_sha256']))
        r['artifacts']={n:source.digest(folder/n) for n in ('outputs.pt','decisions_before_labels.json')}
        r['gates']=dict(operator=opgate,calls_rng_weights=dict(passed=True),current_U_replay=dict(passed=True,relative_error=parity),
                         no_hidden_current_direction=dict(passed=True),current_norm_rule=dict(passed=True,tolerance=2e-6))
    r['status']='complete';signed(path,r)


@contextmanager
def executor_bindings():
    from unittest.mock import patch
    with patch.object(engine,'check',check),patch.object(engine,'item',item),patch.object(engine,'sample',sample):yield


def summarize(root,p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    pairs=[(c,item(root,p,c,'control')) for c in p['cases']];require(all(v for c,v in pairs),'All10 outputs required before GT')
    old={r['case']['key']:r for r in source.load_signed(Path(p['source_evaluation']))['rows']};rows=[];values=[]
    with parent.scan.preserved_rng(),torch.no_grad():
        for c,(f,r,_) in pairs:
            outputs=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True);gt=load_image_as_tensor(c['image']).cpu()
            metrics={n:compute_all_metrics(x,gt,lpips_net='alex') for n,x in outputs.items()}
            metrics['C_IEEE']=dict(old[c['key']]['metrics']['C_IEEE'])
            op,mask=parent.old.make_operator(Path(p['source_A41']),c,'cpu')
            a41=Path(p['source_A41']);p41=source.load_signed(a41/'plan.json');tr,tp=parent.old.training_context(a41,p41,c)
            _,train=parent.old.parent.done(tr,tp,c,'BASELINE');obs=torch.load(train['measurement']['path'],map_location='cpu',weights_only=True)
            y=obs['mean']+c['sigma']*obs['standard_noise']
            for n,x in outputs.items():metrics[n]['loss']=.5*(rules.current.base.informative_rms(y-op.H(x),mask)/c['sigma'])**2
            resolved={n:sum(v['readout']['resolved'] for v in r['traces'][n]) for n in rules.ARMS}
            rows.append(dict(case=c,metrics=metrics,resolved=resolved,arm_seconds=r['arm_seconds'],input_VJPs=r['arm_backward_attempts']))
            values.append(r)
    groups=[]
    for task in rules.current.TASKS:
        subset=[r for r in rows if r['case']['task']==task];arms={}
        for arm in ('LAG2','LAG4'):
            delta={ref:{m:mean(r['metrics'][arm][m]-r['metrics'][ref][m] for r in subset) for m in ('psnr','ssim','lpips')}
                   for ref in ('CURRENT','C_IEEE')}
            current,c=delta['CURRENT'],delta['C_IEEE'];wins=sum(r['metrics'][arm]['psnr']-r['metrics']['CURRENT']['psnr']>=-.10 for r in subset)
            full=all(r['resolved'][n]==30 for r in subset for n in rules.ARMS)
            candidate=(full and current['psnr']>=-.05 and current['ssim']>=-.002 and current['lpips']<=.005 and wins>=4 and
                       c['psnr']>=.05 and c['ssim']>=-.002 and c['lpips']<=.005)
            arms[arm]=dict(mean_delta=delta,fully_resolved=bool(full),images_within_point10=int(wins),
                lower_VJP_cost_development_candidate=bool(candidate),
                same_run_median_seconds_ratio=float(median(r['arm_seconds'][arm]/r['arm_seconds']['CURRENT'] for r in subset)))
        groups.append(dict(task=task,arms=arms))
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_DIRECTION_REFRESH_DIAGNOSIS_ONLY',completed_cases=10,
        planned_cases=10,image_clusters=5,method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in values),
        input_backward_attempts=sum(r['backward_attempts'] for r in values),planned_input_backward_max=550,
        input_VJPs_by_arm={n:sum(r['arm_backward_attempts'][n] for r in values) for n in rules.ARMS},
        measured_seconds_by_arm={n:sum(r['arm_seconds'][n] for r in values) for n in rules.ARMS},
        peak_memory_bytes_by_arm={n:max(r['arm_peak_memory_bytes'][n] for r in values) for n in rules.ARMS},
        linear_solve_actions={n:{key:sum(v['readout'].get('cg',{}).get(key,0) for r in values for v in r['traces'][n])
                for key in ('linear_calls','adjoint_calls')} for n in rules.ARMS},groups=groups,per_image=rows,limits=p['limits'])
    signed(root/'evaluation.json',dict(rows=rows,outputs_frozen_before_GT=True));signed(root/'summary.json',result)
    print('A60 Summary:',json.dumps({n:v for n,v in result.items() if n!='per_image'},ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True)


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
