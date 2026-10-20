"""A22 original-grid one-pulse opportunity; every arm=300 NFE, 1296 diagnostic calls."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_equal_nfe_state_guidance as parent
from utils import diag_coupled_refinement as coupled
from utils import diag_clean_response_precision as precision
from utils import diag_terminal_continuation as terminal
from utils.diag_state_trajectory import digest

WRITE=parent.WRITE;METRICS=parent.METRICS
POINTS=((8,903),(19,641),(24,319),(27,54),(28,14));GAMMAS=(.5,2.)
ARMS=('C','T0_G0','T0_G2',*[f'T{t}_G{g:g}' for _,t in POINTS for g in GAMMAS])
CALLS=30+2*sum(29-k for k,_ in POINTS)
ROOT='/data0/lzy/ZAPS-results/diag_equal_nfe_path_opportunity/A22_20261020'
CARD=PROJECTS/'experiment_cards/A22_equal_nfe_path_opportunity_20261020.md'


def endpoint(rows,actions):
    if not rows or len(rows)!=len(actions):raise ValueError('Nonempty matching action/row counts required')
    delta={k:statistics.mean(r['branches'][a][k]-r['branches']['C'][k] for r,a in zip(rows,actions)) for k in METRICS}
    floor=statistics.mean(parent.old.limits(r['floors'])['loss'] for r in rows)
    image_psnr={i:statistics.mean(r['branches'][a]['psnr']-r['branches']['C']['psnr'] for r,a in zip(rows,actions) if r['case']['image_id']==i) for i in sorted({r['case']['image_id'] for r in rows})}
    worsening={k:sum((r['branches'][a][k]-r['branches']['C'][k])*(1 if k=='lpips' else -1)>0 for r,a in zip(rows,actions)) for k in ('psnr','ssim','lpips')}
    return dict(mean_delta=delta,quality_guard=parent.admissible(delta,floor),actions=list(actions),per_image_psnr_delta=image_psnr,cases_worsened=worsening)


def opportunity(rows,arms):
    if not rows or 'C' not in arms or len(set(arms))!=len(arms):raise ValueError('Need nonempty cases and distinct arms including C')
    fixed={a:endpoint(rows,[a]*len(rows)) for a in arms}
    feasible=[a for a in arms if fixed[a]['quality_guard'] and fixed[a]['mean_delta']['psnr']>=0]
    best=min(feasible,key=lambda a:(-fixed[a]['mean_delta']['psnr'],a!='C',arms.index(a)))
    # Exact discrete oracle in small vectorized chunks; no model/threshold fitting.
    values=np.asarray([[[r['branches'][a][k]-r['branches']['C'][k] for k in METRICS] for a in arms] for r in rows],dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('Nonfinite endpoints')
    n=len(rows);base=len(arms);powers=base**np.arange(n,dtype=np.int64);total=base**n;oracle=fixed[best];best_gain=oracle['mean_delta']['psnr']
    floor=statistics.mean(parent.old.limits(r['floors'])['loss'] for r in rows)
    for start in range(0,total,8192):
        ids=np.arange(start,min(start+8192,total),dtype=np.int64)
        picks=(ids[:,None]//powers[None,:])%base
        means=values[np.arange(n)[None,:],picks].mean(axis=1)
        valid=(means[:,0]<=floor)&(means[:,2]>=-.002)&(means[:,3]<=.005)
        if valid.any():
            gains=np.where(valid,means[:,1],-np.inf);i=int(np.argmax(gains))
            if gains[i]>best_gain+1e-12:
                best_gain=float(gains[i]);oracle=endpoint(rows,[arms[j] for j in picks[i]])
    gap=oracle['mean_delta']['psnr']-fixed[best]['mean_delta']['psnr']
    return dict(best_fixed=best,fixed=fixed,oracle_for_diagnosis_only=oracle,oracle_psnr_headroom=gap,
        classification='ACTION_SELECTION_OPPORTUNITY_ONLY' if gap>=.01 else 'NO_USEFUL_ACTION_SELECTION_HEADROOM',assignments_enumerated=total)


def summarize(records):
    complete=[r for r in records if r['status']=='complete'];groups=[]
    for cohort in ('development','transfer'):
        rows=[r for r in complete if r['cohort']==cohort]
        group=dict(cohort=cohort,completed_cases=len(rows),planned_cases=6)
        if len(rows)==6:
            group.update(global_opportunity=opportunity(rows,ARMS),fixed_t_opportunity={str(t):opportunity(rows,('C',f'T{t}_G0.5',f'T{t}_G2')) for _,t in POINTS})
        groups.append(group)
    train=[r for r in complete if r['cohort']=='development'];test=[r for r in complete if r['cohort']=='transfer']
    calibration=None
    if len(train)==6 and len(test)==6:
        chosen=groups[0]['global_opportunity']['best_fixed'];calibration=dict(chosen_on_development=chosen,transfer=endpoint(test,[chosen]*6))
    cost=sum(r['diagnostic_nfe'] for r in records)
    return dict(classification='OPPORTUNITY_DIAGNOSTIC_ONLY' if len(complete)==12 else 'INCOMPLETE_PILOT',completed_cases=len(complete),planned_cases=12,image_clusters=len({r['case']['image_id'] for r in complete}),method_nfe=0,diagnostic_nfe=cost,
        deployment_nfe_per_case=300,groups=groups,calibration_fixed_transfer=calibration,
        cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=2312+cost,total=9512+cost),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records),
        limits='Oracle/action opportunity only; no new state policy. One pulse among fixed original steps, every scientific arm=300 NFE. Reused 3+3 images with two dependent seeds. No GT output selection in an online method; no generalization/curvature claim.')


def prepare(args):
    source=Path(args.a21_dir).resolve();p=parent.old.read(source/'plan.json');r=parent.old.read(source/'run.json');root=Path(args.root).resolve()
    if (digest(source/'plan.json')!=(source/'plan.sha256').read_text().strip() or digest(source/'run.json')!=(source/'run.sha256').read_text().strip() or
        r['status']!='complete' or r['plan_sha256']!=digest(source/'plan.json') or r['analysis']!=parent.old.read(source/'summary.json') or
        r['analysis']['classification']!='NO_USEFUL_STATE_SELECTION_HEADROOM_IN_THIS_ACTION_SET' or digest(source/'policy.json')!=r['policy_sha256'] or
        not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Need declared completed A21 no-headroom context')
    identities=dict(p['identity_files'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('A21 bound source/code changed: '+path)
    def bind(path,sha=None):
        path=Path(path);v=digest(path)
        if sha is not None and v!=sha:raise ValueError('Source changed: '+str(path))
        identities[str(path)]=v
    for n in ('plan.json','plan.sha256','run.json','run.sha256','summary.json','policy.json','policy.sha256'):bind(source/n)
    for s in p['cases']:
        record=parent.old.read(Path(s['source'])/'run.json');bind(Path(s['source'])/'pre_last_state.pt',record['artifacts_sha256']['pre_last_state.pt'])
        if [(record['rows'][k]['t'],record['rows'][k]['parameter_index']) for k,_ in POINTS]!=[(t,29-k) for k,t in POINTS]:raise ValueError('Original fixed checkpoints differ')
    a13=Path(args.a13_dir).resolve();bind(a13/'plan.json',(a13/'plan.sha256').read_text().strip());r13=precision.existing(a13,14)
    if r13 is None:raise ValueError('Need frozen native runtime context')
    bind(a13/'t14/run.json');bind(a13/'t14/run.sha256')
    own=[Path(__file__),CARD,PROJECTS/'run_a22_20261020.sh',PROJECTS/'utils/test_diag_equal_nfe_path_opportunity.py',
         PROJECTS/'utils/diag_coupled_refinement.py',PROJECTS/'utils/diag_guidance_local_probe.py',PROJECTS/'utils/diag_terminal_continuation.py',PROJECTS/'modules/zaps_algorithm.py']
    for path in own:bind(path)
    if root.is_relative_to(source) or source.is_relative_to(root) or any(root.is_relative_to(Path(s['source']).resolve()) for s in p['cases']):raise ValueError('New root overlaps source')
    plan=dict(experiment='A22_equal_nfe_path_opportunity',cases=p['cases'],config=p['config'],grid=p['grid'],checkpoint_sha256=p['checkpoint_sha256'],
        identity_files=identities,worker_identity_files={str(x):identities[str(x)] for x in own},runtime=r13['runtime'],native_policy=r13['native_policy'],
        points=[list(x) for x in POINTS],gammas=list(GAMMAS),arms=list(ARMS),planned_method_nfe=0,planned_diagnostic_nfe=12*CALLS,
        scientific_nfe_per_case=300,source_A21=str(source))
    if root.exists():
        if parent.old.read(root/'plan.json')!=plan:raise ValueError('Frozen A22 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    print('A22 PREPARE PASS: 12 cases x108 diagnostic calls; all scientific arms=300 NFE.',flush=True)


def continue_suffix(z,y,x,rows,start,noises,evaluate,before=None):
    for k in range(start,len(rows)):
        row=rows[k]
        if before is not None:before(k,row)
        f=evaluate(z,y,x,row['t'],row['parameter_index'])
        x=(terminal.terminal_output(z,x,f) if row['t_prev']<0 else coupled.posterior(x,f['clean'],row['t'],row['t_prev'],z.dm.alphas_cumprod,noises[k])+f['q'])
    return x


def verify_case_sources(plan,spec):
    paths=[Path(spec['source'])/n for n in ('run.json','trajectory.pt','last_opt.pt','pre_last_state.pt')]
    paths += [Path(spec['measurement']['path']),Path(spec['case']['image'])]
    for path in paths:
        if digest(path)!=plan['identity_files'][str(path)]:raise ValueError('Case source changed: '+str(path))


def load_record(root,spec):
    path=root/spec['case']['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A22 retained')
        return None
    r=parent.old.read(path)
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['plan_sha256']!=digest(root/'plan.json') or r['case']!=spec['case'] or
        r['diagnostic_nfe']!=CALLS or r['method_nfe']!=0 or len(r['calls'])!=CALLS or digest(path.parent/'outputs.pt')!=r['outputs_sha256'] or
        set(r['branches'])!=set(ARMS) or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Invalid/interrupted A22 retained')
    return r


def run_case(root,plan,spec,model,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    folder=root/spec['case']['key'];folder.mkdir(exist_ok=False);start=time.perf_counter();context={}
    r=dict(status='running',case=spec['case'],cohort=spec['cohort'],plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
    def save():r['seconds']=time.perf_counter()-start;WRITE(folder/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;r['calls'].append(dict(**context,t=int(inputs[1].flatten()[0])));save()
        if r['diagnostic_nfe']>CALLS or r['calls'][-1]['t']!=context['expected_t']:raise RuntimeError('A22 call/time budget failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        verify_case_sources(plan,spec)
        src=Path(spec['source']);original_record=parent.old.read(src/'run.json');rows=original_record['rows'];external=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True);cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True)
            original=torch.load(src/'last_opt.pt',map_location='cpu',weights_only=True)
            if len(rows)!=30 or any(cache[n].shape!=(30,3,256,256) or cache[n].dtype!=torch.float32 or not torch.isfinite(cache[n]).all() for n in ('latent_inputs','clean','guidance')):raise ValueError('Original cache contract differs')
            if state['tau'].tolist()!=plan['grid'] or not torch.equal(cache['latent_inputs'][:1],state['init_noise']):raise ValueError('Original grid/init differs')
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            noises,replay_gate=coupled.replay_noises(state,cache,dict(rows=rows),z,y,device,original)
            parity=[]
            def checked(z_,y_,x,t,index):
                k=29-index;f=evaluate_field(z_,y_,x,t,index)
                values={name:relative_error(a,b) for name,a,b in (('latent',x.cpu(),cache['latent_inputs'][k:k+1]),('clean',f['clean'].cpu(),cache['clean'][k:k+1]),('q',f['q'].cpu(),cache['guidance'][k:k+1]))}
                if any(not math.isfinite(v) or v>1e-6 for v in values.values()):raise RuntimeError('Custom original replay differs')
                parity.append(dict(step=k,clip_fraction=f['clip_fraction'],**values));return f
            def before(k,row):context.update(step=k,expected_t=row['t'])
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unaccounted sampling noise')):
                context.update(arm='C_replay');repeated=continue_suffix(z,y,state['init_noise'].to(device),rows,0,noises,checked,before).cpu()
                final_error=relative_error(original,repeated)
                if not math.isfinite(final_error) or final_error>1e-6:raise RuntimeError('Final baseline parity failed')
                outputs=dict(C=original,T0_G0=cache['clean'][29:30],T0_G2=cache['clean'][29:30]+2*cache['guidance'][29:30])
                for k,t in POINTS:
                    for gamma in GAMMAS:
                        name=f'T{t}_G{gamma:g}';context.update(arm=name)
                        changed=cache['latent_inputs'][k+1:k+2].to(device)+(gamma-1)*cache['guidance'][k:k+1].to(device)
                        outputs[name]=continue_suffix(z,y,changed,rows,k+1,noises,evaluate_field,before).cpu()
                        if not torch.isfinite(outputs[name]).all():raise RuntimeError('Nonfinite counterfactual endpoint')
            if set(outputs)!=set(ARMS) or r['diagnostic_nfe']!=CALLS or len(parity)!=30:raise RuntimeError('Endpoint/call contract failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('PRE parameters/grid changed')
            tmp=folder/'outputs.pt.tmp';torch.save(dict(outputs=outputs,baseline_repeat=repeated),tmp);tmp.replace(folder/'outputs.pt');r['outputs_sha256']=digest(folder/'outputs.pt')
            gt=load_image_as_tensor(spec['case']['image']).cpu()
            def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
            branches={a:metrics(x) for a,x in outputs.items()};repeat=metrics(repeated);reference=original_record.get('final_metrics',original_record.get('baseline'))
            if abs(repeat['loss']-branches['C']['loss'])>1e-7+2e-5*abs(branches['C']['loss']):raise RuntimeError('Original physics replay drift')
            drift={k:abs(branches['C'][k]-reference[k]) for k in ('psnr','ssim','lpips')}
            if any(drift[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Original quality convention differs')
            floors={k:max(drift.get(k,0.),abs(repeat[k]-branches['C'][k])) for k in METRICS}
            states=[]
            for k,t in POINTS:
                row=rows[k];feature=parent.fit.features(row)
                states.append(dict(step=k,t=t,noise_u=row['noise_u'],features=feature,clip_fraction=parity[k]['clip_fraction']))
            r.update(branches=branches,floors=floors,action_states=states)
        if not same_rng(external,rng_state(device)) or precision.precision_state(torch)!=plan['native_policy']:raise RuntimeError('RNG/precision gate failed')
        verify_case_sources(plan,spec)
        for path,sha in plan['worker_identity_files'].items():
            if digest(path)!=sha:raise ValueError('Worker code changed during run')
        r['gates']=dict(original_noise=replay_gate,original_replay=dict(passed=True,states=parity,final_error=final_error),
            metric_parity=dict(passed=True,drift=drift),parameters_rng_budget=dict(passed=True),sources_unchanged=dict(passed=True))
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')
        print('A22 CASE:',spec['case']['key'],'diagnostic=',r['diagnostic_nfe'],flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def run(args):
    import fcntl,torch
    root=Path(args.root).resolve();plan=parent.old.read(root/'plan.json')
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[s for i,s in enumerate(sorted(plan['cases'],key=lambda s:s['case']['key'])) if i%2==args.shard and load_record(root,s) is None]
        if not todo:return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime'] or precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Native runtime differs; no new calls')
        torch.set_num_threads(2);coupled.archive.recovery.require_space(root)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model=coupled.archive.recovery.load_model(plan,args.device)
            for spec in todo:run_case(root,plan,spec,model,args.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a21-dir',default=parent.ROOT);p.add_argument('--a13-dir',default=precision.ROOT);p.add_argument('--device',default='cuda:0')
    a=p.add_mutually_exclusive_group(required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--shard',type=int,choices=(0,1));a.add_argument('--archive',action='store_true');a.add_argument('--summary',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=parent.old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    for path,sha in plan['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Worker/analysis code changed')
    if args.shard is not None:run(args);return
    if args.summary:
        if digest(root/'summary.json')!=(root/'summary.sha256').read_text().strip():raise ValueError('Summary changed')
        out=parent.old.read(root/'summary.json');compact={k:out[k] for k in ('classification','completed_cases','planned_cases','image_clusters','method_nfe','diagnostic_nfe','deployment_nfe_per_case','calibration_fixed_transfer','cumulative_queue_cost','limits')}
        compact['groups']=[]
        for g in out['groups']:
            v={k:g[k] for k in ('cohort','completed_cases','planned_cases')}
            if 'global_opportunity' in g:
                o=g['global_opportunity'];v.update(best_fixed=o['best_fixed'],best_fixed_result=o['fixed'][o['best_fixed']],oracle_psnr_headroom=o['oracle_psnr_headroom'],classification=o['classification'],fixed_t_headroom={t:x['oracle_psnr_headroom'] for t,x in g['fixed_t_opportunity'].items()})
            compact['groups'].append(v)
        print('A22 Summary:',json.dumps(compact),flush=True);print('Records:',root,flush=True);return
    records=[]
    for spec in plan['cases']:
        path=root/spec['case']['key']/'run.json'
        if path.exists():
            r=parent.old.read(path)
            if r['status']=='complete':load_record(root,spec)
            records.append(r)
    out=summarize(records);WRITE(root/'summary.json',out);(root/'summary.sha256').write_text(digest(root/'summary.json')+'\n');print('A22 Summary:',json.dumps(out),flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A22':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)
    print('Records:',root,flush=True)


if __name__=='__main__':main()
