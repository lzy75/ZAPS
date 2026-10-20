"""A21 original-step state selection: every scientific arm uses 300 NFE; cached NFE=0."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_direction_value as fit
from utils import diag_ieee_response_value as protocol
from utils import diag_terminal_state_strength as old
from utils.diag_state_trajectory import digest

WRITE=old.WRITE;METRICS=old.METRICS;GAMMAS=(0,1,2)
MODELS=dict(physical=fit.BASE,state=fit.DIRECTION)
ROOT='/data0/lzy/ZAPS-results/diag_equal_nfe_state_guidance/A21_20261020'
CARD=PROJECTS/'experiment_cards/A21_equal_nfe_state_guidance_20261020.md'


def admissible(delta,noise=1e-9):
    return delta['ssim']>=-.002 and delta['lpips']<=.005 and delta['loss']<=noise


def choose(predictions):
    eligible=[g for g in GAMMAS if admissible(predictions[str(g)]) and predictions[str(g)]['psnr']>=0]
    if 1 not in eligible:raise ValueError('Nominal action must always be available')
    return min(eligible,key=lambda g:(-predictions[str(g)]['psnr'],abs(g-1),g))


def evaluate(rows,actions):
    if len(rows)!=len(actions):raise ValueError('Action/row count differs')
    delta={k:statistics.mean(r['branches'][str(g)][k]-r['branches']['1'][k] for r,g in zip(rows,actions)) for k in METRICS}
    noise={k:statistics.mean(old.limits(r['floors'])[k] for r in rows) for k in METRICS}
    return dict(mean_delta=delta,numerical_limit=noise,quality_guard=admissible(delta,noise['loss']),
                positive_psnr=delta['psnr']>noise['psnr'],actions=list(actions))


def best_fixed(rows):
    choices={str(g):evaluate(rows,[g]*len(rows)) for g in GAMMAS}
    eligible=[g for g in GAMMAS if choices[str(g)]['quality_guard'] and choices[str(g)]['mean_delta']['psnr']>=0]
    g=min(eligible,key=lambda g:(-choices[str(g)]['mean_delta']['psnr'],abs(g-1),g))
    return g,choices


def policies(train,test_features):
    # test_features has no GT labels/branches. Only calibration rows enter ridge fits.
    if any('branches' in r for r in test_features):raise ValueError('Test labels forbidden while choosing actions')
    out={}
    for name,columns in MODELS.items():
        models={};predictions=[{str(g):dict.fromkeys(METRICS,0.) for g in GAMMAS} for _ in test_features]
        for g in (0,2):
            models[str(g)]={}
            for metric in METRICS:
                labels=[dict(r['features'],target=r['branches'][str(g)][metric]-r['branches']['1'][metric]) for r in train]
                values,model=fit.ridge_predict(labels,test_features,columns,'target');models[str(g)][metric]=model
                for pred,value in zip(predictions,values):pred[str(g)][metric]=value
        out[name]=dict(models=models,predictions=predictions,actions=[choose(p) for p in predictions])
    out['calibration_fixed_gamma']=best_fixed(train)[0]
    return out


def pair_advantage(rows,actions,reference):
    values=[r['branches'][str(a)]['psnr']-r['branches'][str(b)]['psnr'] for r,a,b in zip(rows,actions,reference)]
    mean=statistics.mean(values);floor=statistics.mean(old.limits(r['floors'])['psnr'] for r in rows)
    wins=sum(statistics.mean(v-old.limits(r['floors'])['psnr'] for r,v in zip(rows,values) if r['case']['image_id']==image)>0 for image in sorted({r['case']['image_id'] for r in rows}))
    return dict(mean_psnr_delta=mean,images_psnr_better=wins,passed=mean>floor and wins>=2)


def analyze(rows,policy):
    train=[r for r in rows if r['cohort']=='development'];test=[r for r in rows if r['cohort']=='transfer']
    if len(train)!=6 or len(test)!=6:raise ValueError('Need fixed 3+3 image cohorts, both seeds')
    fixed,controls=best_fixed(test);base=policy['physical']['actions'];full=policy['state']['actions']
    frozen=[policy['calibration_fixed_gamma']]*6;best=[fixed]*6
    evaluation=dict(calibration_fixed=evaluate(test,frozen),physical=evaluate(test,base),state=evaluate(test,full))
    pairs={name:pair_advantage(test,full,ref) for name,ref in dict(calibration_fixed=frozen,strongest_fixed_on_transfer=best,physical=base).items()}
    oracle=None
    for actions in itertools.product(GAMMAS,repeat=6):
        value=evaluate(test,actions)
        if value['quality_guard'] and (oracle is None or value['mean_delta']['psnr']>oracle['mean_delta']['psnr']):oracle=value
    headroom=oracle['mean_delta']['psnr']-controls[str(fixed)]['mean_delta']['psnr']
    images=sorted({r['case']['image_id'] for r in test});by_image={i:{r['case']['sampling_seed']:g for r,g in zip(test,full) if r['case']['image_id']==i} for i in images}
    permutations=[]
    for order in itertools.permutations(images):
        mapping=dict(zip(images,order));actions=[]
        for r in test:
            i=r['case']['image_id'];actions.append(by_image[mapping[i]][r['case']['sampling_seed']])
        permutations.append(evaluate(test,actions))
    shuffle_mean=statistics.mean(v['mean_delta']['psnr'] for v in permutations)
    state=evaluation['state'];different=len(set(full))>1
    support=(headroom>=.01 and state['quality_guard'] and state['positive_psnr'] and all(v['passed'] and v['mean_psnr_delta']>=.01 for v in pairs.values()) and different and
             state['mean_delta']['psnr']>shuffle_mean+state['numerical_limit']['psnr'])
    return dict(completed_cases=12,image_clusters=6,method_nfe=0,diagnostic_nfe=0,deployment_nfe_per_case=dict(original=300,fixed=300,physical=300,state=300),
        classification='NO_USEFUL_STATE_SELECTION_HEADROOM_IN_THIS_ACTION_SET' if headroom<.01 else 'EXPLORATORY_STATE_INCREMENT_AT_EQUAL_NFE' if support else 'STATE_INCREMENT_NOT_SUPPORTED_AT_EQUAL_NFE',
        state_features={k:list(v) for k,v in MODELS.items()},calibration_fixed_gamma=policy['calibration_fixed_gamma'],
        strongest_fixed_gamma_on_transfer_for_evaluation_only=fixed,fixed_controls=controls,policy_evaluation=evaluation,state_increment=pairs,
        oracle_for_diagnosis_only=oracle,oracle_psnr_headroom_over_strongest_fixed=headroom,
        image_permutation_mean_psnr_gain=shuffle_mean,image_permutation_results=permutations,state_actions_vary=different,
        cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=2312,total=9512),
        limits='Original final-step only, not an extra evaluation or full-trajectory controller. Reused 3 calibration+3 transfer images; GT used for calibration and offline evaluation only. Oracle/transfer-best-fixed are diagnostic references, never deployed policies. No independent confirmation or curvature claim.')


def prepare(args):
    source=Path(args.a14_dir).resolve();p=old.read(source/'plan.json');root=Path(args.root).resolve();identities={};cases=[]
    if digest(source/'plan.json')!=(source/'plan.sha256').read_text().strip():raise ValueError('A14 source plan changed')
    def bind(path,sha=None):
        path=Path(path);value=digest(path)
        if sha is not None and value!=sha:raise ValueError('Source changed: '+str(path))
        identities[str(path)]=value
    bind(source/'plan.json');bind(source/'plan.sha256')
    config=p['stage_plans']['14']['config'];grid=p['stage_plans']['14']['grid'];checkpoint=p['stage_plans']['14']['checkpoint_sha256']
    for c in p['cases']:
        s=p['case_sources'][c['key']];folder=Path(s['directory']);r=old.read(folder/'run.json');bind(folder/'run.json',s['record_sha256'])
        parent_plan=None;parent_sha=s.get('protocol_plan_sha256')
        if parent_sha is not None:
            bind(s['protocol_source'],parent_sha);parent_plan=old.read(s['protocol_source'])
        cfg,tau=protocol.source_protocol(r,parent_plan,parent_sha)
        if r['status']!='complete' or not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or cfg!=config or tau!=grid:raise ValueError('Incompatible active source')
        if r['rows'][-1]['t']!=0 or r['rows'][-1]['parameter_index']!=0:raise ValueError('Need original terminal state')
        for n in ('trajectory.pt','last_opt.pt'):bind(folder/n,r['artifacts_sha256'][n])
        bind(r['measurement']['path'],r['measurement']['sha256']);bind(c['image'],c['image_sha256'])
        cases.append(dict(case={k:v for k,v in c.items() if k!='cohort'},cohort=c['cohort'],source=str(folder),measurement=r['measurement']))
    if {(s['case']['image_id'],s['case']['sampling_seed']) for s in cases}!={(f'{i:05d}',seed) for i in range(20,26) for seed in (1000,2000)}:raise ValueError('Fixed cohort differs')
    own=[Path(__file__),CARD,PROJECTS/'run_a21_20261020.sh',PROJECTS/'utils/test_diag_equal_nfe_state_guidance.py',
        PROJECTS/'utils/diag_direction_value.py',PROJECTS/'utils/diag_cached_guidance_geometry.py',PROJECTS/'utils/diag_ieee_response_value.py',
        PROJECTS/'utils/metrics.py',PROJECTS/'modules/main_single.py',PROJECTS/'modules/degradations.py',PROJECTS.parent/'DPS/util/resizer.py']
    for path in own:bind(path)
    if any(root.is_relative_to(Path(s['source']).resolve()) for s in cases) or root.is_relative_to(source):raise ValueError('New root overlaps source')
    plan=dict(experiment='A21_equal_nfe_state_guidance',cases=cases,identity_files=identities,config=config,grid=grid,checkpoint_sha256=checkpoint,
        features={k:list(v) for k,v in MODELS.items()},gammas=list(GAMMAS),method_nfe=0,diagnostic_nfe=0,deployment_nfe_per_case=300,
        observation_step=29,observation_t=0,parameter_index=0,ridge_lambda=1,source_A14=str(source))
    if root.exists():
        if old.read(root/'plan.json')!=plan:raise ValueError('Frozen A21 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    return root,plan


def arrays(spec):
    import torch
    from utils.diag_zaps_trace_audit import relative_error
    source=Path(spec['source']);cache=torch.load(source/'trajectory.pt',map_location='cpu',weights_only=True)
    c=torch.load(source/'last_opt.pt',map_location='cpu',weights_only=True)
    clean=cache['clean'][29:30];q=cache['guidance'][29:30]
    if any(cache[n].shape!=(30,3,256,256) or cache[n].dtype!=torch.float32 for n in ('clean','guidance')) or not all(torch.isfinite(x).all() for x in (clean,q,c)):
        raise ValueError('Invalid original cache')
    error=relative_error(c,clean+q)
    if not math.isfinite(error) or error>1e-7:raise RuntimeError('Original terminal identity failed')
    return cache,c,clean,q,error


def observation(spec,operator):
    import torch
    from utils.diag_cached_guidance_geometry import vector_readouts
    cache,c,clean,q,error=arrays(spec);r=old.read(Path(spec['source'])/'run.json');rows=r['rows'];u=[rows[k]['noise_u'] for k in (27,28,29)]
    geo=vector_readouts(q,cache['guidance'][28:29],cache['guidance'][27:28],u[2],u[1],u[0])
    y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y'];R=float((y-operator.H(clean)).double().square().mean().sqrt()/.05)
    expected=rows[29]
    for a,b,atol in ((R,expected['residual_noise_ratio'],1e-8),(geo['rms'],expected['guidance_rms'],1e-12),(geo['cosine'],expected['guidance_cosine'],1e-8),(geo['variation3_per_u'],expected['guidance_variation3_per_u'],1e-8)):
        if a is None or b is None or not math.isclose(a,b,rel_tol=1e-6,abs_tol=atol):raise RuntimeError('Cached direction/physical observation differs')
    feature=fit.features(dict(residual_noise_ratio=R,guidance_rms=geo['rms'],guidance_cosine=geo['cosine'],guidance_variation3_per_u=geo['variation3_per_u']))
    if not feature['usable']:raise RuntimeError('Unresolved direction retained; analysis stopped')
    return dict(case=spec['case'],cohort=spec['cohort'],features=feature,observation=dict(R=R,qRMS=geo['rms'],cosine=geo['cosine'],turn=geo['turn'],Omega3=geo['variation3_per_u'],t=0,k=29),terminal_identity_error=error)


def measured(spec,obs,operator):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    _,c,clean,q,_=arrays(spec)
    outputs={str(g):clean+g*q for g in GAMMAS};outputs['1']=c
    # Fixed candidate tensors exist before GT is accessed. No future network call follows t0.
    gt=load_image_as_tensor(spec['case']['image']).cpu();y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y']
    branches={a:dict(loss=float((y-operator.H(x)).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex')) for a,x in outputs.items()}
    source=old.read(Path(spec['source'])/'run.json');reference=source.get('final_metrics',source.get('baseline'))
    drift={k:abs(branches['1'][k]-reference[k]) for k in ('psnr','ssim','lpips')}
    if any(drift[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Original baseline metric drift')
    return dict(**obs,branches=branches,floors=dict(loss=0.,**drift),source_gates_passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a14-dir',default='/data0/lzy/ZAPS-results/diag_ieee_response_value/A14_20261020');p.add_argument('--summary',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.summary:print('A21 Summary:',json.dumps(old.read(root/'summary.json')),flush=True);return
    import fcntl,torch
    from unittest.mock import patch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root,plan=prepare(args)
        if (root/'run.json').exists():
            r=old.read(root/'run.json')
            if (r['status']!='complete' or digest(root/'run.json')!=(root/'run.sha256').read_text().strip() or r['plan_sha256']!=digest(root/'plan.json') or
                digest(root/'policy.json')!=r['policy_sha256'] or old.read(root/'summary.json')!=r['analysis'] or not all(g['passed'] for g in r['gates'].values())):
                raise ValueError('Interrupted/invalid A21 retained')
            print('A21 Summary:',json.dumps(old.read(root/'summary.json')),flush=True);return
        r=dict(status='running',plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0);WRITE(root/'run.json',r);start=time.perf_counter();torch.set_num_threads(2)
        try:
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device='cpu',scale_factor=4,noise_sigma=0.),'legacy_bicubic'))
            with torch.random.fork_rng(devices=[]),torch.no_grad(),patch('modules.main_single.load_diffusion_model',side_effect=RuntimeError('UNet load forbidden in A21')):
                observations=[observation(s,operator) for s in plan['cases']];WRITE(root/'states.json',observations)
                train=[measured(s,o,operator) for s,o in zip(plan['cases'],observations) if s['cohort']=='development']
                test_obs=[o for o in observations if o['cohort']=='transfer'];test_features=[o['features'] for o in test_obs]
                policy=policies(train,test_features);policy.update(transfer_case_order=[o['case']['key'] for o in test_obs],test_GT_used=False)
                WRITE(root/'policy.json',policy);(root/'policy.sha256').write_text(digest(root/'policy.json')+'\n')
                # Only after actions are frozen do transfer GT metrics enter the run.
                test=[measured(s,o,operator) for s,o in zip(plan['cases'],observations) if s['cohort']=='transfer']
            out=analyze(train+test,policy)
            for path,sha in plan['identity_files'].items():
                if digest(path)!=sha:raise ValueError('Source/code changed')
            r.update(status='complete',seconds=time.perf_counter()-start,rows=train+test,policy_sha256=digest(root/'policy.json'),analysis=out,
                gates=dict(source_protocol=dict(passed=True),original_terminal=dict(passed=True),causal_states_actions_before_transfer_GT=dict(passed=True),zero_extra_UNet_equal_300=dict(passed=True)))
            WRITE(root/'run.json',r);(root/'run.sha256').write_text(digest(root/'run.json')+'\n');WRITE(root/'summary.json',out)
        except BaseException as error:r.update(status='failed',error=repr(error),seconds=time.perf_counter()-start);WRITE(root/'run.json',r);raise
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A21':str(root)});print('A21 Summary:',json.dumps(out),'\nRecords:',root,'\nArchive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
