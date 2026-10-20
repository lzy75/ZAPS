"""A14 IEEE response cohort + A15 frozen transfer analysis; 220 new diagnostic NFE."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_clean_field_response as field
from utils import diag_clean_response_precision as precision
from utils import diag_direction_value as old
from utils.diag_state_trajectory import digest
WRITE=field.WRITE
BASE='/data0/lzy/ZAPS-results/diag_ieee_response_value'
MODELS={'physical':old.BASE,'generic':old.BASE+('log_orthogonal_gain',),'direction':old.BASE+('log_gain_ratio',)}


def read(p):return field.read(p)


def source_protocol(record,bound_plan=None,bound_plan_sha=None):
    """A2a embeds config/grid; A6 stores them in its SHA-bound parent plan."""
    if 'config' in record and 'grid' in record:return record['config'],record['grid']
    if bound_plan is None or bound_plan_sha is None or record.get('plan_sha256')!=bound_plan_sha:
        raise ValueError('Missing config/grid requires the verified source parent plan')
    config=record.get('config',bound_plan['config']);grid=record.get('grid',bound_plan['grid'])
    if config!=bound_plan['config'] or grid!=bound_plan['grid']:raise ValueError('Record and bound parent protocol differ')
    return config,grid


def prepare(args):
    p13root=Path(args.a13_dir).resolve();p13=read(p13root/'plan.json')
    if digest(p13root/'plan.json')!=(p13root/'plan.sha256').read_text().strip():raise ValueError('A13 plan changed')
    r13={str(t):precision.existing(p13root,t) for t in (14,54)}
    if any(r is None or r['classification']!='IEEE_POLICY_IMPROVES_METROLOGY_THIS_CASE' for r in r13.values()):raise ValueError('Need declared successful A13 context')
    identities={**p13['identity_files']};checked={}
    def verify(path,sha):
        path=str(path)
        if path not in checked:checked[path]=digest(path)
        if checked[path]!=sha:raise ValueError('Source changed: '+path)
        identities[path]=sha
    for path,sha in list(identities.items()):verify(path,sha)
    for name in ('plan.json','plan.sha256','summary.json'):verify(p13root/name,digest(p13root/name))
    for t in (14,54):
        for name in ('run.json','run.sha256','fields.pt'):verify(p13root/f't{t}'/name,digest(p13root/f't{t}'/name))
    a10path=Path(args.a10_dir).resolve()/'plan.json'
    if str(a10path) not in identities:raise ValueError('A10 source not bound to A13')
    verify(a10path,identities[str(a10path)])
    a10=read(a10path);a7root=Path(a10['source_a7']);p7=read(a7root/'plan.json');r7=read(a7root/'run.json')
    if (digest(a7root/'run.json')!=(a7root/'run.sha256').read_text().strip() or r7['status']!='complete' or
        r7['plan_sha256']!=digest(a7root/'plan.json') or r7['positions']!=[old.analyze_layer(r7['rows'],t) for t in (14,54)]):raise ValueError('A7 labels/source analysis differs')
    for name in ('plan.json','run.json','run.sha256'):verify(a7root/name,digest(a7root/name))
    a6=Path(p7['source']);p6=read(a6/'plan.json');a5=Path(p6['sources']['A5']);p5=read(a5/'plan.json')
    cases=[];sources={}
    for point in [v for v in r7['rows'] if v['source_t']==14]:
        cohort,key=point['cohort'],point['case'];directory=Path(p5['source_cases'][key]) if cohort=='development' else a6/key
        record=read(directory/'run.json');case=dict(record['case'],cohort=cohort)
        protocol_plan=None;protocol_sha=None
        if 'config' not in record or 'grid' not in record:
            protocol_path=directory.parent/'plan.json'
            if str(protocol_path) not in identities:raise ValueError('Source parent plan not bound: '+str(protocol_path))
            protocol_sha=identities[str(protocol_path)];verify(protocol_path,protocol_sha);protocol_plan=read(protocol_path)
        config,grid=source_protocol(record,protocol_plan,protocol_sha)
        if (record['status']!='complete' or not record['gates'] or not all(v['passed'] for v in record['gates'].values()) or
            config!=p13['parent_plans']['14']['config'] or grid!=p13['parent_plans']['14']['grid']):raise ValueError('Incompatible source case: '+key)
        artifacts={name:record['artifacts_sha256'][name] for name in ('pre_last_state.pt','trajectory.pt')}
        verify(directory/'run.json',digest(directory/'run.json'))
        for name,sha in artifacts.items():verify(directory/name,sha)
        verify(record['measurement']['path'],record['measurement']['sha256'])
        observations={str(t):next(v for v in r7['rows'] if v['case']==key and v['source_t']==t) for t in (14,54)}
        for t in (14,54):
            row=next(v for v in record['rows'] if v['t']==t);obs=observations[str(t)]
            if row['residual_noise_ratio']!=obs['residual_noise_ratio'] or row['guidance_rms']!=obs['guidance_rms'] or row['noise_u']!=obs['noise_u']:
                raise ValueError('Cached state/A7 feature mismatch')
        directions={}
        if cohort=='transfer':
            for t in (14,54):
                stage=Path(p13['sources'][str(t)]['stage']);original=read(stage/key/'run.json');path=stage/key/'fields.pt'
                verify(path,original['fields_sha256']);verify(stage/key/'run.json',digest(stage/key/'run.json'))
                directions[str(t)]=dict(path=str(path),sha256=original['fields_sha256'])
        cases.append(case);sources[key]=dict(directory=str(directory),record_sha256=digest(directory/'run.json'),artifacts=artifacts,
            protocol_source='embedded_record' if protocol_plan is None else str(directory.parent/'plan.json'),protocol_plan_sha256=protocol_sha,
            measurement=record['measurement'],observations=observations,directions=directions)
    expected={(i,s) for i in ('00020','00021','00022','00023','00024','00025') for s in (1000,2000)}
    if len(cases)!=12 or {(c['image_id'],c['sampling_seed']) for c in cases}!=expected:raise ValueError('Fixed cohort differs')
    own=[Path(__file__),PROJECTS/'utils/test_diag_ieee_response_value.py',PROJECTS/'run_a14_a15_20261020.sh',
        PROJECTS/'experiment_cards/A14_ieee_clean_response_cohort_20261020.md',PROJECTS/'experiment_cards/A15_ieee_direction_increment_20261020.md']
    worker_hashes={str(path):digest(path) for path in own};identities.update(worker_hashes)
    root=Path(args.root).resolve()
    parents=[p13root,Path(args.a10_dir).resolve(),a7root.resolve(),a6.resolve(),*[Path(s['directory']).resolve() for s in sources.values()]]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in parents):raise ValueError('New root overlaps sources')
    plan=dict(experiment='A14_ieee_response_cohort',cases=cases,case_sources=sources,identity_files=identities,worker_identity_files=worker_hashes,
        source_a13=str(p13root),a13_records=r13,stage_plans=p13['parent_plans'],planned_method_nfe=0,planned_diagnostic_nfe=220,
        positions=[14,54],etas=list(field.ETAS),analysis_models={n:list(v) for n,v in MODELS.items()},ridge_lambda=1,
        analysis_root=str(root.parent/'A15_20261020'))
    if root.exists():
        if read(root/'plan.json')!=plan:raise ValueError('Frozen A14 differs; do not overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
        (root/'experiment_card.md').write_text(own[-2].read_text())
    print('A14/A15 PREPARE PASS: 24 states, two A13 states reused, 220 new diagnostic NFE; fixed transfer analysis follows.',flush=True)


def record_path(root,t,key):return root/f't{t}'/key/'run.json'


def load_record(root,t,case):
    path=record_path(root,t,case['key'])
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted case retained')
        return None
    r=read(path);expected=0 if r.get('reused_A13') else 10
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['case']!=case or r['source_t']!=t or
        r['plan_sha256']!=digest(root/'plan.json') or r['method_nfe']!=0 or r['diagnostic_nfe']!=expected or len(r['calls'])!=expected or
        not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or digest(r['fields_path'])!=r['fields_sha256']):raise ValueError('Failed/interrupted/incompatible A14 retained: '+case['key'])
    return r


def reuse(root,plan,t,case):
    original=plan['a13_records'][str(t)];path=Path(plan['source_a13'])/f't{t}'/'fields.pt'
    if digest(path)!=original['fields_sha256']:raise ValueError('Reused A13 field changed')
    folder=record_path(root,t,case['key']).parent;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='complete',case=case,source_t=t,plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],reused_A13=True,
        response=original['ieee'],policy=original['ieee_policy'],runtime=original['runtime'],fields_path=str(path),fields_sha256=original['fields_sha256'],fields_key='ieee',
        gates=dict(source_A13=dict(passed=True),identity=dict(passed=True)))
    WRITE(folder/'run.json',r);(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')


def run_case(root,plan,t,case,model,device,policy,runtime):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field,perturb_inputs
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    folder=record_path(root,t,case['key']).parent;folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    r=dict(status='running',case=case,source_t=t,plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],reused_A13=False,gates={},policy=policy,runtime=runtime)
    def save():r['seconds']=time.perf_counter()-start;WRITE(folder/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;r['calls'].append(dict(t=int(inputs[1].flatten()[0])));save()
        if r['diagnostic_nfe']>10 or r['calls'][-1]['t']!=t:raise RuntimeError('Call/time budget failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        source=plan['case_sources'][case['key']];src=Path(source['directory']);p=plan['stage_plans'][str(t)]
        if digest(src/'run.json')!=source['record_sha256'] or any(digest(src/n)!=sha for n,sha in source['artifacts'].items()):raise ValueError('Source cache changed')
        if digest(source['measurement']['path'])!=source['measurement']['sha256']:raise ValueError('Observation changed')
        before=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True);cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True)
            k,index=p['source_step'],p['parameter_index']
            if state['tau'].tolist()!=p['grid'] or read(src/'run.json')['rows'][k]['t']!=t:raise ValueError('Grid/time differs')
            for name in ('clean','guidance','latent_inputs'):
                if cache[name].shape!=(30,3,256,256) or cache[name].dtype!=torch.float32 or not torch.isfinite(cache[name][k]).all():raise ValueError('Invalid cache tensor')
            x=cache['latent_inputs'][k:k+1].to(device);q=cache['guidance'][k:k+1].to(device)
            if case['cohort']=='transfer':
                d=source['directions'][str(t)]
                if digest(d['path'])!=d['sha256']:raise ValueError('Saved directions changed')
                saved=torch.load(d['path'],map_location='cpu',weights_only=True);directions={n:a.to(device) for n,a in saved['directions'].items()}
                if relative_error(directions['q'],q)>1e-7:raise ValueError('Original q differs')
            else:
                keys=sorted(c['key'] for c in plan['cases'] if c['cohort']=='development');seed=811020+100*keys.index(case['key'])+t
                orth,meta=field.orthogonal(q.cpu().numpy(),seed);directions={'q':q,'orthogonal':torch.from_numpy(orth).to(device)};r['orthogonal']=meta
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**p['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(source['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            def evaluate(z,y,input_,time_,index_):
                f=evaluate_field(z,y,input_,time_,index_)
                return {n:f[n].cpu() if n in ('clean','q') else f[n] for n in ('clean','q','loss','clip_fraction')}
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling noise')):
                centers,pairs=field.collect(z,y,x,directions,t,index,evaluate,perturb_inputs)
            repeat={n:relative_error(centers[0][n],centers[1][n]) for n in ('clean','q')}
            if any(not math.isfinite(v) or v>1e-7 for v in repeat.values()):raise RuntimeError('IEEE center repeat failed')
            r['native_cache_center_drift']={n:relative_error(cache[{'clean':'clean','q':'guidance'}[n]][k:k+1],centers[0][n]) for n in ('clean','q')}
            payload=dict(centers=centers,pairs=pairs,directions={n:d.cpu() for n,d in directions.items()});r['response']=field.assess(payload)
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Parameters changed')
            tmp=folder/'fields.pt.tmp';torch.save(payload,tmp);tmp.replace(folder/'fields.pt');r.update(fields_path=str(folder/'fields.pt'),fields_key=None,fields_sha256=digest(folder/'fields.pt'))
        if not same_rng(before,rng_state(device)) or r['diagnostic_nfe']!=10 or precision.precision_state(torch)!=policy:raise RuntimeError('RNG/precision/budget failed')
        r['gates']=dict(repeat=dict(passed=True,checks=repeat),parameters_rng_policy=dict(passed=True),budget=dict(passed=True,nfe=10));r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')
        print('A14 POINT:',case['key'],'t=',t,r['response']['readout'],flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def run(args):
    import fcntl,os,torch
    root=Path(args.root).resolve();plan=read(root/'plan.json');t=args.stage
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    with (root/f't{t}.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Stage already running') from None
        todo=[c for c in plan['cases'] if load_record(root,t,c) is None]
        if not todo:return
        for path,sha in plan['worker_identity_files'].items():
            if digest(path)!=sha:raise ValueError('Worker code changed')
        field.coupled.archive.recovery.require_space(root);torch.set_num_threads(2)
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['a13_records'][str(t)]['runtime']:raise ValueError('Runtime differs from reused A13; stop before new calls')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),precision.ieee(torch) as policy:
            if policy!=plan['a13_records'][str(t)]['ieee_policy']:raise ValueError('IEEE policy differs from A13')
            model=None
            for case in todo:
                if case['key']==precision.CASE:reuse(root,plan,t,case)
                else:
                    if model is None:model=field.coupled.archive.recovery.load_model(plan['stage_plans'][str(t)],args.device)
                    run_case(root,plan,t,case,model,args.device,policy,runtime)


def metrology(root):
    plan=read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    all_records=[];stages=[]
    for t in (14,54):
        records=[]
        for case in plan['cases']:
            path=record_path(root,t,case['key'])
            if path.exists():
                r=read(path);all_records.append(r)
                if r['status']=='complete':load_record(root,t,case);records.append(r)
        groups=[]
        for cohort in ('development','transfer'):
            subset=[r for r in records if r['case']['cohort']==cohort];counts={}
            for label in ('GUIDANCE_MORE_SENSITIVE','ORTHOGONAL_MORE_SENSITIVE'):
                counts[label]=sum(len([r for r in subset if r['case']['image_id']==i])==2 and all(r['response']['readout']==label for r in subset if r['case']['image_id']==i) for i in {r['case']['image_id'] for r in subset})
            classification='INCOMPLETE_PILOT' if len(subset)!=6 else 'GUIDANCE_ALIGNED_RESPONSE_CANDIDATE' if counts['GUIDANCE_MORE_SENSITIVE']>=2 else 'ORTHOGONAL_RESPONSE_LARGER_CANDIDATE' if counts['ORTHOGONAL_MORE_SENSITIVE']>=2 else 'NO_COHERENT_DIRECTION_SPECIFICITY_ESTABLISHED'
            groups.append(dict(cohort=cohort,classification=classification,completed_cases=len(subset),usable_cases=sum(r['response']['usable'] for r in subset),image_counts_both_seeds=counts))
        stages.append(dict(source_t=t,completed_cases=len(records),groups=groups,
            unresolved=[r['case']['key'] for r in records if not r['response']['usable']],
            max_scale_spread={n:max((r['response']['directions'][n]['relative_vector_spread'] for r in records if r['response']['directions'][n]['relative_vector_spread'] is not None),default=None) for n in ('q','orthogonal')}))
    nfe=sum(r['diagnostic_nfe'] for r in all_records)
    return dict(completed_points=sum(s['completed_cases'] for s in stages),planned_points=24,reused_points=sum(r.get('reused_A13',False) for r in all_records),
        method_nfe=0,diagnostic_nfe=nfe,stages=stages,cumulative_queue_cost=dict(method_nfe=3600,diagnostic_nfe=2044+nfe,total=5644+nfe),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in all_records),
        limits='Six reused images; IEEE diagnostics on native production states. Readability/direction specificity only, no new reconstruction benefit.')


def feature_row(observation,response):
    gains={}
    for n,d in response['directions'].items():
        values=[p['gain'] for p in d['pairs']]
        gains[n]=statistics.mean(values) if all(v is not None and math.isfinite(v) for v in values) else None
    usable=response['usable'] and all(v is not None and math.isfinite(v) and v>0 for v in gains.values())
    return dict(observation,usable=bool(usable),q_gain=gains['q'],orthogonal_gain=gains['orthogonal'],
        log_orthogonal_gain=math.log(gains['orthogonal']) if usable else None,log_gain_ratio=math.log(gains['q']/gains['orthogonal']) if usable else None)


def analyze_layer(rows,t):
    subset=[r for r in rows if r['source_t']==t];train=[r for r in subset if r['cohort']=='development'];test=[r for r in subset if r['cohort']=='transfer']
    for values,ids in ((train,('00020','00021','00022')),(test,('00023','00024','00025'))):
        if len(values)!=6 or {(r['image_id'],r['sampling_seed']) for r in values}!={(i,s) for i in ids for s in (1000,2000)}:raise ValueError('Frozen split differs')
    if not all(r['usable'] for r in subset):return dict(source_t=t,classification='UNRESOLVED_METROLOGY_NO_FIT',unresolved=[r['case'] for r in subset if not r['usable']],targets={})
    targets={}
    for target in old.TARGETS:
        models={}
        for name,columns in MODELS.items():
            values,fit=old.ridge_predict(train,test,columns,target);models[name]=dict(**old.errors(test,values,target),predictions=values,fit=fit)
        values=[statistics.mean(r[target] for r in train)]*len(test);models['stage_mean']=dict(**old.errors(test,values,target),predictions=values)
        d=models['direction'];comparisons={}
        for name in ('physical','generic'):
            b=models[name];ratio=1-d['mae']/b['mae'] if b['mae']>1e-12 else None
            wins=sum(d['per_image_mae'][i]<b['per_image_mae'][i]-1e-12 for i in d['per_image_mae'])
            comparisons[name]=dict(relative_mae_reduction=ratio,images_better=wins,threshold_passed=ratio is not None and ratio>=.1 and wins>=2)
        targets[target]=dict(models=models,comparisons=comparisons,
            engineering_threshold_passed=all(c['threshold_passed'] for c in comparisons.values()) and d['mae']<=models['stage_mean']['mae']+1e-12,
            actual=[dict(case=r['case'],image_id=r['image_id'],value=r[target]) for r in test])
    nonworse=all(targets[k]['models']['direction']['mae']<=min(targets[k]['models'][m]['mae'] for m in ('physical','generic','stage_mean'))+1e-12 for k in old.QUALITY)
    hits=sum(targets[k]['engineering_threshold_passed'] for k in old.QUALITY)
    return dict(source_t=t,classification='IEEE_DIRECTION_INCREMENT_EXPLORATORY_CANDIDATE_ONLY' if nonworse and hits>=2 else 'IEEE_DIRECTION_INCREMENT_NOT_SUPPORTED',
        all_quality_mae_nonworse=nonworse,quality_targets_meeting_threshold=hits,targets=targets)


def analyze(root):
    plan=read(root/'plan.json');rows=[];identities={str(root/'plan.json'):digest(root/'plan.json')}
    for t in (14,54):
        for case in plan['cases']:
            r=load_record(root,t,case)
            if r is None:return None
            rows.append(feature_row(plan['case_sources'][case['key']]['observations'][str(t)],r['response']))
            path=record_path(root,t,case['key']);identities[str(path)]=digest(path);identities[r['fields_path']]=r['fields_sha256']
    target=Path(plan['analysis_root']);config=dict(experiment='A15_ieee_direction_increment',source=str(root),identity_files=identities,
        models={n:list(v) for n,v in MODELS.items()},lambda_=1,method_nfe=0,diagnostic_nfe=0)
    if target.exists():
        if read(target/'plan.json')!=config:raise ValueError('Frozen A15 differs')
    else:
        target.mkdir(parents=True,exist_ok=False);WRITE(target/'plan.json',config)
        (target/'experiment_card.md').write_text((PROJECTS/'experiment_cards/A15_ieee_direction_increment_20261020.md').read_text())
    result=dict(status='complete',plan_sha256=digest(target/'plan.json'),method_nfe=0,diagnostic_nfe=0,rows=rows,positions=[analyze_layer(rows,t) for t in (14,54)])
    if (target/'run.json').exists():
        if digest(target/'run.json')!=(target/'run.sha256').read_text().strip() or read(target/'run.json')!=result:raise ValueError('Existing A15 differs; retained')
    else:
        with (target/'points.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        WRITE(target/'run.json',result);(target/'run.sha256').write_text(digest(target/'run.json')+'\n')
    compact=dict(method_nfe=0,diagnostic_nfe=0,positions=[{**{k:v for k,v in p.items() if k!='targets'},'targets':{k:dict(mae={n:m['mae'] for n,m in v['models'].items()},comparisons=v['comparisons'],engineering_threshold_passed=v['engineering_threshold_passed']) for k,v in p['targets'].items()}} for p in result['positions']],
        progress_percent=45,limits='Exploratory reused-cohort transfer; each response currently costs 10 diagnostic evaluations. No prospective, equal-cost control or reconstruction claim.')
    WRITE(target/'summary.json',compact);return target,compact


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=BASE+'/A14_20261020')
    parser.add_argument('--a13-dir',default=precision.ROOT);parser.add_argument('--a10-dir',default='/data0/lzy/ZAPS-results/diag_clean_roughness/A10_20261020')
    parser.add_argument('--stage',type=int,choices=(14,54));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--summary',action='store_true');parser.add_argument('--archive',action='store_true');args=parser.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args)
    elif args.summary or args.archive:
        for path,sha in read(root/'plan.json')['worker_identity_files'].items():
            if digest(path)!=sha:raise ValueError('Analysis code/card changed')
        out=metrology(root);WRITE(root/'summary.json',out);print('A14 Summary:',json.dumps(out),flush=True)
        roots={'A14':str(root)}
        if out['completed_points']==24:
            result=analyze(root)
            if result is not None:
                target,report=result;print('A15 Summary:',json.dumps(report),flush=True);roots['A15']=str(target)
        else:print('A15 NOT_RUN: A14 incomplete; all partial costs retained.',flush=True)
        if args.archive:
            from utils.archive_refinement_confirmation import export_and_commit
            bundle,commit=export_and_commit(roots);print('Archive:',bundle,'Results commit:',commit,flush=True)
    elif args.stage is not None:run(args)
    else:parser.error('Choose --prepare, --stage, --summary or --archive')


if __name__=='__main__':main()
