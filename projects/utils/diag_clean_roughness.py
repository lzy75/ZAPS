"""A10: fixed clean-field roughness transfer screening; CPU only, zero UNet calls."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_direction_value as old
from utils import diag_terminal_polish_control as terminal
from utils.diag_state_trajectory import digest
WRITE=old.WRITE
CARD=PROJECTS/'experiment_cards/A10_clean_roughness_20261020.md'
MODELS={'physical':old.BASE,'speed':old.BASE+('log1p_speed',),'roughness':old.BASE+('log1p_roughness',)}


def geometry(clean,rows,k=28):
    """Only these three existing states enter; no future state or outcome reads."""
    if k!=28 or len(rows)!=30 or [rows[j]['t'] for j in (26,27,28)]!=[124,54,14]:
        raise ValueError('Need fixed original t124/t54/t14 sequence')
    values=[np.asarray(clean[j],dtype=np.float64) for j in (k-2,k-1,k)]
    if any(v.shape!=values[0].shape or not np.isfinite(v).all() for v in values):raise ValueError('Invalid clean slices')
    u=[float(rows[j]['noise_u']) for j in (k-2,k-1,k)]
    h0,h1=np.diff(u)
    if not all(math.isfinite(x) and x>0 for x in (h0,h1)):raise ValueError('Invalid u spans')
    v0=(values[1]-values[0])/h0;v1=(values[2]-values[1])/h1
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    speed=rms(v1);roughness=rms(v1-v0)/(.5*(h0+h1))
    if not all(math.isfinite(x) for x in (speed,roughness)):raise ValueError('Nonfinite clean geometry')
    return dict(clean_speed=speed,clean_roughness=roughness,log1p_speed=math.log1p(speed),
                log1p_roughness=math.log1p(roughness),h_previous=float(h0),h_current=float(h1))


def load_geometry(path,rows):
    import torch
    cache=torch.load(path,map_location='cpu',weights_only=True)
    clean=cache['clean']
    if tuple(clean.shape)!=(30,3,256,256) or clean.dtype!=torch.float32 or clean.device.type!='cpu':
        raise ValueError('Expected CPU float32 clean[30,3,256,256]')
    return geometry(clean,rows)


def analyze(rows):
    train=[r for r in rows if r['cohort']=='development'];test=[r for r in rows if r['cohort']=='transfer']
    for subset,ids in ((train,('00020','00021','00022')),(test,('00023','00024','00025'))):
        if len(subset)!=6 or {(r['image_id'],r['sampling_seed']) for r in subset}!={(i,s) for i in ids for s in (1000,2000)}:
            raise ValueError('Frozen image/seed split differs')
    if any(r['source_t']!=14 for r in rows):raise ValueError('Only t14 is prespecified')
    if not all(r['usable'] for r in rows):return dict(classification='UNRESOLVED_INPUTS_NO_FIT',targets={})
    targets={}
    for target in old.TARGETS:
        models={}
        for name,columns in MODELS.items():
            values,fit=old.ridge_predict(train,test,columns,target)
            models[name]=dict(**old.errors(test,values,target),predictions=values,fit=fit)
        mean=[statistics.mean(r[target] for r in train)]*len(test)
        models['stage_mean']=dict(**old.errors(test,mean,target),predictions=mean)
        a=models['roughness'];comparisons={}
        for name in ('physical','speed'):
            b=models[name];ratio=1-a['mae']/b['mae'] if b['mae']>1e-12 else None
            wins=sum(a['per_image_mae'][i]<b['per_image_mae'][i]-1e-12 for i in a['per_image_mae'])
            comparisons[name]=dict(relative_mae_reduction=ratio,images_better=wins,
                threshold_passed=ratio is not None and ratio>=.1 and wins>=2)
        qualifies=all(c['threshold_passed'] for c in comparisons.values()) and a['mae']<=models['stage_mean']['mae']+1e-12
        targets[target]=dict(models=models,comparisons=comparisons,engineering_threshold_passed=qualifies,
            actual=[dict(case=r['case'],image_id=r['image_id'],value=r[target]) for r in test])
    nonworse=all(targets[t]['models']['roughness']['mae']<=min(targets[t]['models'][m]['mae'] for m in ('physical','speed','stage_mean'))+1e-12 for t in old.QUALITY)
    hits=sum(targets[t]['engineering_threshold_passed'] for t in old.QUALITY)
    return dict(classification='CLEAN_ROUGHNESS_EXPLORATORY_CANDIDATE_ONLY' if nonworse and hits>=2 else 'CLEAN_ROUGHNESS_INCREMENT_NOT_SUPPORTED',
                all_quality_mae_nonworse=nonworse,quality_targets_meeting_threshold=hits,targets=targets)


def read(path):return json.loads(Path(path).read_text())


def bind(args):
    # Audit existing identities once, without recursively preparing earlier runs.
    root7=Path(args.a7_dir).resolve()
    if not(root7/'plan.json').is_file():raise ValueError('Need completed A7, no parent experiments started')
    plan7=read(root7/'plan.json');r7=read(root7/'run.json');source_rows=r7['rows']
    if (digest(root7/'run.json')!=(root7/'run.sha256').read_text().strip() or r7['status']!='complete' or
        r7['method_nfe']!=0 or r7['diagnostic_nfe']!=0 or r7['plan_sha256']!=digest(root7/'plan.json') or
        r7['positions']!=[old.analyze_layer(source_rows,t) for t in (14,54)]):raise ValueError('A7 frozen analysis differs')
    root9=Path(args.a9_dir).resolve();plan9=read(root9/'plan.json')
    active9=terminal.completed(root9,plan9);summary9=terminal.stored_summary(root9)
    if len(active9)!=6 or summary9!=read(root9/'summary.json') or summary9['classification']!='NO_ROBUST_MIDPOINT_ADVANTAGE_ESTABLISHED':
        raise ValueError('Need complete A9 with declared context; no rerun')
    identities={**plan7['identity_files'],**plan9['identity_files']}
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Parent identity changed: '+path)
    paths=[root7/n for n in ('plan.json','run.json','run.sha256')]+[root9/n for n in ('plan.json','summary.json')]
    for key in active9:paths.extend(root9/key/n for n in ('run.json','run.sha256','outputs.pt'))
    a6root=Path(args.a6_dir).resolve();a6plan=read(a6root/'plan.json')
    a5root=Path(args.a5_dir).resolve();a5plan=read(a5root/'plan.json');rows=[]
    if Path(plan7['source']).resolve()!=a6root or Path(a6plan['sources']['A5']).resolve()!=a5root or Path(plan9['a6_source']).resolve()!=a6root:
        raise ValueError('Parent directory identities differ')
    records={}
    for source in source_rows:
        directory=Path(a5plan['source_cases'][source['case']]) if source['cohort']=='development' else a6root/source['case']
        if str(directory) not in records:records[str(directory)]=read(directory/'run.json')
        record=records[str(directory)]
        if record['status']!='complete' or not record['gates'] or not all(g['passed'] for g in record['gates'].values()):
            raise ValueError('Unverified cached trajectory: '+str(directory))
        t=source['source_t'];own=next(v for v in record['rows'] if v['t']==t)
        if source['cohort']=='development':
            label='A4' if t==14 else 'A5';suffix=read(Path(a6plan['sources'][label])/source['case']/'run.json')
            baseline=suffix['branches']['C'];branches=suffix['branches']
        else:baseline=record['baseline'];branches=record['positions'][str(t)]['branches']
        if old.observation(source['cohort'],record['case'],own,baseline,branches)!=source:raise ValueError('A7 source observation differs')
        if t!=14:continue
        cache=directory/'trajectory.pt'
        if digest(cache)!=record['artifacts_sha256']['trajectory.pt']:raise ValueError('Cached trajectory SHA differs')
        rows.append(dict(source,**load_geometry(cache,record['rows'])))
        paths.extend((directory/'run.json',cache))
    paths.extend((CARD,Path(__file__),PROJECTS/'run_a10_20261020.sh',PROJECTS/'utils/test_diag_clean_roughness.py',Path(args.a5_dir)/'plan.json'))
    identities.update({str(p):digest(p) for p in paths})
    plan=dict(experiment='clean_roughness_plan',identity_files=identities,source_t=14,past_t=[124,54],
        train_images=['00020','00021','00022'],transfer_images=['00023','00024','00025'],models={k:list(v) for k,v in MODELS.items()},
        ridge_lambda=1,method_nfe=0,diagnostic_nfe=0,source_a7=str(root7),source_a9=str(root9))
    directory=Path(args.run_dir).resolve()
    sources=[root7,root9,Path(args.a6_dir).resolve(),*[Path(v).resolve() for v in a5plan['source_cases'].values()]]
    if any(directory.is_relative_to(p) or p.is_relative_to(directory) for p in sources):raise ValueError('New directory overlaps sources')
    if directory.exists():
        if read(directory/'plan.json')!=plan:raise ValueError('Frozen A10 differs; do not overwrite')
    else:
        directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
        (directory/'experiment_card.md').write_text(CARD.read_text())
    return directory,plan,rows


def compact(record):
    analysis=record['analysis']
    return dict(method_nfe=0,diagnostic_nfe=0,status=record['status'],case_position_points=12,image_clusters=6,source_t=14,
        **{k:v for k,v in analysis.items() if k!='targets'},targets={t:dict(mae={m:x['mae'] for m,x in r['models'].items()},
            comparisons=r['comparisons'],engineering_threshold_passed=r['engineering_threshold_passed']) for t,r in analysis['targets'].items()},
        progress_percent=45,cumulative_queue_cost=dict(method_nfe=3600,diagnostic_nfe=1884,total=5484),
        limits='Previously observed outcomes, three training and three transfer image clusters; stochastic-path finite differences, not fixed-time sensitivity or proven curvature. No online action or control benefit.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    for label,folder in (('a2c','diag_one_step_guidance_scale/A2c_20261019'),('a2d','diag_direction_refresh/A2d_20261019'),('a3','diag_coupled_refinement/A3_20261020'),('a4','diag_terminal_continuation/A4_20261020'),('a5','diag_fixed_position_continuation/A5_20261020'),('a6','diag_fresh_refinement_confirmation/A6_20261020'),('a7','diag_direction_value/A7_20261020'),('a9','diag_terminal_polish_control/A9_20261020')):
        p.add_argument('--'+label+'-dir',default='/data0/lzy/ZAPS-results/'+folder)
    p.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_clean_roughness/A10_20261020')
    p.add_argument('--summary-only',action='store_true');args=p.parse_args();directory=Path(args.run_dir).resolve()
    if args.summary_only:
        r=read(directory/'run.json')
        if digest(directory/'run.json')!=(directory/'run.sha256').read_text().strip() or r['status']!='complete':raise ValueError('A10 not complete')
        print('A10 Summary:',json.dumps(compact(r)), '\nRecords:',directory);return
    import fcntl
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A10 already running') from None
        directory,plan,rows=bind(args);start=time.perf_counter();result=analyze(rows)
        if (directory/'run.json').exists():
            saved=read(directory/'run.json')
            if (digest(directory/'run.json')!=(directory/'run.sha256').read_text().strip() or saved['status']!='complete' or saved['rows']!=rows or
                saved['analysis']!=result or saved['plan_sha256']!=digest(directory/'plan.json')):raise ValueError('Existing A10 differs; retained')
        else:
            saved=dict(status='running',plan_sha256=digest(directory/'plan.json'),method_nfe=0,diagnostic_nfe=0)
            WRITE(directory/'run.json',saved)
            try:
                with (directory/'points.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                if any(digest(path)!=sha for path,sha in plan['identity_files'].items()):raise ValueError('Source changed during CPU analysis')
                saved.update(status='complete',rows=rows,analysis=result,seconds=time.perf_counter()-start,gates=dict(source_identity=True,frozen_split=True,no_model_or_cuda=True))
                WRITE(directory/'run.json',saved);(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
            except BaseException as error:
                saved.update(status='failed',error=repr(error));WRITE(directory/'run.json',saved);raise
        WRITE(directory/'summary.json',compact(saved))
        print('A10 Summary:',json.dumps(compact(saved)),'\nRecords:',directory,flush=True)
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A10':str(directory)})
        print('Archive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
