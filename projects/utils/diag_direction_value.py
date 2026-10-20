"""A7: frozen split, phase-conditioned predictive direction-value diagnostic; no new UNet calls."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import numpy as np
PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_fresh_refinement_confirmation as fresh
from utils import diag_state_trajectory as state
from utils.diag_state_trajectory import digest
WRITE=fresh.WRITE
CARD=PROJECTS/'experiment_cards/A7_direction_value_diagnostic_20261020.md'
BASE=('log_R','log_a');DIRECTION=BASE+('turn','log1p_omega')
TARGETS=('gain_psnr','gain_ssim','gain_lpips','gain_loss')
QUALITY=TARGETS[:3]


def features(row):
    R,a,c,omega=(row.get(k) for k in ('residual_noise_ratio','guidance_rms','guidance_cosine','guidance_variation3_per_u'))
    if any(v is None or not math.isfinite(v) for v in (R,a,c,omega)) or R<=0 or a<=0 or not -1<=c<=1 or omega<0:
        return dict(usable=False,reason='UNRESOLVED_OR_INVALID_SOURCE_DIRECTION',log_R=None,log_a=None,turn=None,log1p_omega=None)
    return dict(usable=True,reason='',log_R=math.log(R),log_a=math.log(a),turn=math.sqrt(max(0.,2*(1-c))),log1p_omega=math.log1p(omega))


def observation(cohort,case,row,baseline,branches):
    result=dict(cohort=cohort,case=case['key'],image_id=case['image_id'],sampling_seed=case['sampling_seed'],source_t=row['t'],
                noise_u=row['noise_u'],residual_noise_ratio=row.get('residual_noise_ratio'),guidance_rms=row.get('guidance_rms'),
                guidance_cosine=row.get('guidance_cosine'),omega3=row.get('guidance_variation3_per_u'),**features(row))
    for metric in ('psnr','ssim','lpips','loss'):
        gains=[(branches[a][metric]-baseline[metric])*(1 if metric in ('psnr','ssim') else -1) for a in ('F0','F1')]
        if not all(math.isfinite(v) for v in gains):raise ValueError('Nonfinite endpoint gain')
        result['gain_'+metric]=statistics.mean(gains)
        result['bridge_gap_'+metric]=abs(gains[0]-gains[1])
    return result


def ridge_predict(train,test,columns,target):
    x=np.asarray([[r[c] for c in columns] for r in train],dtype=np.float64)
    xt=np.asarray([[r[c] for c in columns] for r in test],dtype=np.float64)
    y=np.asarray([r[target] for r in train],dtype=np.float64)
    center=x.mean(axis=0);raw=x.std(axis=0);scale=np.where(raw>1e-12,raw,1.)
    z=(x-center)/scale;ym=float(y.mean())
    w=np.linalg.solve(z.T@z+np.eye(len(columns)),z.T@(y-ym))
    values=ym+((xt-center)/scale)@w
    if not np.isfinite(values).all():raise ValueError('Nonfinite frozen ridge prediction')
    return values.tolist(),dict(columns=list(columns),lambda_=1.,x_mean=center.tolist(),x_scale=scale.tolist(),constant_columns=[c for c,s in zip(columns,raw) if s<=1e-12],y_mean=ym,weights=w.tolist())


def errors(test,predictions,target):
    per_image={image:statistics.mean(abs(predictions[k]-r[target]) for k,r in enumerate(test) if r['image_id']==image)
               for image in sorted({r['image_id'] for r in test})}
    return dict(mae=statistics.mean(per_image.values()),per_image_mae=per_image)


def cv(train,columns,target):
    values=[]
    for image in sorted({r['image_id'] for r in train}):
        keep=[r for r in train if r['image_id']!=image];held=[r for r in train if r['image_id']==image]
        predicted,_=ridge_predict(keep,held,columns,target)
        values.append(dict(image_id=image,mae=errors(held,predicted,target)['mae']))
    return dict(group='leave-one-development-image-out',folds=values,mean_mae=statistics.mean(v['mae'] for v in values))


def analyze_layer(rows,t):
    subset=[r for r in rows if r['source_t']==t];train=[r for r in subset if r['cohort']=='development'];test=[r for r in subset if r['cohort']=='transfer']
    identity=lambda values:{(r['image_id'],r['sampling_seed']) for r in values}
    expected_dev={(i,s) for i in ('00020','00021','00022') for s in (1000,2000)}
    expected_test={(i,s) for i in ('00023','00024','00025') for s in (1000,2000)}
    if len(train)!=6 or len(test)!=6 or identity(train)!=expected_dev or identity(test)!=expected_test:raise ValueError('Fixed split differs; never replace cases')
    if not all(r['usable'] for r in subset):return dict(source_t=t,classification='UNRESOLVED_INPUTS_NO_FIT',unresolved=[r['case'] for r in subset if not r['usable']])
    results={}
    for target in TARGETS:
        b,bfit=ridge_predict(train,test,BASE,target);d,dfit=ridge_predict(train,test,DIRECTION,target)
        mean=[statistics.mean(r[target] for r in train)]*len(test)
        B,D,M=(errors(test,v,target) for v in (b,d,mean))
        wins=sum(D['per_image_mae'][i]<B['per_image_mae'][i]-1e-12 for i in D['per_image_mae'])
        ratio=(1-D['mae']/B['mae']) if B['mae']>1e-12 else None
        qualifies=ratio is not None and ratio>=.1 and wins>=2 and D['mae']<=M['mae']+1e-12
        results[target]=dict(baseline=B,direction=D,stage_mean=M,relative_mae_reduction=ratio,images_direction_better=wins,engineering_threshold_passed=bool(qualifies),
            predictions=[dict(case=r['case'],image_id=r['image_id'],actual=r[target],baseline=bp,direction=dp,stage_mean=mp) for r,bp,dp,mp in zip(test,b,d,mean)],
            baseline_fit=bfit,direction_fit=dfit,development_cv=dict(baseline=cv(train,BASE,target),direction=cv(train,DIRECTION,target)))
    nonworse=all(results[k]['direction']['mae']<=min(results[k]['baseline']['mae'],results[k]['stage_mean']['mae'])+1e-12 for k in QUALITY)
    support=nonworse and sum(results[k]['engineering_threshold_passed'] for k in QUALITY)>=2
    return dict(source_t=t,training_images=3,transfer_images=3,classification='INCREMENTAL_DIRECTION_TRANSFER_CANDIDATE_ONLY' if support else 'DIRECTION_INCREMENT_NOT_SUPPORTED_IN_THIS_TRANSFER_TEST',
                all_quality_mae_nonworse=nonworse,quality_targets_meeting_threshold=sum(results[k]['engineering_threshold_passed'] for k in QUALITY),targets=results)


def associations(rows):
    output=[]
    for t in (14,54):
        subset=[r for r in rows if r['source_t']==t and r['usable']]
        for scope in ('development','transfer','pooled_images'):
            cases=[r for r in subset if scope=='pooled_images' or r['cohort']==scope]
            image_rows=[]
            for image in sorted({r['image_id'] for r in cases}):
                rr=[r for r in cases if r['image_id']==image]
                image_rows.append({k:statistics.mean(r[k] for r in rr) for k in (*DIRECTION,*TARGETS)})
            for predictor in DIRECTION:
                for target in TARGETS:
                    rho=state.pearson(state.ranks([r[predictor] for r in image_rows]),state.ranks([r[target] for r in image_rows]))
                    case_rho=state.pearson(state.ranks([r[predictor] for r in cases]),state.ranks([r[target] for r in cases])) if scope!='pooled_images' else None
                    output.append(dict(source_t=t,scope=scope,predictor=predictor,target=target,image_clusters=len(image_rows),image_spearman=rho,
                                       dependent_case_spearman=case_rho,limits='Descriptive only; within-cohort n=3 image correlation unavailable; pooled cohorts may differ; seeds share observations.'))
    return output


def progress():
    items=[('基座协议与可回放',10,10),('状态指标定义及可测性',15,15),('细化动作终局与新图小样本确认',20,20),
           ('方向指标的增量预测证据',20,0),('等成本在线状态控制收益',20,0),('较大确认集、消融与论文验证',15,0)]
    return dict(percent=sum(r[2] for r in items),rubric=[dict(milestone=n,weight=w,earned=e) for n,w,e in items],
                limits='Transparent engineering milestone estimate, not probability of scientific success, elapsed-time estimate or proven innovation percentage. No new credit granted automatically by A7.')


def bind(args):
    source,plan=fresh.prepare(SimpleNamespace(source_plan_dir=args.source_plan_dir,a2c_dir=args.a2c_dir,a2d_dir=args.a2d_dir,
        a3_dir=args.a3_dir,a4_dir=args.a4_dir,a5_dir=args.a5_dir,run_dir=args.a6_dir,data_dir=None,inspect=True))
    active=fresh.completed(source,plan)
    if len(active)!=6 or fresh.stored_summary(source)!=json.loads((source/'summary.json').read_text()):raise ValueError('Need six complete A6 cases and exact summary')
    a5plan=json.loads((Path(plan['sources']['A5'])/'plan.json').read_text())
    files={**plan['identity_files']};paths=[source/'plan.json',source/'summary.json',CARD,Path(__file__),PROJECTS/'run_a7_20261020.sh',PROJECTS/'utils/test_diag_direction_value.py']
    rows=[]
    for key,path in sorted(a5plan['source_cases'].items()):
        p=Path(path)/'run.json';old=json.loads(p.read_text());paths.append(p)
        for t,label in ((14,'A4'),(54,'A5')):
            rp=Path(plan['sources'][label])/key/'run.json';r=json.loads(rp.read_text());paths.append(rp)
            row=next(v for v in old['rows'] if v['t']==t)
            rows.append(observation('development',old['case'],row,r['branches']['C'],{a:r['branches'][a] for a in ('F0','F1')}))
    for key,r in sorted(active.items()):
        paths.extend([source/key/'run.json',source/key/'run.sha256'])
        if digest(r['measurement']['path'])!=r['measurement']['sha256']:raise ValueError('A6 observation SHA differs')
        for t in (14,54):
            row=next(v for v in r['rows'] if v['t']==t)
            rows.append(observation('transfer',r['case'],row,r['baseline'],r['positions'][str(t)]['branches']))
    files.update({str(p):digest(p) for p in paths})
    config=dict(experiment='direction_value_plan',source=str(source),identity_files=files,
        train_images=['00020','00021','00022'],transfer_images=['00023','00024','00025'],positions=[14,54],
        ridge_lambda=1,baseline_features=list(BASE),direction_features=list(DIRECTION),targets=list(TARGETS),new_method_nfe=0,new_diagnostic_nfe=0,
        inference_scope='Exploratory transfer diagnostic; A6 output summary was previously seen. No prospective validation or automatic online policy.')
    directory=Path(args.run_dir).resolve()
    for original in (source,*[Path(p) for p in plan['sources'].values()]):
        if directory.is_relative_to(original) or original.is_relative_to(directory):raise ValueError('Analysis must be separate from originals')
    if directory.exists():
        if json.loads((directory/'plan.json').read_text())!=config:raise ValueError('Frozen A7 plan differs; do not overwrite')
    else:
        directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',config)
        (directory/'experiment_card.md').write_text(CARD.read_text())
    return directory,config,rows


def compact(result):
    out=[]
    for layer in result['positions']:
        if 'targets' not in layer:out.append(layer);continue
        out.append({**{k:v for k,v in layer.items() if k!='targets'},'targets':{k:{'baseline_MAE':v['baseline']['mae'],'direction_MAE':v['direction']['mae'],
            'stage_mean_MAE':v['stage_mean']['mae'],'relative_MAE_reduction':v['relative_mae_reduction'],'images_direction_better':v['images_direction_better'],
            'engineering_threshold_passed':v['engineering_threshold_passed']} for k,v in layer['targets'].items()}})
    return dict(method_nfe=0,diagnostic_nfe=0,case_position_points=24,image_clusters=6,positions=out,progress=progress(),
        cumulative_queue_cost=dict(method_nfe=3600,diagnostic_nfe=1812,total=5412,scope='A2a including recovery through A6; earlier single-image work excluded'),
        limits=result['limits'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    for label,folder in (('a2c','diag_one_step_guidance_scale/A2c_20261019'),('a2d','diag_direction_refresh/A2d_20261019'),('a3','diag_coupled_refinement/A3_20261020'),('a4','diag_terminal_continuation/A4_20261020'),('a5','diag_fixed_position_continuation/A5_20261020'),('a6','diag_fresh_refinement_confirmation/A6_20261020')):
        parser.add_argument('--'+label+'-dir',default='/data0/lzy/ZAPS-results/'+folder)
    parser.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_direction_value/A7_20261020')
    args=parser.parse_args();directory=Path(args.run_dir).resolve();directory.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A7 already running') from None
        directory,plan,rows=bind(args)
        if (directory/'run.json').exists():
            if digest(directory/'run.json')!=(directory/'run.sha256').read_text().strip():raise ValueError('A7 result SHA differs')
            saved=json.loads((directory/'run.json').read_text())
            if saved.get('status')!='complete' or saved.get('method_nfe')!=0 or saved.get('diagnostic_nfe')!=0:raise ValueError('Incomplete A7 result retained')
        else:
            saved=dict(experiment='direction_value_diagnostic',status='complete',plan_sha256=digest(directory/'plan.json'),method_nfe=0,diagnostic_nfe=0,
                rows=rows,positions=[analyze_layer(rows,t) for t in (14,54)],associations=associations(rows),
                limits=plan['inference_scope']+' Three training and three transfer image clusters; fixed regularization; no p-values/causal/control/generalization claims.')
            with (directory/'points.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            if any(digest(p)!=sha for p,sha in plan['identity_files'].items()):raise ValueError('Frozen source changed during A7')
            WRITE(directory/'run.json',saved);(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        if saved['plan_sha256']!=digest(directory/'plan.json') or saved['rows']!=rows:raise ValueError('A7 source rows/plan differ')
        report=compact(saved);WRITE(directory/'summary.json',report)
        print('A7 Summary:',json.dumps(report,ensure_ascii=False),'\nRecords:',directory,flush=True)
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A7':str(directory)})
        print('Archive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
