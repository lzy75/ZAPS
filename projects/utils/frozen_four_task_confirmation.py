"""A46 calibration lock, GT-free reassignment and separate quality/matching readouts."""
import numpy as np
from utils import continuous_uncertainty_feedback as control

WEIGHTS={'super_resolution':.5,'gaussian_deblur':0.,'inpainting':1.,'motion_deblur':0.}
MATCHED={'super_resolution':152.699,'gaussian_deblur':9.178,'inpainting':30.209,'motion_deblur':9.61}
POLICIES=('C','TASK_FIXED','MATCHED_FIXED','PHYSICAL','ISO','STATE','SELECTED','SHUFFLE')
METRICS=('psnr','ssim','lpips','loss')


def cap(phase,task):
    if task not in WEIGHTS:raise ValueError('Unknown task')
    return {'BASELINE':300,'choose':56 if WEIGHTS[task]==.5 else 53,
            'finish':35 if WEIGHTS[task]==.5 else 30}[phase]


def allocation(cases,choices,fixed,matched):
    if len(cases)!=24 or len({c['image_id'] for c in cases})!=6 or set(choices)!={c['key'] for c in cases}:
        raise ValueError('Need all24 frozen choices, six shared images')
    if set(fixed)!=set(WEIGHTS) or set(matched)!=set(WEIGHTS):raise ValueError('Reference tasks differ')
    if any(abs(matched[t]-MATCHED[t])>1e-9 for t in WEIGHTS):raise ValueError('Calibration median differs')
    images=sorted({c['image_id'] for c in cases});result={}
    for task in WEIGHTS:
        group=sorted([c for c in cases if c['task']==task],key=lambda c:c['image_id'])
        if [c['image_id'] for c in group]!=images:raise ValueError('Task/image mapping differs')
        for i,c in enumerate(group):
            own=choices[c['key']];donor=group[(i+1)%6]
            if set(own)!= {'PHYSICAL','ISO','STATE','SELECTED'} or any(not np.isfinite(g) or g<0 for g in own.values()):
                raise ValueError('Invalid label-free multiplier')
            if WEIGHTS[task] in (0.,1.) and own['SELECTED']!=own['PHYSICAL' if WEIGHTS[task]==0. else 'STATE']:
                raise ValueError('Frozen endpoint alias differs')
            result[c['key']]=dict(gamma=dict(C=1.,TASK_FIXED=fixed[task],MATCHED_FIXED=matched[task],**own,
                SHUFFLE=choices[donor['key']]['SELECTED']),donor=donor['key'])
    return result


def summarize(records):
    groups=[]
    for task in WEIGHTS:
        rows=[r for r in records if r['case']['task']==task];comparisons={}
        for name in POLICIES:
            if name=='SELECTED':continue
            values=[{m:r['evaluation']['SELECTED']['delta'][m]-r['evaluation'][name]['delta'][m] for m in METRICS} for r in rows]
            mean={m:float(np.mean([v[m] for v in values])) if rows else None for m in METRICS}
            wins=sum(v['psnr']>1e-9 for v in values)
            comparisons[name]=dict(mean=mean,images_PSNR_positive=wins)
        complete=len(rows)==6;c,f,s=(comparisons[n] for n in ('C','MATCHED_FIXED','SHUFFLE'))
        quality=complete and c['mean']['psnr']>=.05 and c['images_PSNR_positive']>=5 and control.quality_guard(c['mean'])
        noninferior=complete and f['mean']['psnr']>=-.05 and control.quality_guard(f['mean'])
        matching=complete and all(v['mean']['psnr']>=.02 and v['images_PSNR_positive']>=4 and control.quality_guard(v['mean']) for v in (f,s))
        p=comparisons['PHYSICAL']
        covariance=complete and WEIGHTS[task]>0 and p['mean']['psnr']>=.02 and p['images_PSNR_positive']>=4 and control.quality_guard(p['mean'])
        groups.append(dict(task=task,lambda_frozen=WEIGHTS[task],completed_cases=len(rows),SELECTED_minus=comparisons,
            quality_vs_C=bool(quality),engineering_noninferiority_vs_matched=bool(noninferior),
            quality_candidate=bool(quality and noninferior),state_matching_candidate=bool(matching),covariance_increment_candidate=bool(covariance),
            gammas={r['case']['image_id']:r['gamma'] for r in rows},per_image={r['case']['image_id']:
                {name:r['evaluation'][name]['delta'] for name in POLICIES} for r in rows}))
    return groups
