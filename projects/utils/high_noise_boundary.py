"""A47 GT-free noise calibration; quality and current-image matching kept separate."""
import statistics
import numpy as np
from utils import frozen_four_task_confirmation as old

CALIBRATION=('00446','00559','00608')
VALIDATION=('00057','00190','00240')


def allocation(cases,choices,frozen_fixed):
    if len(cases)!=24 or set(choices)!={c['key'] for c in cases}:raise ValueError('Need all24 choices')
    assignments={};matched={}
    for task in old.WEIGHTS:
        cal=sorted([c for c in cases if c['task']==task and c['cohort']=='calibration'],key=lambda c:c['image_id'])
        val=sorted([c for c in cases if c['task']==task and c['cohort']=='validation'],key=lambda c:c['image_id'])
        if tuple(c['image_id'] for c in cal)!=CALIBRATION or tuple(c['image_id'] for c in val)!=VALIDATION:
            raise ValueError('Frozen split differs')
        for c in cal+val:
            own=choices[c['key']]
            if set(own)!={'PHYSICAL','ISO','STATE','SELECTED'} or any(not np.isfinite(g) or g<0 for g in own.values()):raise ValueError('Invalid choice')
            w=old.WEIGHTS[task]
            if w in (0.,1.) and own['SELECTED']!=own['PHYSICAL' if w==0. else 'STATE']:raise ValueError('Endpoint alias differs')
        matched[task]=float(statistics.median(choices[c['key']]['SELECTED'] for c in cal))
        for i,c in enumerate(val):
            donor=val[(i+1)%3]
            assignments[c['key']]=dict(gamma=dict(C=1.,TASK_FIXED=frozen_fixed[task],MATCHED_FIXED=matched[task],
                **choices[c['key']],SHUFFLE=choices[donor['key']]['SELECTED']),donor=donor['key'])
    return assignments,matched


def summarize(records):
    groups=[]
    for task in old.WEIGHTS:
        rows=[r for r in records if r['case']['task']==task];comparisons={}
        for name in old.POLICIES:
            if name=='SELECTED':continue
            values=[{m:r['evaluation']['SELECTED']['delta'][m]-r['evaluation'][name]['delta'][m] for m in old.METRICS} for r in rows]
            comparisons[name]=dict(mean={m:float(np.mean([v[m] for v in values])) if rows else None for m in old.METRICS},
                images_PSNR_positive=sum(v['psnr']>1e-9 for v in values))
        complete=len(rows)==3;c,f,s=(comparisons[n] for n in ('C','MATCHED_FIXED','SHUFFLE'))
        quality=complete and c['mean']['psnr']>=.05 and c['images_PSNR_positive']==3 and old.control.quality_guard(c['mean'])
        noninferior=complete and f['mean']['psnr']>=-.05 and old.control.quality_guard(f['mean'])
        matching=complete and all(v['mean']['psnr']>=.02 and v['images_PSNR_positive']>=2 and old.control.quality_guard(v['mean']) for v in (f,s))
        groups.append(dict(task=task,lambda_frozen=old.WEIGHTS[task],completed_cases=len(rows),SELECTED_minus=comparisons,
            quality_vs_C=bool(quality),quality_candidate=bool(quality and noninferior),state_matching_candidate=bool(matching),
            per_image={r['case']['image_id']:{n:r['evaluation'][n]['delta'] for n in old.POLICIES} for r in rows},
            gammas={r['case']['image_id']:r['gamma'] for r in rows}))
    return groups
