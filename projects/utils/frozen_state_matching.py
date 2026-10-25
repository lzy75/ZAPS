"""A44 fixed rules, label-free cyclic reassignment and per-task screening."""
import numpy as np
from utils import continuous_uncertainty_feedback as feedback

POLICIES=('C','FIXED8','TASK_FIXED','PHYSICAL','ISO','STATE','SHUFFLE')
REFERENCE={'super_resolution':127.086,'gaussian_deblur':7.589,'inpainting':28.526,'motion_deblur':8.327}
IMAGES=('00446','00559','00608','00873','00951')
METRICS=('psnr','ssim','lpips','loss')


def allocation(cases,choices,fixed):
    """choices contains only frozen selected multipliers; no GT/quality/ranking arguments."""
    if set(fixed)!=set(REFERENCE) or any(abs(fixed[t]-REFERENCE[t])>1e-9 for t in fixed):
        raise ValueError('A43 fixed reference differs')
    if len(cases)!=20 or set(choices)!={c['key'] for c in cases}:raise ValueError('Need all 20 choices')
    out={}
    for task in REFERENCE:
        group=sorted([c for c in cases if c['task']==task],key=lambda c:c['image_id'])
        if tuple(c['image_id'] for c in group)!=IMAGES:raise ValueError('Frozen five-image cohort differs')
        for i,c in enumerate(group):
            donor=group[(i+1)%len(group)];own=choices[c['key']]
            if set(own)!={'PHYSICAL','ISO','STATE'} or any(not np.isfinite(g) or g<0 for g in own.values()):
                raise ValueError('Invalid label-free choice')
            out[c['key']]=dict(gamma=dict(C=1.,FIXED8=8.,TASK_FIXED=fixed[task],**own,
                SHUFFLE=choices[donor['key']]['STATE']),donor=donor['key'])
    return out


def task_summary(records):
    groups=[]
    for task in REFERENCE:
        rows=[r for r in records if r['case']['task']==task]
        mean={name:{m:float(np.mean([r['evaluation'][name]['delta'][m] for r in rows])) if rows else None
                    for m in METRICS} for name in POLICIES}
        comparisons={}
        for name in POLICIES:
            if name=='STATE':continue
            differences=[{m:r['evaluation']['STATE']['delta'][m]-r['evaluation'][name]['delta'][m] for m in METRICS} for r in rows]
            value={m:float(np.mean([v[m] for v in differences])) if rows else None for m in METRICS}
            wins=sum(v['psnr']>1e-9 for v in differences)
            required=.05 if name=='C' else .02
            passed=len(rows)==5 and value['psnr']>=required and wins>=(4 if name=='C' else 3) and feedback.quality_guard(value)
            comparisons[name]=dict(mean=value,images_PSNR_positive=wins,engineering_passed=bool(passed))
        supported=len(rows)==5 and all(comparisons[n]['engineering_passed'] for n in ('C','TASK_FIXED','PHYSICAL','ISO','SHUFFLE'))
        groups.append(dict(task=task,completed_cases=len(rows),mean_delta_vs_C=mean,STATE_minus=comparisons,
            state_matching_candidate=bool(supported),gammas={r['case']['image_id']:r['gamma'] for r in rows}))
    return groups
