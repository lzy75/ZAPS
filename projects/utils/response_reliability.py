"""A48 bounded, label-free action checks; all readouts precede final quality."""
import math
import numpy as np
from utils import uncertainty_feedback as sensor

WEIGHTS={'super_resolution':.5,'gaussian_deblur':0.,'inpainting':1.,'motion_deblur':0.}
FRACTIONS=(0.,.25,.5,.75,1.)
RHO_MIN=.75
CASE_CAP=20
METRICS=('psnr','ssim','lpips','loss')
POLICIES=('C','LEGACY','LOCAL','TRUST','MATCHED_FIXED')


def candidates(gamma):
    if not math.isfinite(gamma) or gamma<0:raise ValueError('Invalid legacy gamma')
    return list(dict.fromkeys(round(1.+f*(gamma-1.),3) for f in FRACTIONS))


def normalization(values,weight):
    if weight not in (0.,.5,1.):raise ValueError('Frozen task weight required')
    loss=np.array([values[g]['loss'] for g in sensor.GAMMAS],dtype=float)
    risk=np.array([values[g]['risks']['STATE'] for g in sensor.GAMMAS],dtype=float) if weight else np.zeros(5)
    if not np.isfinite([loss,risk]).all():raise ValueError('Nonfinite source curves')
    return dict(loss_center=float(loss[2]),risk_center=float(risk[2]),
        loss_scale=max(float(np.ptp(loss)),1e-12),risk_scale=max(float(np.ptp(risk)),1e-12))


def objective(value,norm,weight):
    loss=value.get('loss');risk=value.get('risks',{}).get('STATE')
    if loss is None or not math.isfinite(loss):return None
    if weight and (risk is None or not math.isfinite(risk)):return None
    return float((1-weight)*(loss-norm['loss_center'])/norm['loss_scale']+
        (weight*(risk-norm['risk_center'])/norm['risk_scale'] if weight else 0.))


def choose(values,legacy_gamma,weight,usable=True):
    """LOCAL and TRUST see identical paid readouts. No labels, tasks' scores or GT."""
    gs=candidates(legacy_gamma)
    if not usable or legacy_gamma==1.:
        return dict(gamma={'LOCAL':1.,'TRUST':1.},readouts=[],normalization=None,
            reason='SOURCE_UNRESOLVED_OR_NO_LEGACY_ACTION')
    norm=normalization(values,weight);x=np.array(sensor.GAMMAS)
    y=np.array([objective(values[g],norm,weight) for g in x])
    coef=np.linalg.lstsq(np.column_stack((x*x,x,np.ones(5))),y,rcond=None)[0]
    base=values[1.]['loss'];rows=[];feasible=[];trusted=[]
    for g in gs:
        value=values[g];j=objective(value,norm,weight)
        predicted=float(np.polyval(coef,1.)-np.polyval(coef,g))
        actual=-j if j is not None else None
        rho=float(actual/predicted) if actual is not None and predicted>1e-12 else None
        ok=bool(j is not None and j< -1e-12 and value['loss']<=base+1e-10*max(1.,abs(base)))
        reliable=bool(ok and rho is not None and rho>=RHO_MIN)
        rows.append(dict(gamma=g,objective=j,predicted_decrease=predicted,actual_decrease=actual,
            rho=rho,physically_feasible=ok,reliable=reliable))
        if ok:feasible.append((j,abs(g-1.),g))
        if reliable:trusted.append((j,abs(g-1.),g))
    select=lambda pool:float(min(pool)[2]) if pool else 1.
    return dict(gamma={'LOCAL':select(feasible),'TRUST':select(trusted)},normalization=norm,
        readouts=rows,coefficients=coef.tolist(),rho_min=RHO_MIN,
        reason='IDENTICAL_PROBES_WITH_OR_WITHOUT_MODEL_AGREEMENT_GATE')


def geometry(gammas,clean,q_rms):
    """Same-time response slopes, corrected for rounded unequal action spacing."""
    gs=sorted(set(gammas));out=[]
    for indices in ((0,2,4),(1,2,3)):
        if len(gs)!=5:break
        a,b,c=(gs[j] for j in indices)
        if any(g not in clean for g in (a,b,c)):continue
        ca,cb,cc=(np.asarray(clean[g],dtype=float).reshape(-1) for g in (a,b,c))
        u=(cb-ca)/(b-a);v=(cc-cb)/(c-b);nu,nv=np.linalg.norm(u),np.linalg.norm(v)
        floor=32*np.finfo(np.float32).eps*max(1.,np.linalg.norm(ca),np.linalg.norm(cb),np.linalg.norm(cc))
        resolved=bool(np.linalg.norm(cb-ca)>floor and np.linalg.norm(cc-cb)>floor)
        out.append(dict(gammas=[a,b,c],input_displacement_RMS=[(b-a)*q_rms,(c-b)*q_rms],
            resolved=resolved,cosine=float(np.clip(u@v/(nu*nv),-1.,1.)) if resolved else None,
            normalized_slope_change=float(np.linalg.norm(v-u)/max(nu+nv,1e-30)) if resolved else None))
    return dict(scales=out,scope='Same t218/same coupled noise action response; geometry logged, not used by A48 TRUST. '
        'TRUST ablates rho only. Not adjacent-time denoising curvature or a final-quality certificate.')


def summarize(records):
    groups=[]
    for task in WEIGHTS:
        rows=[r for r in records if r['case']['task']==task]
        if not rows:continue
        comparisons={}
        for other in ('C','LEGACY','LOCAL','MATCHED_FIXED'):
            diffs=[{m:r['evaluation']['TRUST']['metrics'][m]-r['evaluation'][other]['metrics'][m] for m in METRICS} for r in rows]
            comparisons[other]=dict(mean={m:float(np.mean([d[m] for d in diffs])) for m in METRICS},
                images_PSNR_positive=sum(d['psnr']>1e-9 for d in diffs),worst_PSNR=min(d['psnr'] for d in diffs))
        c=comparisons['C'];local=comparisons['LOCAL'];legacy=comparisons['LEGACY']
        quality=lambda d:d['ssim']>=-.002 and d['lpips']<=.005
        changed=sum(r['gamma']['TRUST']!=r['gamma']['LOCAL'] for r in rows)
        benefit=bool(len(rows)==5 and changed>=2 and local['mean']['psnr']>=.02 and quality(local['mean'])
            and legacy['mean']['psnr']>=-.05 and quality(legacy['mean']))
        groups.append(dict(task=task,completed_cases=len(rows),TRUST_minus=comparisons,
            changed_by_reliability_gate=changed,
            quality_vs_C=bool(len(rows)==5 and c['mean']['psnr']>=.05 and c['images_PSNR_positive']>=4 and quality(c['mean'])),
            reliability_increment_development_candidate=benefit,
            per_image={r['case']['image_id']:dict(gamma=r['gamma'],decision=r['decision'],geometry=r['geometry'],
                delta_vs_C={n:r['evaluation'][n]['delta'] for n in POLICIES}) for r in rows}))
    return groups
