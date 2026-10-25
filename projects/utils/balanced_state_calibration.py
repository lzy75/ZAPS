"""A45 dimensionless composite target; finite development calibration, no online GT."""
import numpy as np
from utils import continuous_uncertainty_feedback as control

GAMMAS=control.old.GAMMAS
LAMBDAS=(0.,.5,1.)
NAMES={0.:'PHYSICAL',.5:'MIX',1.:'STATE'}


def normalization(readouts):
    loss=np.array([readouts[g]['loss'] for g in GAMMAS],dtype=float)
    risk=np.array([readouts[g]['risks']['STATE'] for g in GAMMAS],dtype=float)
    if not np.isfinite([loss,risk]).all():raise ValueError('Nonfinite local curves')
    return dict(loss_center=float(readouts[1.]['loss']),risk_center=float(readouts[1.]['risks']['STATE']),
        loss_scale=max(float(np.ptp(loss)),1e-12),risk_scale=max(float(np.ptp(risk)),1e-12))


def objective(value,norm,weight=.5):
    if weight not in LAMBDAS:raise ValueError('Only frozen finite weight grid permitted')
    if value['loss'] is None or value['risks'].get('STATE') is None:return None
    return float((1-weight)*(value['loss']-norm['loss_center'])/norm['loss_scale']+
                  weight*(value['risks']['STATE']-norm['risk_center'])/norm['risk_scale'])


def response_geometry(c0,c1,c2):
    a,b,c=(np.asarray(x,dtype=float).reshape(-1) for x in (c0,c1,c2))
    u,v=b-a,c-b;nu,nv=float(np.linalg.norm(u)),float(np.linalg.norm(v))
    floor=32*np.finfo(np.float32).eps*max(1.,np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c))
    resolved=nu>floor and nv>floor
    return dict(resolved=bool(resolved),adjacent_action_increment_cosine=float(np.clip(u@v/(nu*nv),-1.,1.)) if resolved else None,
        normalized_second_difference=float(np.linalg.norm(v-u)/max(nu+nv,floor)),
        scope='Same t/same noise action response; not adjacent denoising-time curvature.')


def select_weight(rows,grid=LAMBDAS):
    """DEVELOPMENT ONLY: offline metrics are permitted here, never in objective/action choice."""
    if not grid or any(w not in LAMBDAS for w in grid):raise ValueError('Only a subset of the frozen grid permitted')
    feasible=[];scores={}
    for weight in grid:
        name=NAMES[weight];differences=[r['evaluation'][name]['delta'] for r in rows]
        mean={m:float(np.mean([v[m] for v in differences])) for m in ('psnr','ssim','lpips')}
        relative={m:float(np.mean([r['evaluation'][name]['delta'][m]-r['evaluation']['TASK_FIXED']['delta'][m] for r in rows]))
                  for m in ('psnr','ssim','lpips')}
        wins=sum(v['psnr']>1e-9 for v in differences);minimum_wins=int(np.ceil(.8*len(rows)))
        ok=bool(mean['psnr']>=.05 and wins>=minimum_wins and control.quality_guard(mean) and
                relative['psnr']>=-.05 and control.quality_guard(relative))
        scores[str(weight)]=dict(mean_delta_vs_C=mean,mean_delta_vs_TASK_FIXED=relative,images_PSNR_positive=wins,feasible=ok)
        if ok:feasible.append((mean['psnr'],weight))
    if not feasible:return dict(chosen_lambda=None,scores=scores,status='NO_FEASIBLE_FROZEN_GRID_CANDIDATE')
    best=max(v[0] for v in feasible);chosen=min(w for value,w in feasible if best-value<=1e-9)
    return dict(chosen_lambda=chosen,scores=scores,status='DEVELOPMENT_CALIBRATION_CANDIDATE_ONLY')
