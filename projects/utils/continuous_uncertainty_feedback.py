"""A43 label-free continuous actions and full-column PSD completion (NumPy only)."""
import math
import numpy as np
from utils import uncertainty_feedback as old

CONTROLLERS=('PHYSICAL','ISO','PROJECTED','NYSTROM')
POLICIES=('C','FIXED2','FIXED4','FIXED8','NOISE_BUDGET')+CONTROLLERS
FIT_TOLERANCE=.05


def nystrom_factor(V,Y,B):
    V,Y,B=(np.asarray(x,dtype=np.float64) for x in (V,Y,B))
    if (V.ndim!=2 or V.shape!=Y.shape or B.shape!=(V.shape[1],V.shape[1]) or
            any(not np.isfinite(x).all() for x in (V,Y,B))):
        raise ValueError('Invalid covariance sketch')
    if not B.size:raise ValueError('Empty covariance sketch')
    eig,rot=np.linalg.eigh((B+B.T)*.5);scale=max(float(np.max(np.abs(eig))),1e-20)
    if eig.min() < -1e-6*scale:raise ValueError('Indefinite covariance sketch')
    keep=eig>1e-12*scale
    if not keep.any():raise ValueError('No resolved positive eigenvalue')
    F=(Y@rot[:,keep])/np.sqrt(eig[keep])[None,:]
    closure=float(np.linalg.norm(F@(F.T@V)-Y))/max(float(np.linalg.norm(Y)),1e-20)
    if not np.isfinite(F).all() or not math.isfinite(closure):raise ValueError('Nonfinite completion')
    return F,dict(column_closure_relative_error=closure,rank=int(keep.sum()),
                  covariance_nonzero_eigenvalues=np.linalg.eigvalsh(F.T@F).tolist())


def noise_budget(alpha_t,alpha_s,q_rms):
    if not (0<alpha_t<alpha_s<1) or not math.isfinite(q_rms) or q_rms<0:
        raise ValueError('Invalid posterior/noise input')
    variance=(1-alpha_s)/(1-alpha_t)*(1-alpha_t/alpha_s)
    sigma=math.sqrt(max(0.,variance))
    cap=max(1.,sigma/q_rms) if q_rms>1e-12 else 1.
    if not math.isfinite(cap):raise ValueError('Nonfinite multiplier cap')
    return dict(posterior_noise_RMS=sigma,q_RMS=float(q_rms),cap=float(cap),
                quantized_cap=math.floor(cap*1000.)/1000.,q_resolved=bool(q_rms>1e-12))


def proposal(curve,cap):
    x=np.array(old.GAMMAS);y=np.array([curve[g] for g in old.GAMMAS],dtype=np.float64)
    if not np.isfinite(y).all() or not math.isfinite(cap) or cap<1.:raise ValueError('Invalid action curve/cap')
    variation=float(np.ptp(y));scale=max(float(np.max(np.abs(y))),1.)
    if variation<=1e-12*scale:return dict(usable=False,reason='FLAT',gamma=1.)
    X=np.column_stack((x*x,x,np.ones(len(x))));a,b,c=np.linalg.lstsq(X,y,rcond=None)[0]
    error=float(np.max(np.abs(X@np.array([a,b,c])-y))/variation)
    usable=bool(a>1e-12*scale and error<=FIT_TOLERANCE)
    vertex=float(-b/(2*a)) if a>1e-12*scale else None
    upper=math.floor(cap*1000.)/1000.
    gamma=min(upper,max(0.,round(min(cap,max(0.,vertex)),3))) if usable else 1.
    return dict(usable=usable,reason='BOUNDED_QUADRATIC_PROPOSAL' if usable else 'UNRELIABLE_LOCAL_FIT',
                gamma=float(gamma),vertex=vertex,fit_relative_error=error,cap=float(cap),
                coefficients=[float(a),float(b),float(c)])


def no_worse(value,reference):
    return bool(value is not None and math.isfinite(value) and value<=reference+1e-10*max(1.,abs(reference)))


def continuous_choice(curve,cap,evaluate,usable=True):
    """evaluate(g)->{loss,objective}; no GT, suffix scores or final labels accepted."""
    fit=proposal(curve,cap);record=dict(fit=fit,gamma=1.,attempts=[])
    if not usable or not fit['usable']:
        record['reason']='STATE_UNRESOLVED' if not usable else fit['reason'];return record
    baseline=evaluate(1.);seen=set()
    for divisor in (1.,2.,4.):
        g=round(1.+(fit['gamma']-1.)/divisor,3)
        if g in seen or g==1.:continue
        seen.add(g);value=evaluate(g)
        accepted=no_worse(value['loss'],baseline['loss']) and no_worse(value['objective'],baseline['objective'])
        record['attempts'].append(dict(gamma=g,**value,accepted=bool(accepted)))
        if accepted:
            record.update(gamma=g,reason='ACTUAL_LOCAL_CHECK_ACCEPTED');return record
    record['reason']='LOCAL_CHECK_REJECTED_OR_NO_CHANGE';return record


def quality_guard(delta):return bool(delta['ssim']>=-.002 and delta['lpips']<=.005)


def state_increment(deltas):
    full=deltas['NYSTROM']
    return bool(full['psnr']>=.05 and quality_guard(full) and all(
        full['psnr']-deltas[name]['psnr']>=.02 and quality_guard(
            {m:full[m]-deltas[name][m] for m in ('ssim','lpips')}) for name in ('PHYSICAL','ISO')))
