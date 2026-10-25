"""A42 projected covariance state -> local Gaussian target -> label-free action choice."""
import math
import numpy as np

POINTS=((19,641),(24,319))
GAMMAS=(0.,.5,1.,1.5,2.)
ETAS=(.002,.004)
POLICIES=('KEEP1','FIXED2','PHYSICAL','ISO','FULL')


def plain(value):
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {plain(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def active(value,mask=None):
    x=np.asarray(value,dtype=np.float64)
    if mask is None:return x.reshape(-1)
    flag=np.broadcast_to(np.asarray(mask,dtype=bool),x.shape)
    if not flag.any():raise ValueError('No informative observation')
    return x[flag]


def basis(vectors):
    columns=[]
    for item in vectors:
        vector=np.asarray(item,dtype=np.float64).reshape(-1).copy();original=float(np.linalg.norm(vector))
        if not math.isfinite(original):raise ValueError('Nonfinite direction')
        if original<=1e-10:continue
        for _ in range(2):
            for q in columns:vector-=q*float(q@vector)
        norm=float(np.linalg.norm(vector))
        if norm>max(1e-10,1e-3*original):columns.append(vector/norm)
    return np.stack(columns,axis=1) if columns else np.empty((np.asarray(vectors[0]).size,0))


def covariance_gate(matrices,input_span_error=0.):
    if len(matrices)!=2:raise ValueError('Need two frozen scales')
    a,b=(np.asarray(v,dtype=np.float64) for v in matrices)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite([a,b]).all():
        raise ValueError('Invalid covariance matrices')
    if not a.size:return dict(usable=False,reason='NO_RESOLVED_CONTROL_DIRECTION',rank=0),a
    norm=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),1e-20)
    spread=float(np.linalg.norm(a-b))/norm
    skew=max(float(np.linalg.norm(v-v.T))/max(float(np.linalg.norm(v)),1e-20) for v in (a,b))
    # Validate both scales; use the smaller fixed scale, never choose a scale by final quality.
    symmetric=[(v+v.T)*.5 for v in (a,b)]
    eigen=[np.linalg.eigvalsh(v) for v in symmetric]
    eigen_scale=max(float(np.max(np.abs(v))) for v in eigen)
    tolerance=1e-6*max(eigen_scale,1e-20)
    minimum=min(float(v.min()) for v in eigen)
    usable=spread<=.10 and skew<=.05 and minimum>=-tolerance and input_span_error<=1e-3
    eig,rot=np.linalg.eigh(symmetric[0]);projected=(rot*np.maximum(eig,0.))@rot.T
    gate=dict(usable=bool(usable),reason='USABLE_PROJECTED_COVARIANCE' if usable else 'COVARIANCE_METROLOGY_UNRESOLVED',
        rank=len(eig),relative_scale_spread=spread,relative_skew=skew,eigenvalues_by_scale=[v.tolist() for v in eigen],
        input_span_relative_error=float(input_span_error),PSD_negative_tolerance=tolerance,
        clipped_tiny_negative_eigenvalues=int(np.sum(eig<0)) if usable else 0)
    return gate,projected


def gaussian_target(mean,V,B,W,residual,sigma):
    """Rank-r Gaussian conditioning using an SPD r x r solve, no large inverse."""
    mean=np.asarray(mean,dtype=np.float64).reshape(-1);V=np.asarray(V,dtype=np.float64)
    B=np.asarray(B,dtype=np.float64);W=np.asarray(W,dtype=np.float64);r=np.asarray(residual,dtype=np.float64).reshape(-1)
    if sigma<=0 or B.shape!=(V.shape[1],V.shape[1]) or W.shape!=(len(r),V.shape[1]):raise ValueError('Gaussian dimensions differ')
    eig,rot=np.linalg.eigh(B)
    if eig.min() < -1e-12:raise ValueError('Indefinite conditional covariance')
    square=(rot*np.sqrt(np.maximum(eig,0.)))@rot.T;gram=W.T@W
    system=np.eye(len(eig))+square@gram@square/(sigma*sigma)
    a=square@np.linalg.solve(system,square@(W.T@r)/(sigma*sigma))
    target=mean+V@a
    innovation=float(r@(r-W@a)/(sigma*sigma)/len(r))
    if innovation < -1e-8 or not np.isfinite(target).all():raise ValueError('Invalid Gaussian target/innovation')
    return target,dict(uncertainty_noise_ratio=float(np.trace(B@gram)/(len(r)*sigma*sigma)),
        whitened_innovation=max(0.,innovation),raw_residual_noise_ratio=float(np.sqrt(np.mean(r*r))/sigma),
        posterior_shift_RMS=float(np.sqrt(np.mean((target-mean)**2))),projected_prior_eigenvalues=eig.tolist())


def choose(cleans,means,y,mask,sigma,mean,V,B,W,usable,raw_observation=None):
    """No GT/final metrics accepted. Constraints are on predicted/local quantities only."""
    if set(cleans)!=set(GAMMAS) or set(means)!=set(GAMMAS):raise ValueError('Incomplete action bank')
    obs=active(y,mask);loss={g:float(np.mean((obs-active(means[g],mask))**2)/(2*sigma*sigma)) for g in GAMMAS}
    order=(1.,.5,1.5,0.,2.)
    physical=min(order,key=lambda g:loss[g])
    if loss[1.]-loss[physical]<=1e-10*max(1.,loss[1.]):physical=1.
    result=dict(gamma=dict(KEEP1=1.,FIXED2=2.,PHYSICAL=physical,ISO=1.,FULL=1.),local_measurement_loss=loss,
                state={},predicted_risk={},admissible={})
    if not usable:return result
    residual=obs-active(raw_observation,mask) if raw_observation is not None else None
    # y-A(mean) belongs to the same RAW reference used to estimate covariance.
    if residual is None:raise ValueError('Missing raw reference observation')
    for name,cov in (('ISO',np.eye(len(B))*float(np.trace(B))/len(B)),('FULL',B)):
        target,state=gaussian_target(mean,V,cov,W,residual,sigma)
        risks={g:float(np.mean((np.asarray(cleans[g],dtype=np.float64).reshape(-1)-target)**2)) for g in GAMMAS}
        admissible=[g for g in order if loss[g]<=loss[1.]+1e-10*max(1.,loss[1.])
                    and risks[g]<=risks[1.]+1e-10*max(1.,risks[1.])]
        chosen=min(admissible,key=lambda g:risks[g])
        if risks[1.]-risks[chosen]<=1e-10*max(1.,risks[1.]):chosen=1.
        result['gamma'][name]=chosen;result['state'][name]=state
        result['predicted_risk'][name]=risks;result['admissible'][name]=admissible
    return result


def calls(grid,ranks):
    times=list(reversed(grid))
    if len(times)!=30 or len(ranks)!=len(POINTS) or any(times[k]!=t for k,t in POINTS) or any(r not in (0,1,2) for r in ranks):
        raise ValueError('Grid/rank contract differs')
    return times+[times[k+1] for (k,_),r in zip(POINTS,ranks) for _ in range(6+4*r)]
