"""A52 ablation of the q-direction constraint, not a new posterior theorem."""
import math
import numpy as np
from utils import current_score_runtime_r2 as base
from utils import current_score_geometry as algebra

ARMS=('PROX_Q','ALIGN_Q','FULL_PROX','FULL_ALIGN')


def numpy_target(q,v,u,coefficient,mode):
    """Independent finite-dimensional reference for server Torch admission."""
    q,v=np.asarray(q,dtype=float),np.asarray(v,dtype=float)
    if q.shape!=v.shape or not np.isfinite(q).all() or not np.isfinite(v).all() or coefficient<=0:
        raise ValueError('Invalid geometry')
    if mode=='FULL_PROX':target=coefficient*v
    elif mode=='FULL_ALIGN':
        u=np.asarray(u,dtype=float)
        if u.shape!=q.shape or not np.isfinite(u).all() or float(np.sum(u*u))==0:
            raise ValueError('Resolved u required')
        target=coefficient*float(np.sum(v*u))/float(np.sum(u*u))*u
    else:raise ValueError('Unknown full target')
    decision=algebra.nonnegative_projection(q,target)
    return target,decision


def target_from_readout(q,v,u,coefficient,mode,reading):
    import torch
    if not reading['resolved']:raise ValueError('Unresolved geometry must use explicit fallback')
    if mode=='FULL_PROX':target=coefficient*v
    elif mode=='FULL_ALIGN':target=coefficient*reading['alignment_scale']*u
    else:raise ValueError('Unknown full target')
    if not torch.isfinite(target).all():raise RuntimeError('Nonfinite target')
    # Compare the projection of THIS target with the old rule at the SAME state.
    qq=q.detach().double();tt=target.detach().double()
    gamma=max(0.,float((qq*tt).sum()/qq.square().sum()))
    old_mode='PROX_Q' if mode=='FULL_PROX' else 'ALIGN_Q'
    expected=reading['gamma'][old_mode]
    error=abs(gamma-expected)/max(1.,abs(expected))
    if error>2e-6:raise RuntimeError('Shadow q projection differs from A50 geometry')
    norm=float(tt.norm());bound=coefficient*float(v.detach().double().norm())
    if norm>bound*(1.+2e-6)+1e-12:raise RuntimeError('Projection norm bound violated')
    projection=gamma*qq
    extra=dict(shadow_gamma=gamma,shadow_projection_error=error,
               full_target_RMS=float(tt.square().mean().sqrt()),
               discarded_fraction=float((tt-projection).norm())/norm if norm else 0.,
               cosine_q_target=float((qq*tt).sum()/qq.norm()/tt.norm()) if norm else None)
    return target.detach(),extra


def sensed_field(z,y,x,t,next_t,positions,sigma,mode,mask=None,before_backward=None):
    import torch
    if mode in ('PROX_Q','ALIGN_Q'):
        return base.sensed_field(z,y,x,t,next_t,positions,sigma,mode,mask,before_backward)
    if mode not in ('FULL_PROX','FULL_ALIGN'):raise ValueError('Unknown arm')
    inp,raw=base.raw_mean(z.dm,x,t,graph=True)
    field=base.field_from_mean(z,y,raw,t,positions)
    with torch.no_grad():
        residual=y-z.A.H(raw.detach())
        if mask is not None:residual=residual*mask.to(dtype=residual.dtype)
        row=dict(R_raw=base.informative_rms(residual,mask)/sigma,
                 R_clean=base.informative_rms(y-z.A.H(field['clean']),mask)/sigma,
                 clip_fraction=float((raw.detach().abs()>1).float().mean()),backward_calls=0)
    v,gate=algebra.proximal_vector(lambda w:.5*z.A.base.H(w),z.A.transpose,residual,
        float(z.dm.alphas_cumprod[t]),sigma,rtol=base.CG_RTOL,maxiter=base.CG_MAXITER)
    row['cg']=gate
    if v is None:
        row.update(resolved=False,reason='CG_UNRESOLVED_FALLBACK',constraint_removed=False)
        return field,1.,row
    u=None
    if mode=='FULL_ALIGN':
        if before_backward is not None:before_backward()
        row['backward_calls']=1
        u=algebra.input_vjp(raw,inp,v)
    alpha_s=float(z.dm.alphas_cumprod[next_t]) if next_t>=0 else 1.
    coefficient=algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]),alpha_s)
    reading=base.gram_readout(field['q'],v,u,coefficient,float(x.detach().double().square().mean().sqrt()))
    row.update(reading,coefficient=coefficient,constraint_removed=False)
    if not reading['resolved']:return field,1.,row
    target,extra=target_from_readout(field['q'],v,u,coefficient,mode,reading)
    row.update(extra,constraint_removed=True)
    field=dict(field,q=target)
    return field,1.,row
