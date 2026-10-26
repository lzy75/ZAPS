"""A60 lag only u direction; current physical vector/length at every original step."""
import math
from utils import angular_feedback_control as current

INTERVALS=(1,2,4)
ARMS=('CURRENT','LAG2','LAG4')


def refresh_steps(interval):
    if interval not in INTERVALS:raise ValueError('Frozen refresh intervals are1/2/4')
    return tuple(k for k in range(30) if k%interval==0 or k==29)


def sensed_field(z,y,x,t,s,positions,sigma,k,interval,cache,mask=None,before_backward=None):
    import torch
    due=k in refresh_steps(interval)
    if due:
        from utils.diag_fixed_time_geometry_value import observed_field
        field,row,u=observed_field(z,y,x,t,s,positions,sigma,mask,before_backward)
        if row.get('resolved') and u is not None:
            cache=dict(u=u.detach().clone(),u_RMS=row['u_RMS'],step=k,t=t)
        else:cache=None
        row.update(refresh_due=True,used_current_direction=bool(row.get('resolved')),direction_age=0)
        return field,1.,row,cache
    base,algebra=current.base,current.algebra
    _,raw=base.raw_mean(z.dm,x,t,graph=True)
    field=base.field_from_mean(z,y,raw,t,positions)
    with torch.no_grad():
        residual=y-z.A.H(raw.detach())
        if mask is not None:residual=residual*mask.to(dtype=residual.dtype)
        row=dict(R_raw=base.informative_rms(residual,mask)/sigma,
            R_clean=base.informative_rms(y-z.A.H(field['clean']),mask)/sigma,
            clip_fraction=float((raw.detach().abs()>1).float().mean()),backward_calls=0,
            refresh_due=False,used_current_direction=False,direction_age=k-cache['step'] if cache else None)
    v,gate=algebra.proximal_vector(lambda a:.5*z.A.base.H(a),z.A.transpose,residual,
        float(z.dm.alphas_cumprod[t]),sigma,rtol=base.CG_RTOL,maxiter=base.CG_MAXITER)
    row['cg']=gate
    if v is None:
        row.update(resolved=False,reason='CG_UNRESOLVED_ORIGINAL_Q_FALLBACK');return field,1.,row,cache
    coefficient=algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]),float(z.dm.alphas_cumprod[s]) if s>=0 else 1.)
    reading=base.gram_readout(field['q'],v,None,coefficient,float(x.detach().double().square().mean().sqrt()))
    row.update(reading,coefficient=coefficient)
    if not reading['resolved'] or cache is None or reading['v_RMS']<=0:
        row.update(resolved=False,reason='DIRECTION_CACHE_OR_PHYSICAL_TARGET_UNRESOLVED');return field,1.,row,cache
    target=(coefficient*reading['v_RMS']/cache['u_RMS'])*cache['u']
    rms=float(target.detach().double().square().mean().sqrt());expected=coefficient*reading['v_RMS']
    error=abs(rms-expected)/max(expected,1e-30)
    if not torch.isfinite(target).all() or not math.isfinite(error) or error>2e-6:
        raise RuntimeError('Lagged target violates CURRENT physical norm rule')
    row.update(resolved=True,reason='CACHED_DIRECTION_CURRENT_PHYSICAL_LENGTH',target_RMS=rms,
        norm_rule_relative_error=error,direction_refresh_step=cache['step'],direction_refresh_t=cache['t'],
        cached_u_RMS=cache['u_RMS'])
    return dict(field,q=target.detach()),1.,row,cache
