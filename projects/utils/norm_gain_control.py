"""A54: current response-norm gain versus task/time/mismatched gains; no cosine action."""
import math
from statistics import median
import numpy as np
from utils import angular_feedback_control as previous

base,algebra=previous.base,previous.algebra
TASKS,IMAGES=previous.TASKS,previous.IMAGES
CONTROLS=('TIME_GAIN','SHUFFLE_GAIN','CONST_GAIN')


def allocate(cases,curves):
    if len(cases)!=10 or len({c['key'] for c in cases})!=10 or set(curves)!={c['key'] for c in cases}:
        raise ValueError('Need ten complete, unique NORM_ONLY trajectories')
    result={}
    for task in TASKS:
        group=sorted((c for c in cases if c['task']==task),key=lambda c:c['image_id'])
        if [c['image_id'] for c in group]!=list(IMAGES):raise ValueError('Frozen cohort differs')
        for c in group:
            if len(curves[c['key']])!=30 or any(not math.isfinite(v) or v<=0 for v in curves[c['key']]):
                raise ValueError('Finite positive resolved gains required; no imputation')
        for i,c in enumerate(group):
            others=[v for v in group if v['key']!=c['key']];donor=group[(i+1)%5]
            constant=float(median(v for o in others for v in curves[o['key']]))
            result[c['key']]=dict(TIME_GAIN=[float(median(curves[o['key']][k] for o in others)) for k in range(30)],
                SHUFFLE_GAIN=list(curves[donor['key']]),CONST_GAIN=[constant]*30,
                references=[o['key'] for o in others],donor=donor['key'])
    return dict(assignments=result,GT_used=False,
        scope='LOO development gain controls; cosine attenuation absent from EVERY arm. CURRENT u direction retained. '
              'CONST_GAIN is a matched median, not a GT-tuned strongest fixed policy.')


def numpy_target(u,coefficient,gain):
    u=np.asarray(u,dtype=float)
    if not np.isfinite(u).all() or not math.isfinite(coefficient) or coefficient<=0 or not math.isfinite(gain) or gain<=0:
        raise ValueError('Finite vector and positive coefficient/gain required')
    return coefficient*gain*u


def controlled_target(q,v,u,coefficient,reading,gain):
    import torch
    if not math.isfinite(gain) or gain<=0:raise ValueError('Invalid frozen gain')
    # Reuse A53's exact NORM_ONLY reference, including its no-mutation checks.
    reference,_=previous.controlled_target(q,v,u,coefficient,reading,1.)
    current_gain=reading['v_RMS']/reading['u_RMS']
    current=coefficient*current_gain*u
    error=float((current.detach().double()-reference.double()).norm()/reference.double().norm().clamp(min=1e-30))
    if error>2e-6:raise RuntimeError('Current gain does not reproduce NORM_ONLY')
    target=(coefficient*gain)*u
    if not torch.isfinite(target).all():raise RuntimeError('Nonfinite gain target; no hidden cap/retry')
    rms=float(target.detach().double().square().mean().sqrt())
    expected=coefficient*gain*reading['u_RMS']
    closure=abs(rms-expected)/max(expected,1e-30)
    if closure>2e-6:raise RuntimeError('Applied gain/vector norm mismatch')
    # Foreign/time gains may exceed the CURRENT physical reference norm. That is
    # the intervention; capping them with current_gain would reintroduce feedback.
    return target.detach(),dict(current_gain=current_gain,used_gain=float(gain),
        relative_dose_to_current=gain/current_gain,cosine_used_for_action=False,
        current_norm_reference_error=error,gain_norm_closure_error=closure,target_RMS=rms)


def sensed_field(z,y,x,t,next_t,positions,sigma,gain,mask=None,before_backward=None):
    import torch
    inp,raw=base.raw_mean(z.dm,x,t,graph=True);field=base.field_from_mean(z,y,raw,t,positions)
    with torch.no_grad():
        residual=y-z.A.H(raw.detach())
        if mask is not None:residual=residual*mask.to(dtype=residual.dtype)
        row=dict(R_raw=base.informative_rms(residual,mask)/sigma,
                 R_clean=base.informative_rms(y-z.A.H(field['clean']),mask)/sigma,
                 clip_fraction=float((raw.detach().abs()>1).float().mean()),backward_calls=0)
    v,gate=algebra.proximal_vector(lambda a:.5*z.A.base.H(a),z.A.transpose,residual,
        float(z.dm.alphas_cumprod[t]),sigma,rtol=base.CG_RTOL,maxiter=base.CG_MAXITER)
    row['cg']=gate
    if v is None:
        row.update(resolved=False,reason='CG_UNRESOLVED_FALLBACK');return field,1.,row
    if before_backward is not None:before_backward()
    row['backward_calls']=1;u=algebra.input_vjp(raw,inp,v)
    alpha_s=float(z.dm.alphas_cumprod[next_t]) if next_t>=0 else 1.
    coefficient=algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]),alpha_s)
    reading=base.gram_readout(field['q'],v,u,coefficient,float(x.detach().double().square().mean().sqrt()))
    row.update(reading,coefficient=coefficient)
    if not reading['resolved']:return field,1.,row
    target,extra=controlled_target(field['q'],v,u,coefficient,reading,gain);row.update(extra)
    return dict(field,q=target),1.,row
