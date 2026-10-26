"""A53: change only cosine attenuation, retaining CURRENT v/u norms and u direction."""
import math
from statistics import median
import numpy as np
from utils import feedback_vector_constraint as original
from utils import current_score_runtime_r2 as base
from utils import current_score_geometry as algebra

TASKS=('super_resolution','inpainting')
IMAGES=('00446','00559','00608','00873','00951')
CONTROLS=('TIME_COS','SHUFFLE_COS','NORM_ONLY')


def allocate(cases,curves):
    if len(cases)!=10 or len({c['key'] for c in cases})!=10 or set(curves)!={c['key'] for c in cases}:
        raise ValueError('Need all10 unique unlabeled STATE trajectories')
    result={}
    for task in TASKS:
        group=sorted((c for c in cases if c['task']==task),key=lambda c:c['image_id'])
        if [c['image_id'] for c in group]!=list(IMAGES):raise ValueError('Frozen cohort differs')
        for c in group:
            values=curves[c['key']]
            if len(values)!=30 or any(v is None or not math.isfinite(v) or abs(v)>1+1e-12 for v in values):
                raise ValueError('Unresolved/nonfinite STATE cosine; stop before paid controls, no imputation')
        for i,c in enumerate(group):
            others=[v for v in group if v['key']!=c['key']];donor=group[(i+1)%5]
            result[c['key']]=dict(TIME_COS=[float(median(curves[v['key']][k] for v in others)) for k in range(30)],
                SHUFFLE_COS=list(curves[donor['key']]),NORM_ONLY=[1.]*30,
                references=[v['key'] for v in others],donor=donor['key'])
    return dict(assignments=result,GT_used=False,
        scope='LOO development calibration of c only. CURRENT u direction and v/u norms remain in EVERY arm. '
              'No whole-alpha/gamma transfer and no state-free full-method claim.')


def numpy_target(v,u,coefficient,cosine):
    v,u=np.asarray(v,dtype=float),np.asarray(u,dtype=float)
    if v.shape!=u.shape or not np.isfinite(v).all() or not np.isfinite(u).all() or np.linalg.norm(u)==0:
        raise ValueError('Finite resolved vectors required')
    if not math.isfinite(coefficient) or coefficient<=0 or not math.isfinite(cosine) or abs(cosine)>1+1e-12:
        raise ValueError('Invalid coefficient/cosine')
    return coefficient*np.linalg.norm(v)/np.linalg.norm(u)*cosine*u


def controlled_target(q,v,u,coefficient,reading,cosine):
    import torch
    if not math.isfinite(cosine) or abs(cosine)>1+1e-12:raise ValueError('Invalid frozen cosine')
    reference,_=original.target_from_readout(q,v,u,coefficient,'FULL_ALIGN',reading)
    scale=coefficient*reading['v_RMS']/reading['u_RMS']
    current=scale*reading['cosine_v_u']*u
    error=float((current.detach().double()-reference.double()).norm()/reference.double().norm().clamp(min=1e-30))
    if error>2e-6:raise RuntimeError('Cosine factorization differs from A52 target')
    target=scale*cosine*u
    if not torch.isfinite(target).all():raise RuntimeError('Nonfinite angular target')
    norm=float(target.detach().double().norm());bound=coefficient*float(v.detach().double().norm())
    if norm>bound*(1+2e-6)+1e-12:raise RuntimeError('Angular target norm bound violated')
    return target.detach(),dict(used_cosine=float(cosine),current_cosine=reading['cosine_v_u'],
        used_alignment_scale=reading['v_RMS']/reading['u_RMS']*cosine,
        factorization_relative_error=error,target_RMS=norm/math.sqrt(target.numel()))


def sensed_field(z,y,x,t,next_t,positions,sigma,cosine,mask=None,before_backward=None):
    import torch
    inp,raw=base.raw_mean(z.dm,x,t,graph=True)
    field=base.field_from_mean(z,y,raw,t,positions)
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
    row['backward_calls']=1
    u=algebra.input_vjp(raw,inp,v)
    alpha_s=float(z.dm.alphas_cumprod[next_t]) if next_t>=0 else 1.
    coefficient=algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]),alpha_s)
    reading=base.gram_readout(field['q'],v,u,coefficient,float(x.detach().double().square().mean().sqrt()))
    row.update(reading,coefficient=coefficient)
    if not reading['resolved']:return field,1.,row
    target,extra=controlled_target(field['q'],v,u,coefficient,reading,cosine)
    row.update(extra)
    return dict(field,q=target),1.,row
