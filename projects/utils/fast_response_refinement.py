"""A98 fixed-budget coupled response descent. No inner CG or target change."""
import math
from unittest.mock import patch
from utils import response_target_lm as lm,response_target_runtime as rt,full_unroll_response as full
ARMS=('SCALAR_K0','DIAG_K1','DIAG_K3','COARSE_K3')
STEPS=dict(SCALAR_K0=0,DIAG_K1=1,DIAG_K3=3,COARSE_K3=3)
MAX_TRIALS=2


def solve(provider,initial,active,updates):
    """Exact linear-model steepest-descent length, up to2 true nonlinear checks."""
    p=initial;trace=[];first=None;final=None;initial_gradient=None;status='UPDATE_BUDGET'
    for k in range(updates):
        value,gradient=provider.linearize(p);residual=value-provider.target
        objective=.5*lm.dot(residual,residual);g=gradient*active;gn=lm.norm(g)
        if first is None:first=objective;initial_gradient=gn
        final=objective
        if gn<=lm.GRAD_ATOL+lm.GRAD_RTOL*initial_gradient:status='GRADIENT_TOLERANCE';break
        response=provider.jvp(p,g);denominator=lm.dot(response,response)
        if not math.isfinite(denominator) or denominator<0:raise RuntimeError('Nonfinite local response energy')
        if denominator==0:status='ZERO_LOCAL_RESPONSE';break
        eta=lm.dot(g,g)/denominator;accepted=False
        for trial in range(MAX_TRIALS):
            scale=eta/(2**trial);candidate=p-scale*g
            prediction=provider.value(candidate);newvalue=.5*lm.dot(prediction-provider.target,prediction-provider.target)
            predicted=scale*lm.dot(g,g)-.5*scale*scale*denominator
            if not all(math.isfinite(v) for v in (newvalue,predicted,scale)):raise RuntimeError('Nonfinite fast response step')
            ratio=(objective-newvalue)/predicted if predicted>0 else float('-inf')
            admitted=newvalue<objective and ratio>.1
            trace.append(dict(iteration=k,trial=trial,eta=scale,old_value=objective,new_value=newvalue,predicted_gain=predicted,
                actual_gain=objective-newvalue,agreement=ratio if predicted>0 else None,accepted=admitted,gradient_norm=gn))
            if admitted:p=candidate.detach() if hasattr(candidate,'detach') else candidate;final=newvalue;accepted=True;break
        if not accepted:status='TRIAL_BUDGET';break
    return p,dict(status=status,converged=status=='GRADIENT_TOLERANCE',initial_value=first,final_value=final,trace=trace,
        accepted_updates=sum(v['accepted'] for v in trace),requested_updates=updates,
        note='Fixed-budget descent; no final stationarity claim after accepted last update. No inner CG.')


def active_mask(coefficients,coarse=False):
    """LL2 of the SAME db4/level3 packing: quarter spatial extent, no re-banding."""
    active=coefficients!=0
    if coarse:
        active=active.clone() if hasattr(active,'clone') else active.copy()
        h,w=coefficients.shape[-2:];active[...,h//4:,:]=False;active[...,:,w//4:]=False
    return active


def action(model,op,y,z,uncond,q,clean,t,s,arm,count):
    import torch
    from utils.shrink_multitask_runtime import informative
    from utils.noise_signal_shrink import select
    if arm not in ARMS:raise ValueError('Unknown A98 arm')
    with torch.no_grad():
        if s>=0:rt.increment(count,'response_base_forward_attempts');base_clean=rt.predict_clean(model,uncond,s)
        else:base_clean=uncond
        base=informative(op.H(base_clean),op);r=informative(y,op)-base
        omega=lm.noise_weight(lm.dot(r,r),r.numel(),.05);target=base+omega*r;qn=lm.norm(q)
        if qn==0:return q*0,dict(status='ZERO_NATIVE_FEASIBLE_SET',omega=omega,q_norm=0.),{}
        if s>=0:residual,d=rt.original_scalar(model,op,y,base_clean,q,s)
        else:residual=r.double();d=informative(op.H(q),op).double()
        scalar=select(lm.dot(residual,d),lm.dot(residual,residual),lm.dot(d,d),r.numel(),.05)
        gamma=scalar['gamma'];initial_delta=gamma*q
    detail=dict(status='CLOSED_FORM',scalar=dict(status='CLOSED_FORM_NEXT_PROXY',**scalar),omega=omega,q_norm=qn,scalar_norm=lm.norm(initial_delta),
        residual_energy=lm.dot(r,r),measurements=r.numel(),noise_energy=r.numel()*.05**2)
    if STEPS[arm]==0:return initial_delta,detail,{}
    geometry=rt.Geometry(q,z.dwt,False,count);initial=geometry.c*gamma
    active=active_mask(geometry.c,arm=='COARSE_K3')
    provider=full.Provider(model,op,uncond,s,geometry,target,count)
    p,info=solve(provider,initial,active,STEPS[arm]);delta=geometry.image(p).detach()
    if not torch.equal(p[~active],initial[~active]):raise RuntimeError('Frozen coefficients changed')
    detail.update(diagonal=full.compact(info),active_coefficients=int(active.sum()),total_coefficients=active.numel(),
        diagonal_norm=lm.norm(delta),compensation_norm=lm.norm(delta-initial_delta),
        diagonal_cosine_to_native=lm.dot(delta,q)/max(lm.norm(delta)*qn,1e-30))
    return delta,detail,dict(diagonal=info['trace'])


def step(model,z,op,y,x,k,arm,count):
    # Old step contains the unchanged native DDPM draw, endpoint/call/RNG gates.
    with patch.object(full,'action',action):result=full.step(model,z,op,y,x,k,arm,count)
    s=result[2]['next_t'];limit=1 if s<0 else 2+4*STEPS[arm]
    if result[2]['NN_calls_this_step']>limit:raise RuntimeError('Fast controller exceeded registered NN budget')
    return result


def budget():
    per={arm:30+29*(1+4*STEPS[arm]) for arm in ARMS}
    return dict(arms=48,per_suffix_forward_max=per,max_new_neural_forwards=12*sum(per.values()),
        max_new_neural_VJP=12*29*sum(STEPS.values()),max_new_neural_JVP=12*29*sum(STEPS.values()),
        optimizer_backward=0,probe_F_VJP_JVP_per_load=[2,1,1],terminal_operator_derivatives_separate=True)
