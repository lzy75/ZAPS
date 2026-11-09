"""A100 equal-call native-ray versus full-coefficient response refinement."""
import math
from unittest.mock import patch
from utils import response_target_lm as lm,response_target_runtime as rt,full_unroll_response as full

ARMS=('RAY_K1_EQ','DIAG_K1_EQ')
TRIALS=2


def solve(provider,initial,project):
    """One response-gradient update with exactly one VJP/JVP and two value calls."""
    p=initial
    value,gradient=provider.linearize(p)
    residual=value-provider.target
    objective=.5*lm.dot(residual,residual)
    response=provider.jvp(p,gradient)
    numerator=lm.dot(gradient,gradient);denominator=lm.dot(response,response)
    eta=numerator/denominator if denominator>0 else 0.
    trials=[]
    for index,scale in enumerate((eta,eta/2)):
        candidate=project(p-scale*gradient)
        step=p-candidate
        effective=lm.dot(step,gradient)/numerator if numerator>0 else 0.
        prediction=provider.value(candidate)
        new_value=.5*lm.dot(prediction-provider.target,prediction-provider.target)
        predicted=effective*numerator-.5*effective*effective*denominator
        ratio=(objective-new_value)/predicted if predicted>0 else float('-inf')
        trials.append(dict(trial=index,eta=scale,effective_eta=effective,old_value=objective,new_value=new_value,
            predicted_gain=predicted,actual_gain=objective-new_value,
            agreement=ratio if predicted>0 else None,accepted=new_value<objective and ratio>.1,
            gradient_norm=math.sqrt(max(numerator,0.))))
    accepted=next((i for i,row in enumerate(trials) if row['accepted']),None)
    result=project(p-trials[accepted]['eta']*gradient) if accepted is not None else p
    final=trials[accepted]['new_value'] if accepted is not None else objective
    return result,dict(status='ACCEPTED' if accepted is not None else 'TRIAL_BUDGET',initial_value=objective,
        final_value=final,accepted_trial=accepted,trace=trials,
        note='Fixed equal-call K1: one VJP, one JVP, two true-value checks in both arms.')


def action(model,op,y,z,uncond,q,clean,t,s,arm,count):
    import torch
    from utils.shrink_multitask_runtime import informative
    from utils.noise_signal_shrink import select
    if arm not in ARMS:raise ValueError('Unknown A100 arm')
    with torch.no_grad():
        if s>=0:rt.increment(count,'response_base_forward_attempts');base_clean=rt.predict_clean(model,uncond,s)
        else:base_clean=uncond
        base=informative(op.H(base_clean),op);r=informative(y,op)-base
        omega=lm.noise_weight(lm.dot(r,r),r.numel(),.05);target=base+omega*r;qn=lm.norm(q)
        if qn==0:return q*0,dict(status='ZERO_NATIVE_FEASIBLE_SET',omega=omega,q_norm=0.),{}
        if s>=0:residual,d=rt.original_scalar(model,op,y,base_clean,q,s)
        else:residual=r.double();d=informative(op.H(q),op).double()
        scalar=select(lm.dot(residual,d),lm.dot(residual,residual),lm.dot(d,d),r.numel(),.05)
        gamma=scalar['gamma']
    is_ray=arm=='RAY_K1_EQ'
    geometry=rt.Geometry(q,z.dwt,is_ray,count)
    initial=(torch.tensor([gamma*qn],device=q.device,dtype=q.dtype) if is_ray else geometry.c*gamma)
    provider=full.Provider(model,op,uncond,s,geometry,target,count)
    solution,info=solve(provider,initial,geometry.project)
    delta=geometry.image(solution).detach()
    detail=dict(status=info['status'],omega=omega,q_norm=qn,scalar=dict(status='CLOSED_FORM_NEXT_PROXY',**scalar),
        scalar_norm=gamma*qn,applied_norm=lm.norm(delta),parameterization='native_ray' if is_ray else 'full_wavelet_coefficients',
        response_update=full.compact(info))
    return delta,detail,dict(response_update=info['trace'])


def step(model,z,op,y,x,k,arm,count):
    with patch.object(full,'action',action):result=full.step(model,z,op,y,x,k,arm,count)
    expected=1 if result[2]['next_t']<0 else 6
    if result[2]['NN_calls_this_step']!=expected:raise RuntimeError('A100 fixed-call contract violated')
    return result


def budget():
    return dict(arms=24,per_suffix_forward_exact=175,per_suffix_input_VJP_exact=29,
        per_suffix_input_JVP_exact=29,max_new_neural_forwards=24*175,
        max_new_neural_VJP=24*29,max_new_neural_JVP=24*29,optimizer_backward=0,
        terminal_operator_derivatives_separate=True,probe_F_VJP_JVP_per_load=[2,1,1])
