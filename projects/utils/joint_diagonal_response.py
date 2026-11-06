"""A94 full-coefficient joint nonlinear likelihood optimization; no model training."""
import numpy as np
from utils import overall_direction_audit as old
ARMS=('SIGNED_JOINT','SCALAR_TIED','NONNEG_JOINT','SIGNED_DATA_ONLY')
UPDATES=2
MAX_BACKTRACK=6


class UnresolvedSolve(RuntimeError):
    def __init__(self,reason,trace):super().__init__(reason);self.trace=trace


def precision(arm):
    if arm not in ARMS:raise ValueError('Unknown joint arm')
    return 0. if arm=='SIGNED_DATA_ONLY' else 1.


def prox(gamma,gradient,eta,prior_precision,nonnegative=False):
    value=(gamma-eta*gradient)/(1+eta*prior_precision)
    return value*(value>0) if nonnegative else value


def bound_ok(new,old_value,dot,distance2,eta,tolerance):
    return new<=old_value+dot+distance2/(2*eta)+tolerance


def tangent_gradient(gradient,arm):
    # Orthogonal projection onto the common-gain subspace. F itself, including
    # sum(gamma_j^2), is unchanged: this is a nested restriction, not a new prior.
    return gradient*0+gradient.mean() if arm=='SCALAR_TIED' else gradient


def data_value(model,op,y,x,t,counts,gradient=False):
    import torch
    from utils.current_score_runtime import raw_mean
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.shrink_multitask_runtime import informative
    counts['solver_forward_attempts']=counts.get('solver_forward_attempts',0)+1
    if gradient:
        with torch.enable_grad(),input_vjp_parameters(model.model):
            inp,raw=raw_mean(model,x,t,graph=True);clean=raw.clamp(-1,1)
            residual=informative(y-op.H(clean),op)
            loss=.5*residual.double().square().sum()
            counts['input_vjp_attempts']+=1
            derivative=torch.autograd.grad(loss,inp,create_graph=False)[0]
            counts['input_vjp_completed']=counts.get('input_vjp_completed',0)+1
        return float(loss.detach()),derivative.detach()
    with torch.no_grad():
        _,raw=raw_mean(model,x,t,graph=False)
        residual=informative(y-op.H(raw.clamp(-1,1)),op)
        return float(.5*residual.double().square().sum()),None


def solve(model,op,y,bundle,dwt,sigma,arm,counts):
    import torch
    from utils.shrink_multitask_runtime import informative
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive known measurement sigma required')
    prior=precision(arm);nonnegative=arm=='NONNEG_JOINT';state=bundle['state'];device=model.device
    q=state['q'].to(device);origin=state['uncond'].to(device);t=state['next_t']
    def analysis(value):
        counts['solver_dwt_analysis']=counts.get('solver_dwt_analysis',0)+1
        return dwt.analysis(value)
    def synthesis(value):
        counts['solver_dwt_synthesis']=counts.get('solver_dwt_synthesis',0)+1
        return dwt.synthesis(value)
    with torch.no_grad():
        coefficients=analysis(q);gamma=torch.zeros_like(coefficients)
        residual=informative(y-op.H(bundle['reference_clean'].to(device)),op)
        current_data=float(.5*residual.double().square().sum())/(sigma*sigma)
        derivative=-coefficients*analysis(bundle['physical'].to(device))/(sigma*sigma)
    derivative=tangent_gradient(derivative,arm)
    initial_data=current_data;trace=[];accepted=0
    # Numerical initialization in gain coordinates, not a correction-norm cap.
    # In particular, coarse-scale gradient tails must not be diluted by n^-1/2.
    eta=1/max(1.,float(derivative.detach().abs().max()))
    epsilon=torch.finfo(q.dtype).eps
    for iteration in range(UPDATES):
        if iteration:
            with torch.no_grad():x=origin+synthesis(coefficients*gamma)
            value,gradient=data_value(model,op,y,x,t,counts,True)
            value/=sigma*sigma
            tolerance=64*epsilon*max(1.,abs(value),abs(current_data))
            if abs(value-current_data)>tolerance:raise RuntimeError('Accepted objective and gradient-forward value disagree')
            current_data=value
            with torch.no_grad():derivative=tangent_gradient(coefficients*analysis(gradient)/(sigma*sigma),arm)
        if not bool(torch.isfinite(derivative).all()):raise RuntimeError('Nonfinite joint derivative')
        old_prior=.5*prior*float(gamma.double().square().sum());old_objective=current_data+old_prior
        accepted_trial=False
        for attempt in range(MAX_BACKTRACK):
            with torch.no_grad():
                proposal=prox(gamma,derivative,eta,prior,nonnegative)
                if not bool(torch.isfinite(proposal).all()):raise RuntimeError('Nonfinite joint proposal')
                difference=proposal-gamma;distance2=float(difference.double().square().sum())
                if distance2==0:
                    trace.append(dict(iteration=iteration,reason='STATIONARY_AT_WORKING_PRECISION',eta=eta,objective=old_objective))
                    accepted_trial=True;break
                displacement=synthesis(coefficients*proposal)
                dot=float((derivative.double()*difference.double()).sum())
            value,_=data_value(model,op,y,origin+displacement,t,counts,False);new_data=value/(sigma*sigma)
            new_prior=.5*prior*float(proposal.double().square().sum());new_objective=new_data+new_prior
            tolerance=64*epsilon*max(1.,abs(old_objective),abs(new_objective))
            qualified=bound_ok(new_data,current_data,dot,distance2,eta,tolerance) and new_objective<=old_objective+tolerance
            trace.append(dict(iteration=iteration,backtrack=attempt,eta=eta,data=new_data,prior=new_prior,objective=new_objective,
                old_objective=old_objective,majorization_rhs=current_data+dot+distance2/(2*eta),tolerance=tolerance,accepted=qualified))
            if qualified:
                gamma=proposal.detach();current_data=new_data;accepted+=1;accepted_trial=True;break
            eta*=.5
        if not accepted_trial:raise UnresolvedSolve('Backtracking budget exhausted; no fallback reconstruction',trace)
        if distance2==0:break
    with torch.no_grad():
        correction=synthesis(coefficients*gamma)
        if not bool(torch.isfinite(correction).all()):raise RuntimeError('Nonfinite final joint correction')
        quantiles=torch.quantile(gamma.detach().double().abs().reshape(-1),torch.tensor([.5,.9,.99,1.],dtype=torch.float64,device=device)).cpu().tolist()
        reference=bundle['physical'].to(device).double();qn=float(q.double().norm());cn=float(correction.double().norm());rn=float(reference.norm())
        info=dict(arm=arm,prior_precision=prior,reference_full_field_prior_std=None if prior==0 else 1.,
            effective_scalar_prior_std=(gamma.numel()**-.5) if arm=='SCALAR_TIED' else None,
            constraint='all gains identical; same full-field objective/prior' if arm=='SCALAR_TIED' else 'nonnegative' if nonnegative else 'signed full field',
            accepted_updates=accepted,solver_status='FIXED_UPDATE_BUDGET' if accepted==UPDATES else 'STATIONARY_AT_WORKING_PRECISION',
            initial_data=initial_data,final_data=current_data,final_prior=.5*prior*float(gamma.double().square().sum()),
            coefficient_count=gamma.numel(),negative_fraction=float((gamma<0).float().mean()),
            gamma_RMS=float(gamma.double().square().mean().sqrt()),gamma_abs_quantiles=dict(zip(('p50','p90','p99','max'),quantiles)),
            native_norm=qn,correction_norm=cn,
            cosine_to_initial_physical=None if cn==0 or rn==0 else float((correction.double()*reference).sum())/(cn*rn),
            cosine_to_native=None if cn==0 or qn==0 else float((correction.double()*q.double()).sum())/(cn*qn),
            first_gradient_reused=True,trace=trace,
            objective='0.5*observed_residual_sum_of_squares/sigma_y^2 + 0.5*prior_precision*sum(gamma^2)',
            claims='Nonlinear working MAP objective, not unbiased full-image risk or converged global optimum')
    return correction.detach(),gamma.detach().cpu(),info


def run(model,op,y,entry,bundle,task,arm,sigma,counts,size=256,level=3):
    import torch,time
    z=old.make_z(model,op,entry,task,size,level);before=old.rng_state(model.device)
    start=time.perf_counter();correction,gamma,solver=solve(model,op,y,bundle,z.dwt,sigma,arm,counts)
    if str(model.device).startswith('cuda'):torch.cuda.synchronize()
    solver_seconds=time.perf_counter()-start;after=old.rng_state(model.device)
    if not torch.equal(before['cpu_rng'],after['cpu_rng']) or any(not torch.equal(a,b) for a,b in zip(before['cuda_rng'],after['cuda_rng'])):
        raise RuntimeError('Joint solver consumed sampling RNG')
    start=time.perf_counter();output,first,rng=old.continue_run(model,z,y,bundle['state'],correction)
    if str(model.device).startswith('cuda'):torch.cuda.synchronize()
    return output,first,rng,dict(solver=solver,solver_seconds=solver_seconds,continuation_seconds=time.perf_counter()-start),gamma,correction.detach().cpu()
