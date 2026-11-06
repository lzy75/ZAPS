"""A95 numerical LM tools. No Torch import, learned parameters, or GT access."""
import math
import numpy as np

ARMS=('LEGACY_SCALAR','NEXT_PROXY_SCALAR','REAL_SCALAR','REAL_DIAG')
MAX_OUTER=20
MAX_TRIALS=4
CG_MAX=12
CG_RTOL=.1
GRAD_RTOL=1e-3
GRAD_ATOL=1e-6


def dot(a,b):
    if isinstance(a,np.ndarray):return float(np.sum(a.astype(np.float64)*b.astype(np.float64)))
    return float((a.detach().double()*b.detach().double()).sum())


def norm(a):
    squared=dot(a,a)
    if not math.isfinite(squared):raise RuntimeError('Nonfinite vector norm')
    return math.sqrt(max(0.,squared))


def noise_weight(energy,count,sigma):
    if count<=0 or sigma<=0 or energy<0 or not all(math.isfinite(v) for v in (energy,sigma)):raise ValueError('Invalid known-noise statistics')
    return max(0.,1-count*sigma*sigma/energy) if energy else 0.


def cg(normal,rhs,damping,maxiter=CG_MAX,rtol=CG_RTOL):
    """Inexact LM solve; verify the TRUE damped normal residual before admission."""
    x=rhs*0;r=rhs.copy() if isinstance(rhs,np.ndarray) else rhs.clone();p=r;rr=dot(r,r)
    if not math.isfinite(rr):raise RuntimeError('Nonfinite CG RHS')
    if rr==0:return x,dict(passed=True,iterations=0,true_relative_residual=0.)
    initial=math.sqrt(rr);iteration=0
    for iteration in range(1,maxiter+1):
        ap=normal(p)+damping*p;curvature=dot(p,ap)
        if not math.isfinite(curvature) or curvature<=0:raise RuntimeError('Nonpositive/nonfinite damped Gauss-Newton curvature')
        alpha=rr/curvature;x=x+alpha*p;r=r-alpha*ap;newrr=dot(r,r)
        if not math.isfinite(newrr):raise RuntimeError('Nonfinite CG residual')
        if math.sqrt(max(newrr,0.))<=rtol*initial:break
        p=r+(newrr/rr)*p;rr=newrr
    true=normal(x)+damping*x-rhs;relative=norm(true)/initial
    return x,dict(passed=relative<=rtol,iterations=iteration,true_relative_residual=relative)


def lm(provider,initial,project=lambda v:v,max_outer=MAX_OUTER,max_trials=MAX_TRIALS,cg_max=CG_MAX,
       grad_rtol=GRAD_RTOL,grad_atol=GRAD_ATOL):
    """Provider returns true values, AD Jv/J^T v; damping regularizes steps ONLY."""
    p=project(initial);damping=1.;trace=[];initial_gradient=None;accepted=0;initial_value=None
    for iteration in range(max_outer+1):
        prediction,gradient=provider.linearize(p);residual=prediction-provider.target
        value=.5*dot(residual,residual)
        # Unit-metric projected gradient handles the scalar nonnegative boundary.
        pg=p-project(p-gradient);gn=norm(pg)
        if not all(math.isfinite(v) for v in (value,gn)):raise RuntimeError('Nonfinite working objective/gradient')
        if initial_gradient is None:initial_gradient=gn;initial_value=value
        threshold=grad_atol+grad_rtol*initial_gradient
        if gn<=threshold:
            status='GRADIENT_TOLERANCE';break
        if iteration==max_outer:status='OUTER_BUDGET';break
        succeeded=False
        for trial in range(max_trials):
            step,linear=cg(lambda v:provider.vjp(p,provider.jvp(p,v)),-gradient,damping,cg_max)
            row=dict(iteration=iteration,trial=trial,old_value=value,gradient_norm=gn,damping=damping,cg=linear,accepted=False)
            if not linear['passed']:
                row['reason']='INEXACT_LINEAR_GATE';trace.append(row);damping*=4;continue
            candidate=project(p+step);step=candidate-p
            if norm(step)==0:
                row['reason']='ZERO_STEP_WITH_NONSTATIONARY_GRADIENT';trace.append(row);damping*=4;continue
            response=provider.jvp(p,step);predicted=value-.5*dot(residual+response,residual+response)
            if not math.isfinite(predicted) or predicted<=0:
                row.update(reason='NONPOSITIVE_MODEL_GAIN',predicted_gain=predicted);trace.append(row);damping*=4;continue
            actual_prediction=provider.value(candidate);newvalue=.5*dot(actual_prediction-provider.target,actual_prediction-provider.target)
            actual=value-newvalue;ratio=actual/predicted
            row.update(new_value=newvalue,predicted_gain=predicted,actual_gain=actual,agreement=ratio,step_norm=norm(step))
            if math.isfinite(newvalue) and actual>0 and ratio>.1:
                row['accepted']=True;p=candidate;accepted+=1;succeeded=True
                if ratio>.75:damping=max(1e-12,damping/3)
                elif ratio<.25:damping*=2
                trace.append(row);break
            row['reason']='ACTUAL_GAIN_GATE';trace.append(row);damping*=4
        if not succeeded:status='TRIAL_BUDGET';break
    return p,dict(status=status,converged=status=='GRADIENT_TOLERANCE',accepted_updates=accepted,
        initial_value=initial_value,final_value=value,
        gradient_norm=gn,gradient_threshold=threshold,initial_gradient_norm=initial_gradient,trace=trace,
        objective='0.5*||actual_next_observation - (base_observation+omega*base_residual)||^2',
        note='Local projected stationarity only. Budget exhaustion is not convergence; no GT or best-image selection.')


def budget():
    normal_per_trial=CG_MAX+1  # final true-residual check is an extra product
    jvp=MAX_OUTER*MAX_TRIALS*(normal_per_trial+1)
    vjp=(MAX_OUTER+1)+MAX_OUTER*MAX_TRIALS*normal_per_trial
    value=MAX_OUTER*MAX_TRIALS
    return dict(max_solver_forward=jvp+vjp+value,max_solver_jvp=jvp,max_solver_vjp=vjp,
        cases=12,new_arms=48,shared_prefix_F=12*16,continuation_F=48*14,
        max_new_F=12*16+48*14+24*(jvp+vjp+value),max_new_JVP=24*jvp,max_new_VJP=24*vjp,
        probe_F_VJP_JVP_per_load=[2,1,1],optimizer_backward=0)
