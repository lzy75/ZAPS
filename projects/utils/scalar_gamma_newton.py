"""A69 projected scalar search; no integer candidates or multiplier upper cap."""
import math
from utils.next_response_bayes import cases

ARMS=('BASE','FIRST_NOISE','REAL_ISO','MODEL_NEWTON','AUTO_NEWTON')
MAX_FULL=dict(BASE=300,FIRST_NOISE=300,REAL_ISO=503,MODEL_NEWTON=387,AUTO_NEWTON=387)
MAX_REPLAY={k:v-270 for k,v in MAX_FULL.items()}
MAX_FORWARD_POINTS=3
MAX_GRADIENT_POINTS=2
MAX_UPDATES=2
RELATIVE_STEP_TOL=.01
ARMIJO=1e-4


def residual(gamma,gradient,curvature):
    return abs(gamma-max(0.,gamma-gradient/curvature))/(1+gamma)


def solve(evaluate,curvature,*,forward_budget=MAX_FORWARD_POINTS,gradient_budget=MAX_GRADIENT_POINTS):
    """evaluate(g,want_gradient,accept_gradient_if)->{value,gradient}.

    Gradient at a trial is requested only if actual Armijo decrease holds.
    A value-only second update may improve the output; its KKT is UNMEASURED.
    This is explicitly a finite-budget local optimizer, not a global oracle.
    """
    if not math.isfinite(curvature) or curvature<=0:raise ValueError('Positive finite curvature required')
    first=evaluate(0.,True,None)
    if first['value'] is None or first['gradient'] is None:raise ValueError('Invalid reference value/gradient')
    if not all(math.isfinite(first[k]) for k in ('value','gradient')):raise ValueError('Nonfinite reference')
    gamma,value,gradient=0.,first['value'],first['gradient']
    points=1;gradients=1;updates=0;trials=[];eta_history=[]
    r=residual(gamma,gradient,curvature);reason='REFERENCE_STATIONARY' if r<=RELATIVE_STEP_TOL else None
    while reason is None and updates<MAX_UPDATES:
        if points>=forward_budget:reason='FORWARD_BUDGET';break
        if gradient is None:reason='GRADIENT_BUDGET_FINAL_KKT_UNMEASURED';break
        direction=max(0.,gamma-gradient/curvature)-gamma
        eta=1.;accepted=False
        while points<forward_budget:
            trial=gamma+eta*direction
            threshold=value+ARMIJO*eta*gradient*direction
            def accept_gradient_if(candidate):
                return candidate is not None and math.isfinite(candidate) and candidate<=threshold
            row=evaluate(trial,gradients<gradient_budget,accept_gradient_if);points+=1
            if row['gradient'] is not None and not math.isfinite(row['gradient']):
                raise ValueError('Nonfinite scalar gradient')
            candidate=row['value'];ok=accept_gradient_if(candidate)
            trials.append(dict(gamma=trial,eta=eta,objective=candidate,armijo_threshold=threshold,accepted=bool(ok)))
            if row['gradient'] is not None:gradients+=1
            if ok:
                gamma,value,gradient=trial,candidate,row['gradient'];updates+=1;eta_history.append(eta);accepted=True
                if gradient is not None:
                    r=residual(gamma,gradient,curvature)
                    if r<=RELATIVE_STEP_TOL:reason='STATIONARITY_TOLERANCE'
                else:r=None;reason='GRADIENT_BUDGET_FINAL_KKT_UNMEASURED'
                break
            eta*=.5
        if not accepted:reason='LINE_SEARCH_BUDGET_NO_ACCEPTED_STEP'
    if reason is None:reason='UPDATE_BUDGET'
    return dict(gamma=gamma,objective=value,reference_objective=first['value'],
                projected_step_residual=r,stationarity_met=reason in ('REFERENCE_STATIONARY','STATIONARITY_TOLERANCE'),
                final_gradient_measured=gradient is not None,stop_reason=reason,
                value_points=points,gradient_points=gradients,updates=updates,trials=trials,etas=eta_history)
