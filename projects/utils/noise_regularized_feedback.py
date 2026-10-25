"""A37: fixed noise-energy regularization of the existing scalar guidance response."""
import math
from utils import continuous_response_feedback as old

LAMBDA = 1.
UPPER = 2.
SIGMA_REFERENCE = .05
ARMS = ('REG', 'BLIND', 'TIME_NOISE')


def select(dot, r2, a2, count, sigma, strength=LAMBDA):
    if (not all(math.isfinite(v) for v in (dot,r2,a2,sigma,strength)) or r2 < 0 or a2 < 0
            or type(count) is not int or count <= 0 or sigma <= 0 or strength < 0):
        raise ValueError('Invalid regularized response')
    base = old.select(dot,r2,a2,UPPER)
    penalty = strength*count*sigma*sigma
    resolved = base['resolved']
    trust = a2/(a2+penalty) if a2+penalty > 0 else 0.
    if strength == 0 or not resolved:
        result = dict(base)
    else:
        c,b = base['cosine'],base['relative_response']
        raw = (dot+penalty)/(a2+penalty)
        candidate = max(0.,min(UPPER,raw))
        def objective(g):
            return .5*(max(0.,1.-c*c)+(b*g-c)**2+(penalty/r2)*(g-1.)**2)
        improvement = (objective(1.)-objective(candidate))/max(objective(1.),.01,1e-24)
        fallback = improvement < old.MIN_IMPROVEMENT
        result = dict(gamma=1. if fallback else candidate,raw_gamma=raw,cosine=c,relative_response=b,
                      predicted_improvement=improvement,resolved=True,cap_hit=raw > UPPER,fallback=fallback)
    result.update(gamma_LS=base['raw_gamma'],trust_weight=trust,
                  response_noise_ratio=a2/(count*sigma*sigma),policy_sigma=sigma,
                  policy_R=math.sqrt(r2/count)/sigma,lambda_value=strength)
    if any(isinstance(v,float) and not math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite regularized decision')
    return result


def sensor(z,y,clean,q,t,sigma,assumed_sigma=None):
    hz = z.A.H(clean)
    r = (y-hz).double()
    v = q/z.dm.alphas_cumprod[t].sqrt().clamp(min=1e-8)
    a = (z.A.H(clean+v)-hz).double()
    result = select(float((r*a).sum()),float(r.square().sum()),float(a.square().sum()),r.numel(),
                    sigma if assumed_sigma is None else assumed_sigma)
    result.update(R=float(r.square().mean().sqrt()/sigma),qRMS=float(q.double().square().mean().sqrt()))
    return result


def reference_profiles(records,grid,sigmas):
    """Pooled OLD response quadratics preserve positivity; no target trajectories/GT."""
    expected = {(f'{i:05d}',s) for i in range(20,26) for s in (1000,2000)}
    if len(records) != 12 or {(r['case']['image_id'],r['case']['sampling_seed']) for r in records} != expected:
        raise ValueError('Need twelve original calibration traces')
    if any([r['t'] for r in item['rows']] != list(reversed(grid)) for item in records):
        raise ValueError('Calibration grid differs')
    stats = []
    for k in range(30):
        triples = []
        for item in records:
            row = item['rows'][k];c,b = row['cosine'],row['relative_response']
            if c is None or b is None:raise ValueError('Unresolved old response; no silent interpolation')
            r2 = (row['R']*SIGMA_REFERENCE)**2
            triples.append((c*b*r2,r2,b*b*r2))
        dot,r2,a2 = [sum(v[j] for v in triples)/len(triples) for j in range(3)]
        stats.append(dict(dot_per_element=dot,r2_per_element=r2,a2_per_element=a2))
    values = {str(sigma):[select(v['dot_per_element'],v['r2_per_element'],v['a2_per_element'],1,sigma)
                          for v in stats] for sigma in sigmas}
    return dict(statistics=stats,values=values,GT_used=False,
                scope='Mean response quadratics on old I_cont2 trajectories at sigma=.05, evaluated at known target sigma. '
                      'TIME_NOISE has no target-image state and no target outcome fitting. This is a frozen reference '
                      'response table, not an oracle or a profile collected on new REG trajectories.')


def active_arms(sigma):
    return ('REG','TIME_NOISE') if sigma == SIGMA_REFERENCE else ARMS


def calls(sigma):
    return 30*len(active_arms(sigma))
