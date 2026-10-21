"""A34: discrepancy-inspired first crossing of a noise-level target on a response ray."""
import hashlib
import json
import math
from utils.continuous_response_feedback import stage_profile

POLICIES = {'N090': .9, 'N100': 1., 'N110': 1.1}
PAIRS = {'N090': 'TIME090', 'N100': 'TIME100', 'N110': 'TIME110'}
SIGMA = .05


def profile_signature(data, arm):
    stage = PAIRS[arm]
    bound = dict(plan_sha256=data['plan_sha256'], arm=arm, GT_used=False,
                 aggregation=data['aggregation'],
                 values={k: v[stage] for k, v in data['values'].items() if stage in v},
                 sources={k: v for k, v in data['sources'].items() if k.endswith('/'+arm)})
    return hashlib.sha256(json.dumps(bound, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select(dot, r2, a2, count, target=1., sigma=SIGMA):
    if (not all(math.isfinite(v) for v in (dot, r2, a2, target, sigma))
            or r2 < 0 or a2 < 0 or count <= 0 or target <= 0 or sigma <= 0):
        raise ValueError('Invalid discrepancy input')
    R = math.sqrt(r2/count)/sigma
    c = max(-1., min(1., dot/math.sqrt(r2)/math.sqrt(a2))) if r2 > 1e-24 and a2 > 1e-24 else None
    b = math.sqrt(a2/r2) if r2 > 1e-24 else None
    raw = max(0., c/b) if c is not None else None
    fallback = False
    if R <= target:
        gamma, reason = 0., 'ALREADY_WITHIN_NOISE_TARGET'
    elif c is None:
        gamma, reason, fallback = 1., 'UNRESOLVED_RESPONSE_KEEP_ORIGINAL', True
    elif c <= 0:
        gamma, reason = 0., 'NO_DESCENT_ALIGNMENT'
    else:
        ratio = target/R
        discriminant = c*c-1.+ratio*ratio
        if discriminant < 0:
            gamma, reason = raw, 'TARGET_UNREACHABLE_USE_RAY_MINIMUM'
        else:
            # Smaller root; stable even when the target is very close to current R.
            gamma = (1.-ratio*ratio)/(b*(c+math.sqrt(discriminant)))
            reason = 'FIRST_NOISE_TARGET_CROSSING'
    predicted = R*math.sqrt(max(0., 1.-c*c)+(b*gamma-c)**2) if c is not None else R
    if not math.isfinite(gamma) or gamma < 0 or not math.isfinite(predicted):
        raise ValueError('Nonfinite discrepancy decision')
    return dict(gamma=gamma, raw_gamma=raw, R=R, cosine=c, relative_response=b,
                R_target=target, predicted_R_after=predicted, reason=reason,
                resolved=c is not None, cap_hit=False, fallback=fallback)


def sensor(z, y, clean, q, t, target):
    hz = z.A.H(clean)
    r = (y-hz).double()
    v = q/z.dm.alphas_cumprod[t].sqrt().clamp(min=1e-8)
    a = (z.A.H(clean+v)-hz).double()
    result = select(float((r*a).sum()), float(r.square().sum()),
                    float(a.square().sum()), r.numel(), target)
    q_rms = float(q.double().square().mean().sqrt())
    result.update(qRMS=q_rms, guidance_rms=result['gamma']*q_rms)
    return result
