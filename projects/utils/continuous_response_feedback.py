"""A33: continuous, GT-free scaling of the A32 identity-response surrogate."""
import math
import statistics
import hashlib
import json

POLICIES = {'I_cont2': 2., 'I_cont8': 8., 'I_free': None}
PAIRS = {name: name.replace('I_', 'STAGE_', 1) for name in POLICIES}
MIN_IMPROVEMENT = .01


def profile_signature(data, arm):
    """Bind each range independently so completing another range cannot invalidate it."""
    stage = PAIRS[arm]
    bound = dict(plan_sha256=data['plan_sha256'], arm=arm, GT_used=False,
                 aggregation=data['aggregation'],
                 values={key: values[stage] for key, values in data['values'].items() if stage in values},
                 sources={key: value for key, value in data['sources'].items() if key.endswith('/'+arm)})
    return hashlib.sha256(json.dumps(bound, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select(dot, r2, a2, upper=None):
    if not all(math.isfinite(v) for v in (dot, r2, a2)) or r2 < 0 or a2 < 0:
        raise ValueError('Invalid response quadratic')
    if upper is not None and (not math.isfinite(upper) or upper < 1.):
        raise ValueError('Upper bound must include the original gamma=1')
    if r2 <= 1e-24 or a2 <= 1e-24:
        return dict(gamma=1., raw_gamma=None, cosine=None, relative_response=None,
                    predicted_improvement=0., resolved=False, cap_hit=False, fallback=True)
    b = math.sqrt(a2 / r2)
    c = max(-1., min(1., dot / math.sqrt(r2) / math.sqrt(a2)))
    raw = c / b
    candidate = max(0., raw)
    capped = upper is not None and candidate > upper
    if upper is not None:
        candidate = min(candidate, upper)
    # Normalized equivalent of .5||r-gamma*a||^2, avoiding cancellation at its minimum.
    def loss(g):
        return .5 * (max(0., 1.-c*c) + (b*g-c)**2)
    gain = (loss(1.)-loss(candidate)) / max(loss(1.), .01, 1e-24)
    fallback = gain < MIN_IMPROVEMENT
    gamma = 1. if fallback else candidate
    if not all(math.isfinite(v) for v in (b, raw, gain, gamma)):
        raise ValueError('Nonfinite continuous decision; no silent fallback')
    return dict(gamma=gamma, raw_gamma=raw, cosine=c, relative_response=b,
                predicted_improvement=gain, resolved=True, cap_hit=bool(capped),
                fallback=fallback)


def stage_profile(profiles, image, arm):
    """Median of actual feedback trajectories; exclude BOTH seeds of the held image."""
    other = [v['traces'][arm] for v in profiles.values() if v['image_id'] != image]
    if not other or any(len(rows) != 30 for rows in other):
        raise ValueError('Need complete other-image continuous trajectories')
    values = [[row['gamma'] for row in rows] for rows in other]
    if any(not math.isfinite(g) or g < 0 for rows in values for g in rows):
        raise ValueError('Invalid reference multiplier')
    return [statistics.median(rows[k] for rows in values) for k in range(30)]


def sensor(z, y, clean, q, t, upper):
    r = (y-z.A.H(clean)).double()
    v = q / z.dm.alphas_cumprod[t].sqrt().clamp(min=1e-8)
    a = (z.A.H(clean+v)-z.A.H(clean)).double()
    result = select(float((r*a).sum()), float(r.square().sum()),
                    float(a.square().sum()), upper)
    q_rms = float(q.double().square().mean().sqrt())
    result.update(R=float(r.square().mean().sqrt()/.05), qRMS=q_rms,
                  guidance_rms=result['gamma']*q_rms)
    return result
