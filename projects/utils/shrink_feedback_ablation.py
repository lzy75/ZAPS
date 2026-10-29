"""A72 label-free, leave-one-image-out controls for frozen A71 feedback."""
import math
import statistics
from utils import noise_signal_shrink as original

SELECT = original.select

ARMS = ('NO_NOISE', 'FIXED_SCALAR', 'FIXED_PROFILE', 'MISMATCH')
CHECK = 'REPLAY_CHECK'
EXECUTED = (CHECK,) + ARMS
REFERENCE = ('BASE', 'SHRINK')


def calibrate(cases, curves):
    """Freeze from other-image A71 traces; never inspect quality or GT."""
    result = {}
    for c in cases:
        peers = sorted((v for v in cases if v['task'] == c['task'] and
                        v['sigma'] == c['sigma'] and v['image_id'] != c['image_id']),
                       key=lambda v: v['image_id'])
        if len(peers) != 2:
            raise ValueError('Need exactly two OTHER-image donors')
        donor_curves = [curves[v['key']] for v in peers]
        if any(len(v) != 30 or any(not math.isfinite(x) or x < 0 for x in v)
               for v in donor_curves):
            raise ValueError('Incomplete or invalid donor multiplier curve')
        images = sorted({c['image_id'], *(v['image_id'] for v in peers)})
        donor_id = images[(images.index(c['image_id']) + 1) % len(images)]
        donor = next(v for v in peers if v['image_id'] == donor_id)
        result[c['key']] = dict(
            fixed_scalar=statistics.median(x for v in donor_curves for x in v),
            fixed_profile=[statistics.median(v[k] for v in donor_curves) for k in range(30)],
            mismatch=list(curves[donor['key']]), donor_keys=[v['key'] for v in peers],
            mismatch_donor=donor['key'], excluded_image=c['image_id'], GT_used=False)
    return result


def selector(arm, calibration):
    if arm not in ARMS:
        raise ValueError('Unregistered A72 arm')
    state = dict(calls=0)

    def choose(dot, r2, d2, count, sigma):
        k = state['calls']
        if k >= 30:
            raise ValueError('A72 suffix exceeded frozen30 grid')
        state['calls'] += 1
        row = SELECT(dot, r2, d2, count, sigma)
        if arm == 'NO_NOISE':
            gamma = max(0., dot) / d2 if d2 > 1e-24 and r2 > 1e-24 else 0.
        elif arm == 'FIXED_SCALAR':
            gamma = calibration['fixed_scalar']
        elif arm == 'FIXED_PROFILE':
            gamma = calibration['fixed_profile'][k]
        else:
            gamma = calibration['mismatch'][k]
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError('Nonfinite or negative frozen multiplier; no fallback')
        return dict(row, gamma=float(gamma), reason=arm, applied_noise_correction=arm != 'NO_NOISE',
                    current_state_used_for_gamma=arm == 'NO_NOISE',
                    predicted_residual=max(0., r2 - 2*gamma*dot + gamma*gamma*d2))

    return choose, state
