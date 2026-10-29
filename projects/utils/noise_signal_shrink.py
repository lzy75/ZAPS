"""A70 noise-corrected residual shrinkage, on the original ZAPS ray."""
import math
import statistics

ARMS=('BASE','FIRST_NOISE','SHRINK','FIXED_FIRST','MISMATCH_FIRST','FIXED_SHRINK','MISMATCH_SHRINK')
FEEDBACK=ARMS[:3]
CONTROLS=ARMS[3:]
SIGMAS=(.05,.10)
NEW_IMAGES=4

def select(dot,r2,d2,count,sigma):
    if count<=0 or sigma<=0 or min(r2,d2)<0 or not all(math.isfinite(v) for v in (dot,r2,d2,sigma)):
        raise ValueError('Invalid residual/response/noise')
    if dot*dot>r2*d2+1e-9*max(r2*d2,1e-24):raise ValueError('Invalid response Gram')
    noise=count*sigma*sigma
    if d2<=1e-24 or r2<=1e-24:
        return dict(gamma=0.,weight=0.,signal_variance=0.,resolved=False,reason='ZERO_RESPONSE_OR_RESIDUAL',noise_energy=noise)
    weight=max(0.,1-noise/r2)
    gamma=weight*max(0.,dot)/d2
    if not math.isfinite(gamma):raise ValueError('Nonfinite multiplier')
    return dict(gamma=gamma,weight=weight,signal_variance=max(0.,r2/count-sigma*sigma),
                noise_energy=noise,resolved=True,reason='NOISE_CORRECTED_PROJECTION',
                residual_energy=r2,response_energy=d2,residual_response_dot=dot,
                predicted_residual=max(0.,r2-2*gamma*dot+gamma*gamma*d2))

def constant(curves):
    values=[float(v) for curve in curves for v in curve]
    if not values or any(len(c)!=30 for c in curves) or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Need complete finite nonnegative calibration curves')
    return float(statistics.median(values))

def mismatches(cases):
    images=sorted({c['image_id'] for c in cases})
    if len(images)<2:raise ValueError('Need two image clusters for mismatch')
    donor={v:images[(i+1)%len(images)] for i,v in enumerate(images)}
    return {c['key']:next(v['key'] for v in cases if v['image_id']==donor[c['image_id']] and v['sigma']==c['sigma']) for c in cases}
