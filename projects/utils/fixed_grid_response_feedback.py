"""GT-free finite action selection for the A32 measurement-response surrogate."""
import math
GAMMAS = (0., .5, 1., 2.)
ARMS = ('C','S','M','I','STAGE')


def select(dot, r2, a2, magnitude_only=False):
    if not all(math.isfinite(v) for v in (dot,r2,a2)) or r2 < 0 or a2 < 0:
        raise ValueError('Invalid response quadratic')
    if r2 <= 1e-24 or a2 <= 1e-24:
        return dict(gamma=1.,cosine=None,relative_response=None,predicted_improvement=0.,resolved=False)
    cosine=max(-1.,min(1.,dot/math.sqrt(r2*a2)))
    effective=math.sqrt(r2*a2) if magnitude_only else dot
    losses={g:.5*(r2-2*g*effective+g*g*a2) for g in GAMMAS}
    best=min(GAMMAS,key=lambda g:(losses[g],abs(g-1),g))
    improvement=(losses[1.]-losses[best])/max(losses[1.],.01*r2,1e-24)
    gamma=best if improvement >= .01 else 1.
    return dict(gamma=gamma,cosine=cosine,relative_response=math.sqrt(a2/r2),
                predicted_improvement=improvement,resolved=True)


def stage_profile(profiles, image):
    other=[v['S'] for v in profiles.values() if v['image_id'] != image]
    if not other or any(len(v)!=30 for v in other): raise ValueError('Need other-image full profiles')
    return [min(GAMMAS,key=lambda g:(-sum(v[k]['gamma']==g for v in other),abs(g-1),g)) for k in range(30)]


def sensor(z,y,clean,q,t,index,identity=False,magnitude_only=False):
    import torch
    ab=z.dm.alphas_cumprod[t]
    v=q if identity else q+(1-ab)*z.dwt.synthesis(z.D[index]*z.dwt.analysis(q))
    v=v/ab.sqrt().clamp(min=1e-8)
    hz=z.A.H(clean);r=(y-hz).double();a=(z.A.H(clean+v)-hz).double()
    dot=float((r*a).sum());r2=float(r.square().sum());a2=float(a.square().sum())
    result=select(dot,r2,a2,magnitude_only)
    result.update(R=float(r.square().mean().sqrt()/.05),qRMS=float(q.double().square().mean().sqrt()))
    return result
