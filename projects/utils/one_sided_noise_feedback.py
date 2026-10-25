"""A38 mechanism ablation: regularize only one side of the current LS recommendation."""
from utils import continuous_response_feedback as old
from utils import noise_regularized_feedback as regularized

ARMS = ('UP_REG', 'DOWN_REG')
CALLS = 60


def select(dot, r2, a2, count, sigma, arm):
    if arm not in ARMS:
        raise ValueError('Unknown one-sided ablation')
    original = old.select(dot, r2, a2, 2.)
    constrained = regularized.select(dot, r2, a2, count, sigma)
    raw = original['raw_gamma']
    apply = raw is not None and (raw > 1. if arm == 'UP_REG' else raw < 1.)
    gamma = constrained['gamma'] if apply else original['gamma']
    return dict(gamma=gamma, gamma_LS=raw, original_gamma=original['gamma'],
                regularized_gamma=constrained['gamma'], trust_weight=constrained['trust_weight'],
                selected_component='REG' if apply else 'OLD',
                changed_from_OLD=abs(gamma-original['gamma']) > 1e-12,
                cosine=original['cosine'], relative_response=original['relative_response'],
                resolved=original['resolved'])


def sensor(z, y, clean, q, t, sigma, arm):
    hz = z.A.H(clean)
    r = (y-hz).double()
    v = q/z.dm.alphas_cumprod[t].sqrt().clamp(min=1e-8)
    a = (z.A.H(clean+v)-hz).double()
    result = select(float((r*a).sum()), float(r.square().sum()), float(a.square().sum()),
                    r.numel(), sigma, arm)
    result.update(R=float(r.square().mean().sqrt()/sigma),
                  qRMS=float(q.double().square().mean().sqrt()))
    return result
