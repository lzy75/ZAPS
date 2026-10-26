"""A55 fixed-grid direction ablation with the same state-dependent norm rule."""
import math
import numpy as np
from utils import current_score_runtime_r2 as base
from utils import current_score_geometry as algebra

ARMS = ('V_PHYSICAL', 'Q_ORIGINAL')


def numpy_target(q, v, coefficient, mode):
    q, v = (np.asarray(a, dtype=np.float64) for a in (q, v))
    if mode not in ARMS or q.shape != v.shape or not q.size:
        raise ValueError('Matched finite vectors and known direction required')
    if not np.isfinite(q).all() or not np.isfinite(v).all() or not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Finite vectors and positive coefficient required')
    if mode == 'V_PHYSICAL':
        return coefficient*v
    qnorm = float(np.linalg.norm(q))
    if qnorm == 0:
        raise ValueError('Cannot normalize zero original direction')
    return coefficient*float(np.linalg.norm(v))/qnorm*q


def target(q, v, coefficient, mode, reading):
    import torch
    if mode not in ARMS or not reading['resolved']:
        raise ValueError('Resolved known direction required')
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Positive coefficient required')
    result = coefficient*v if mode == 'V_PHYSICAL' else coefficient*reading['v_RMS']/reading['q_RMS']*q
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite direction target; no hidden cap')
    actual = float(result.detach().double().square().mean().sqrt())
    expected = coefficient*reading['v_RMS']
    error = abs(actual-expected)/max(expected, 1e-30)
    if error > 2e-6:
        raise RuntimeError('Direction target violates matched norm rule')
    return result.detach(), dict(target_RMS=actual, expected_target_RMS=expected,
                                  norm_rule_relative_error=error, direction=mode,
                                  current_Jacobian_used=False, backward_calls=0)


def sensed_field(z, y, x, t, next_t, positions, sigma, mode, mask=None):
    import torch
    if mode not in ARMS:
        raise ValueError('Unknown direction')
    # Match A53 graph-enabled/IEEE forward arithmetic, without evaluating a VJP.
    _, raw = base.raw_mean(z.dm, x, t, graph=True)
    field = base.field_from_mean(z, y, raw, t, positions)
    with torch.no_grad():
        residual = y-z.A.H(raw.detach())
        if mask is not None:
            residual = residual*mask.to(dtype=residual.dtype)
        row = dict(R_raw=base.informative_rms(residual, mask)/sigma,
                   R_clean=base.informative_rms(y-z.A.H(field['clean']), mask)/sigma,
                   clip_fraction=float((raw.detach().abs() > 1).float().mean()),
                   backward_calls=0, current_Jacobian_used=False)
    v, gate = algebra.proximal_vector(lambda a: .5*z.A.base.H(a), z.A.transpose, residual,
        float(z.dm.alphas_cumprod[t]), sigma, rtol=base.CG_RTOL, maxiter=base.CG_MAXITER)
    row['cg'] = gate
    if v is None:
        row.update(resolved=False, reason='CG_UNRESOLVED_FALLBACK')
        return field, 1., row
    alpha_s = float(z.dm.alphas_cumprod[next_t]) if next_t >= 0 else 1.
    coefficient = algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]), alpha_s)
    reading = base.gram_readout(field['q'], v, None, coefficient,
                                float(x.detach().double().square().mean().sqrt()))
    row.update(reading, coefficient=coefficient)
    if not reading['resolved']:
        return field, 1., row
    result, extra = target(field['q'], v, coefficient, mode, reading)
    row.update(extra)
    return dict(field, q=result), 1., row
