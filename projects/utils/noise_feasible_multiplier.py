"""A67: scalar control on the ORIGINAL ZAPS ray; no JAPS vector construction."""
import math

ARMS = ('BASE', 'A33', 'FIRST_NOISE', 'NEAREST_NOISE', 'TIME_NEAREST')


def select(dot, r2, d2, count, sigma, arm):
    """Pure scalar algebra. Noise energy is a discrepancy target, not a CI.

    FIRST_NOISE is the bounded first-crossing A34-style control. NEAREST_NOISE
    projects gamma=1 onto the SAME feasible interval, isolating the anchor.
    Infeasible targets use the bounded residual minimizer; unresolved d keeps1.
    """
    if arm not in ARMS[:-1] or count <= 0 or sigma <= 0:
        raise ValueError('Invalid policy/count/noise')
    if not all(math.isfinite(v) for v in (dot, r2, d2, sigma)) or min(r2, d2) < 0:
        raise ValueError('Invalid response Gram values')
    if dot*dot > r2*d2 + 1e-10*max(r2*d2, 1e-24):
        raise ValueError('Invalid response Cauchy bound')
    energy = count*sigma*sigma
    row = dict(gamma=1., noise_energy=energy, resolved=False,
               residual_before=r2, reason='UNRESOLVED_KEEP_ORIGINAL')
    if arm == 'BASE':
        row.update(resolved=True, reason='ORIGINAL'); return row
    if d2 <= 1e-24 or r2 <= 1e-24:
        return row
    center = dot/d2
    optimum = min(2., max(0., center))
    def residual(g):
        return max(0., r2-2*g*dot+g*g*d2)
    row.update(resolved=True, least_squares_gamma=center,
               residual_at_original=residual(1.))
    if arm == 'A33':
        # Exact A33 select() denominator: normalized loss(1), with .01 floor.
        improvement = (residual(1.)-residual(optimum))/max(residual(1.), .02*r2, 1e-24)
        gamma = optimum if improvement >= .01 else 1.
        row.update(predicted_improvement=improvement, reason='A33_LS' if gamma == optimum else 'A33_ONE_PERCENT_KEEP')
    else:
        perpendicular = max(0., r2-dot*center)
        discriminant = (energy-perpendicular)/d2
        feasible = discriminant >= 0.
        if feasible:
            radius = math.sqrt(discriminant)
            low, high = max(0., center-radius), min(2., center+radius)
            feasible = low <= high
        if feasible:
            gamma = low if arm == 'FIRST_NOISE' else min(high, max(low, 1.))
            row.update(feasible_interval=[low, high], reason='FIRST_FEASIBLE' if arm == 'FIRST_NOISE' else 'NEAREST_ORIGINAL_FEASIBLE')
        else:
            gamma = optimum
            row['reason'] = 'NO_FEASIBLE_ACTION_USE_BOUNDED_LS'
        row['noise_target_feasible'] = feasible
    row.update(gamma=float(gamma), predicted_residual=residual(gamma),
               predicted_R=math.sqrt(residual(gamma)/count)/sigma)
    return row


def cases(directory):
    """Six OLD strong-reference images; boundaries on the first three only."""
    from pathlib import Path
    result = []
    for index in range(6):
        for sigma in ((.025, .05, .10) if index < 3 else (.05,)):
            image_id = f'{index:05d}'
            result.append(dict(image_id=image_id, image=str(Path(directory)/(image_id+'.png')),
                               seed=1000+index, sigma=sigma,
                               key=f'{image_id}_n{round(sigma*1000):03d}_s{1000+index}'))
    return result
