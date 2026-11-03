"""Noise-corrected ray fit retaining a host's known quadratic correction cost."""
import math
from utils import noise_signal_shrink


def select(dot,r2,d2,count,sigma,prior_energy=0.):
    if not math.isfinite(prior_energy) or prior_energy<0:
        raise ValueError('Host quadratic energy must be finite and nonnegative')
    old=noise_signal_shrink.select(dot,r2,d2,count,sigma)
    denominator=d2+prior_energy
    gamma=(old['weight']*max(0.,dot)/denominator
           if old['resolved'] and denominator>0 else 0.)
    if not math.isfinite(gamma):raise ValueError('Nonfinite host-aware multiplier')
    return dict(old,gamma=gamma,original_projection_gamma=old['gamma'],
                reason='HOST_QUADRATIC_NOISE_PROJECTION',prior_energy=prior_energy,
                host_denominator=denominator,
                stationarity_relative_error=abs(dot-d2-prior_energy)/max(abs(dot)+d2+prior_energy,1e-30),
                gamma_minus_noise_weight=gamma-old['weight'],
                predicted_residual=max(0.,r2-2*gamma*dot+gamma*gamma*d2))
