"""Exact zero-padded blur actions and a prospective, bounded CPU solve protocol.

FFT only accelerates the action; it does not diagonalize the cropped normal
operator. No circular boundary substitution and no change to regularization.
"""
import math
import time
import numpy as np


def fast_length(n):
    while True:
        remainder = n
        for p in (2, 3, 5):
            while remainder % p == 0: remainder //= p
        if remainder == 1: return n
        n += 1


class ZeroBlur:
    def __init__(self, kernel, shape):
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.shape = tuple(shape[-2:])
        if (self.kernel.ndim != 2 or any(n % 2 != 1 for n in self.kernel.shape)
                or not np.isfinite(self.kernel).all() or min(self.shape) < 1):
            raise ValueError('Finite odd 2D kernel and positive image dimensions required')
        self.pad = tuple(n//2 for n in self.kernel.shape)
        self.fft_shape = tuple(fast_length(n+k-1) for n, k in zip(self.shape, self.kernel.shape))
        self.spectra = [np.fft.rfft2(k, s=self.fft_shape) for k in (self.kernel[::-1, ::-1], self.kernel)]
        self.calls = [0, 0]

    def apply(self, value, adjoint=False):
        value = np.asarray(value, dtype=np.float64)
        if value.shape[-2:] != self.shape or not np.isfinite(value).all():
            raise ValueError('Input shape or finiteness differs')
        i = int(adjoint); self.calls[i] += 1
        full = np.fft.irfft2(np.fft.rfft2(value, s=self.fft_shape)*self.spectra[i], s=self.fft_shape)
        y, x = self.pad; h, w = self.shape
        return np.ascontiguousarray(full[..., y:y+h, x:x+w])

    def H(self, value): return self.apply(value)
    def transpose(self, value): return self.apply(value, True)


def solve_budget(kernel, eta, rtol=1e-5, ceiling=2048):
    """Exact-arithmetic CG bound, used to set a budget, NEVER to certify a solve.

    For B=.25 HH^T+eta I, kappa<=1+.25||kernel||_1^2/eta.
    ||r_k||/||r_0|| <= 2 sqrt(kappa) ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k.
    Floating-point and implementation errors require the independent true check.
    """
    if not math.isfinite(eta) or eta <= 0 or not 0 < rtol < 1:
        raise ValueError('Positive eta and valid tolerance required')
    kappa = 1+.25*float(np.abs(kernel).sum())**2/eta
    root = math.sqrt(kappa); rho = (root-1)/(root+1)
    steps = 1 if rho == 0 else max(1, math.ceil(math.log(rtol/(2*root))/math.log(rho)))
    return dict(kappa_upper=kappa, contraction_bound=rho, required_iterations=steps,
                maximum_iterations=steps if steps <= ceiling else None, hard_ceiling=ceiling,
                bound_scope='Exact arithmetic budget only; actual original-operator residual certifies')


def cg(action, rhs, maximum, *, rtol=1e-5, seconds_limit=300, progress=None):
    rhs = np.asarray(rhs, dtype=np.float64)
    if maximum < 1 or not np.isfinite(rhs).all(): raise ValueError('Invalid solve contract')
    dot = lambda a, b: float(np.sum(a*b))
    start = time.monotonic(); calls = checks = restarts = 0
    x = np.zeros_like(rhs); residual = rhs.copy(); direction = residual.copy()
    rr = dot(residual, residual); scale = math.sqrt(rr)
    if scale == 0:
        return x, dict(passed=True, iterations=0, operator_calls=0, relative_residual=0., seconds=0., reason='ZERO_RHS')
    relative = 1.
    for step in range(1, maximum+1):
        if time.monotonic()-start > seconds_limit:
            reason = 'WALLTIME_BUDGET_EXHAUSTED'; break
        ap = action(direction); calls += 1; curvature = dot(direction, ap)
        if not math.isfinite(curvature) or curvature <= 0: raise ValueError('Non-SPD or invalid action')
        alpha = rr/curvature; x += alpha*direction; residual -= alpha*ap
        new_rr = dot(residual, residual)
        if not math.isfinite(new_rr): raise ValueError('Nonfinite residual')
        if math.sqrt(new_rr) <= rtol*scale or step == maximum:
            residual = rhs-action(x); calls += 1; checks += 1
            new_rr = dot(residual, residual); relative = math.sqrt(new_rr)/scale
            if relative <= rtol:
                reason = 'FFT_TRUE_RESIDUAL_PASSED'; break
            if step == maximum:
                reason = 'ITERATION_BUDGET_EXHAUSTED'; break
            # A spurious recursive convergence uses only the remaining budget.
            direction = residual.copy(); rr = new_rr; restarts += 1
        else:
            direction = residual+(new_rr/rr)*direction; rr = new_rr
        if progress and step % 100 == 0:
            progress(dict(iterations=step, maximum=maximum, operator_calls=calls,
                          recursive_relative_residual=math.sqrt(new_rr)/scale))
    else: raise AssertionError('Unreachable')
    actual = rhs-action(x); calls += 1; checks += 1
    relative = math.sqrt(dot(actual, actual))/scale
    return x, dict(passed=bool(relative <= rtol), iterations=step, operator_calls=calls,
        relative_residual=relative, true_checks=checks, restarts=restarts,
        seconds=time.monotonic()-start, reason=reason,
        stopped_for_budget=reason.endswith('BUDGET_EXHAUSTED'))
