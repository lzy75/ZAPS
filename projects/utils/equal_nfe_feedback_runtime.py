"""A101 controllers: exactly the native 300 denoiser calls in every arm."""
import math
from utils import equal_nfe_feedback as rules
from utils import noise_signal_shrink as scalar_rule
from utils import paper_clock_runtime as clock


def _dot(a, b):
    return float((a.detach().double() * b.detach().double()).sum())


def _norm(a):
    return math.sqrt(max(0., _dot(a, a)))


def _informative(value, operator):
    mask = getattr(operator, '_mask_cache', None)
    return value.reshape(-1) if mask is None else value[mask.bool().expand_as(value)]


def _observed_tensor(vector, template, operator):
    """Adjoint of _informative: zeros outside an inpainting mask."""
    import torch
    mask = getattr(operator, '_mask_cache', None)
    if mask is None:
        return vector.reshape_as(template)
    result = torch.zeros_like(template)
    result[mask.bool().expand_as(template)] = vector.reshape(-1).to(result.dtype)
    return result


def diagonal_update(z, operator, clean, q, residual, alpha, sigma, counts):
    """One exact steepest-descent step for the frozen linear observation proxy.

    The scalar solution is a feasible initializer.  All work is H/H^T and db4;
    this function never calls the denoiser.
    """
    import torch
    scale = alpha.sqrt().clamp(min=1e-8)
    with torch.no_grad():
        base = operator.H(clean)
        r = _informative(residual, operator).double()
        native_response = _informative(operator.H(clean + q / scale) - base, operator).double()
        scalar = scalar_rule.select(_dot(r, native_response), _dot(r, r),
                                    _dot(native_response, native_response), r.numel(), sigma)
        omega = scalar['weight']
        counts['module_dwt_analysis'] += 1
        coefficients = z.dwt.analysis(q)
        support = coefficients != 0
        p0 = coefficients * scalar['gamma']

        counts['module_dwt_synthesis'] += 1
        delta0 = z.dwt.synthesis(p0 * support)
        response0 = _informative(operator.H(clean + delta0 / scale) - base, operator).double()
        error = response0 - omega * r
        old_value = .5 * _dot(error, error)

        full_error = _observed_tensor(error, base, operator)
        image_gradient = operator.transpose(full_error.to(base.dtype)) / scale
        counts['module_adjoint_actions'] += 1
        counts['module_dwt_analysis'] += 1
        gradient = z.dwt.analysis(image_gradient) * support
        numerator = _dot(gradient, gradient)
        if numerator <= 0:
            return delta0, dict(status='ZERO_PROXY_GRADIENT', scalar=scalar,
                                proxy_initial=old_value, proxy_final=old_value,
                                eta=0., correction_norm=_norm(delta0)), counts

        counts['module_dwt_synthesis'] += 1
        gradient_image = z.dwt.synthesis(gradient)
        gradient_response = _informative(operator.H(clean + gradient_image / scale) - base,
                                         operator).double()
        denominator = _dot(gradient_response, gradient_response)
        if denominator <= 0 or not math.isfinite(denominator):
            return delta0, dict(status='ZERO_PROXY_CURVATURE', scalar=scalar,
                                proxy_initial=old_value, proxy_final=old_value,
                                eta=0., correction_norm=_norm(delta0)), counts

        eta = numerator / denominator
        candidate = (p0 - eta * gradient) * support
        counts['module_dwt_synthesis'] += 1
        delta = z.dwt.synthesis(candidate)
        new_response = _informative(operator.H(clean + delta / scale) - base, operator).double()
        new_error = new_response - omega * r
        new_value = .5 * _dot(new_error, new_error)
        tolerance = 128 * torch.finfo(clean.dtype).eps * max(1., old_value)
        if not math.isfinite(new_value) or new_value > old_value + tolerance:
            raise RuntimeError('Closed diagonal proxy failed its exact quadratic descent gate')
        qn = _norm(q); dn = _norm(delta)
        return delta, dict(
            status='CLOSED_DIAGONAL_ACCEPTED', scalar=scalar,
            proxy_initial=old_value, proxy_final=new_value,
            proxy_gain=old_value - new_value, eta=eta,
            native_norm=qn, correction_norm=dn,
            cosine_to_native=_dot(delta, q) / max(dn * qn, 1e-30),
            negative_coefficient_fraction=float((candidate[support] * coefficients[support] < 0).float().mean()),
        ), counts


def make_operator(case, device, count=None, size=256, kernel=None):
    from utils.shrink_multitask_runtime import make_operator as original
    return original(case, device, count=count, size=size, kernel=kernel)


def controller_class():
    import torch
    from modules.zaps_algorithm import ZAPS, ddpm_posterior_step

    class Controlled(ZAPS):
        def _reverse_diffusion(self, y, nfe_counter=None, eta_override=None,
                               init_noise=None, record_indicators=False):
            self.unrolls = getattr(self, 'unrolls', 0) + 1
            if self.unrolls != 10:
                return super()._reverse_diffusion(y, nfe_counter, eta_override,
                                                  init_noise, record_indicators)
            self.entry = clock.snapshot(self, init_noise)
            self.trace = []
            self.module_counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0,
                                      module_adjoint_actions=0)
            if self.arm == 'ZAPS_LOCAL_300':
                return super()._reverse_diffusion(y, nfe_counter, eta_override,
                                                  init_noise, record_indicators)
            x = init_noise.to(self.device)
            ab = self.dm.alphas_cumprod
            for k, i in enumerate(range(len(self.tau) - 1, -1, -1)):
                t = int(self.tau[i]); s = int(self.tau[i - 1]) if i else -1
                with torch.no_grad():
                    eps = self.dm._predict_eps(
                        x, torch.full((y.shape[0],), t, device=self.device, dtype=torch.long))
                if nfe_counter is not None:
                    nfe_counter[0] += 1
                clean = self._tweedie_estimate(x, eps, ab[t], i)
                uncond = ddpm_posterior_step(
                    x, clean, t, s, ab,
                    eta=self.eta if eta_override is None else eta_override,
                    learned_log_var=None, mode='ddpm')
                residual = y - self.A.H(clean)
                v = self.A.transpose(residual)
                hv = self.dwt.synthesis(self.D[i] * self.dwt.analysis(v))
                q = self.zeta[i] * ((v + (1 - ab[t]) * hv) /
                                    ab[t].sqrt().clamp(min=1e-8))
                if self.arm == 'CLOSED_SCALAR_300':
                    with torch.no_grad():
                        base = self.A.H(clean)
                        r = _informative(residual, self.A).double()
                        d = _informative(self.A.H(clean + q / ab[t].sqrt().clamp(min=1e-8)) - base,
                                         self.A).double()
                        scalar = scalar_rule.select(_dot(r, d), _dot(r, r), _dot(d, d),
                                                    r.numel(), self.sigma)
                    delta = scalar['gamma'] * q
                    detail = dict(status='CLOSED_SCALAR', scalar=scalar,
                                  native_norm=_norm(q), correction_norm=_norm(delta),
                                  cosine_to_native=1. if _norm(delta) else None)
                elif self.arm == 'CLOSED_DIAG_300':
                    delta, detail, self.module_counts = diagonal_update(
                        self, self.A, clean, q, residual, ab[t], self.sigma,
                        self.module_counts)
                else:
                    raise ValueError('Unknown A101 arm')
                x = uncond + delta
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite A101 output; no fallback')
                self.trace.append(dict(step=k, t=t, next_t=s, **detail))
            return x
    return Controlled


def execute(model, operator, y, case, arm, size=256, level=3):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(case['seed'])
    z = controller_class()(model, operator, **rules.configuration(case['task'], size, level))
    z.arm, z.sigma = arm, case['sigma']
    loss = z.optimize(y, verbose=False, x0_gt=None)
    return z._last_opt_x0.detach().cpu(), dict(
        entry=z.entry, trace=z.trace, loss=loss,
        module_counts=getattr(z, 'module_counts', {}))
