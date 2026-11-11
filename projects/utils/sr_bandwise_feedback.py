"""Noise-aware measurement target shared by ray and diagonal SR feedback."""
import math

import numpy as np

from utils.directional_reliability import select as weights_from_energy, subbands
from utils.noise_signal_shrink import select as original_scalar
from utils.paper_clock_alignment import require


def band_weights(energy, sizes, removed, sigma):
    removed = np.asarray(removed, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    require(removed.shape == sizes.shape and np.isfinite(removed).all()
            and (removed >= 0).all() and removed.sum() <= 1+1e-5 and (sizes >= 1).all()
            and math.isfinite(sigma) and sigma > 0, 'Invalid band noise moments')
    noise = sigma**2 * np.maximum(0., sizes-removed)
    return weights_from_energy(energy, noise), noise


def measurement_target(wavelet, residual, response, gamma, fitted, sigma, counts):
    """Return the de-noised part of the residual left outside the fitted ray."""
    import torch

    remaining = residual - gamma*response
    coeff = wavelet.analysis(remaining)
    ray = wavelet.analysis(response/response.double().norm()) if fitted else None
    counts['measurement_dwt_analysis'] = counts.get('measurement_dwt_analysis', 0)+1+int(fitted)
    bands = subbands(*residual.shape[-2:], wavelet.level)
    energy, removed, sizes = [], [], []
    for _, (ys, xs) in bands:
        c = coeff[..., ys, xs]
        energy.append(c.double().square().sum())
        removed.append(ray[..., ys, xs].double().square().sum() if fitted else c.new_zeros(()))
        sizes.append(c.numel())
    weights, noise = band_weights(torch.stack(energy).cpu().numpy(), sizes,
                                  torch.stack(removed).cpu().numpy(), sigma)
    detail = dict(weights={band[0]:float(w) for band,w in zip(bands, weights)},
                  nominal_noise_energy=float(noise.sum()))
    if not weights.any():
        return torch.zeros_like(residual), detail
    for (_, (ys, xs)), weight in zip(bands, weights):
        coeff[..., ys, xs] *= float(weight)
    counts['measurement_dwt_synthesis'] = counts.get('measurement_dwt_synthesis', 0)+1
    return wavelet.synthesis(coeff), detail


def controller(use_diagonal):
    def update(zaps, operator, clean, q, residual, alpha, sigma, counts):
        import torch

        def dot(a, b):
            return float((a.double()*b.double()).sum())

        def fallback(gamma, detail):
            with torch.enable_grad():
                return gamma*q, detail, counts

        with torch.no_grad():
            a = alpha.sqrt().clamp(min=1e-8)
            legacy = operator.H(clean+q/a)-operator.H(clean)
            original = original_scalar(dot(residual, legacy), dot(residual, residual),
                                       dot(legacy, legacy), residual.numel(), sigma)
            gamma = original['gamma']
            detail = dict(scalar=original, ray_gamma=gamma, proxy_at_original=None,
                          proxy_at_ray=None, proxy_at_selected=None, weights={}, reason='ORIGINAL_OMEGA_ZERO')
            if original['weight'] == 0:
                return fallback(gamma, detail)
            d = operator.H(q/a)
            d2 = dot(d, d)
            if d2 <= 1e-24:
                detail['reason'] = 'ZERO_LINEAR_RESPONSE'
                return fallback(gamma, detail)
            initial_projection = max(0., dot(residual, d))/d2
            increment, bands = measurement_target(zaps.dwt, residual, d, initial_projection,
                                                  initial_projection > 0, sigma, counts)
            detail.update(bands)
            if not any(detail['weights'].values()):
                detail['reason'] = 'NO_BAND_SIGNAL'
                return fallback(gamma, detail)
            target = gamma*d.double()+increment.double()
            gamma_ray = max(0., dot(target, d)/d2)
            anchor = gamma_ray*q
            e = operator.H(anchor/a).double()-target
            e_original = operator.H(gamma*q/a).double()-target
            loss_original = .5*dot(e_original, e_original)
            loss_ray = .5*dot(e, e)
            value = anchor
            loss_selected = loss_ray
            detail.update(ray_gamma=gamma_ray, proxy_at_original=loss_original, proxy_at_ray=loss_ray,
                          reason='BAND_TARGET_RAY')
            if use_diagonal and loss_ray > 0:
                support = zaps.dwt.analysis(q) != 0
                adjoint = operator.base.transpose(e.to(q.dtype))/a
                gradient = zaps.dwt.analysis(adjoint)*support
                counts['module_dwt_analysis'] += 2
                counts['module_adjoint_actions'] += 1
                norm2 = dot(gradient, gradient)
                if norm2 > 0:
                    image = zaps.dwt.synthesis(gradient)
                    counts['module_dwt_synthesis'] += 1
                    response = operator.H(image/a)
                    curvature = dot(response, response)
                    require(curvature > 0 and math.isfinite(curvature), 'Invalid band target curvature')
                    eta = norm2/curvature
                    value = anchor-eta*image
                    error = operator.H(value/a).double()-target
                    loss_selected = .5*dot(error, error)
                    detail.update(reason='BAND_TARGET_DIAGONAL', eta=eta)
            detail['proxy_at_selected'] = loss_selected
            tol = 128*torch.finfo(q.dtype).eps
            require(bool(torch.isfinite(value).all()) and math.isfinite(loss_selected)
                    and loss_ray <= loss_original+tol*max(1., loss_original)
                    and loss_selected <= loss_ray+tol*max(1., loss_ray),
                    'Band target descent gate failed')
        return value+(q-q.detach()), detail, counts
    return update
