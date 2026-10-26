"""A58: periodic FFT preconditioner for the UNCHANGED zero-boundary blur system."""
import math


def periodic_inverse(kernel, shape, eta):
    """P=.25 CC^T+eta I, where C alone has periodic boundary conditions.

    Caller must still supply the original zero-boundary target action to PCG.
    Kernel coefficients are promoted, never regenerated at a different precision.
    """
    import torch
    if kernel.ndim != 4 or kernel.shape[:2] != (1, 1) or eta <= 0 or not math.isfinite(eta):
        raise ValueError('One finite convolution kernel and positive eta required')
    h, w = shape[-2:]; kh, kw = kernel.shape[-2:]
    if kh > h or kw > w or not torch.isfinite(kernel).all():
        raise ValueError('Finite kernel must fit image')
    embedded = torch.zeros((h, w), device=kernel.device, dtype=torch.float64)
    embedded[:kh, :kw] = kernel[0, 0].double()
    embedded = torch.roll(embedded, shifts=(-(kh//2), -(kw//2)), dims=(-2, -1))
    spectrum = .25*torch.fft.rfft2(embedded).abs().square()+eta
    def inverse(rhs):
        if tuple(rhs.shape[-2:]) != (h, w) or rhs.dtype != torch.float64:
            raise ValueError('Preconditioner requires matching float64 RHS')
        return torch.fft.irfft2(torch.fft.rfft2(rhs)/spectrum, s=(h, w))
    return inverse


def preconditioned_cg(action, rhs, inverse, *, rtol=1e-5, maxiter=128):
    """True-residual certified PCG; every attempted action and FFT inverse counted.

    If recursive convergence is spurious, restart within the original cap.
    No residual criterion, target, regularization or iteration budget is relaxed.
    """
    import torch
    if (rhs.dtype != torch.float64 or not rhs.numel() or not torch.isfinite(rhs).all()
            or not 0 < rtol < 1 or not isinstance(maxiter, int) or maxiter < 1):
        raise ValueError('Finite double RHS and valid fixed solve contract required')
    dot = lambda a, b: float((a*b).sum())
    calls = pcalls = checks = restarts = 0
    with torch.no_grad():
        x = torch.zeros_like(rhs); r = rhs.clone(); scale = math.sqrt(dot(r, r))
        def record(passed, step, relative, reason):
            return dict(passed=bool(passed), iterations=step, operator_calls=calls,
                preconditioner_calls=pcalls, true_checks=checks, restarts=restarts,
                relative_residual=relative, reason=reason, arithmetic='float64', rtol=rtol, maxiter=maxiter)
        if scale == 0:
            return x, record(True, 0, 0., 'ZERO_RHS')
        z = inverse(r); pcalls += 1; direction = z.clone(); rz = dot(r, z)
        if not math.isfinite(rz) or rz <= 0:
            raise ValueError('Preconditioner failed SPD gate')
        for step in range(1, maxiter+1):
            ap = action(direction); calls += 1
            if ap.shape != rhs.shape or not torch.isfinite(ap).all():
                raise ValueError('Invalid original target action')
            pap = dot(direction, ap)
            if not math.isfinite(pap) or pap <= 0:
                raise ValueError('Original target failed SPD gate')
            alpha = rz/pap; x.add_(direction, alpha=alpha); r.sub_(ap, alpha=alpha)
            apparent = math.sqrt(max(0., dot(r, r))) <= rtol*scale
            if apparent or step == maxiter:
                r = rhs-action(x); calls += 1; checks += 1
                relative = math.sqrt(max(0., dot(r, r)))/scale
                if relative <= rtol:
                    return x, record(True, step, relative, 'TRUE_RESIDUAL_PASSED')
                if step == maxiter:
                    return x, record(False, step, relative, 'ITERATION_BUDGET_EXHAUSTED')
                z = inverse(r); pcalls += 1; direction = z.clone(); rz = dot(r, z); restarts += 1
            else:
                z = inverse(r); pcalls += 1; new_rz = dot(r, z)
                if not math.isfinite(new_rz) or new_rz <= 0:
                    raise ValueError('Nonpositive preconditioned residual product')
                direction = z+(new_rz/rz)*direction; rz = new_rz
            if not math.isfinite(rz) or rz <= 0:
                raise ValueError('Preconditioner restart failed SPD gate')
        raise AssertionError('Unreachable fixed iteration cap')


def proximal_vector(base64, residual, alpha_t, sigma_y, *, rtol=1e-5, maxiter=128):
    """The SAME .25 HH^T+eta I target with promoted frozen kernel coefficients."""
    import torch
    alpha_t, sigma_y = float(alpha_t), float(sigma_y)
    if not 0 < alpha_t < 1 or sigma_y <= 0 or not math.isfinite(sigma_y):
        raise ValueError('Known positive noise/alpha required')
    a, b2 = math.sqrt(alpha_t), 1-alpha_t; eta = sigma_y*sigma_y*a/b2
    inverse = periodic_inverse(base64.kernel, residual.shape, eta)
    action = lambda x: .25*base64.H(base64.transpose(x))+eta*x
    with torch.no_grad():
        w, gate = preconditioned_cg(action, residual.detach().double(), inverse, rtol=rtol, maxiter=maxiter)
        v = ((.5*a/b2)*base64.transpose(w)).to(residual.dtype) if gate['passed'] else None
        if v is not None and not torch.isfinite(v).all():
            raise ValueError('Nonfinite cast target cotangent')
    return v, dict(**gate, eta=eta, conditional_variance_proxy=b2/a,
        linear_calls=gate['operator_calls'], adjoint_calls=gate['operator_calls']+int(v is not None),
        spectral_setup_ffts=1, fft_transforms=1+2*gate['preconditioner_calls'],
        target_boundary='original_zero', preconditioner_boundary='periodic_only',
        kernel_coefficients='original_float32_promoted_not_regenerated', cotangent_dtype=str(residual.dtype))
