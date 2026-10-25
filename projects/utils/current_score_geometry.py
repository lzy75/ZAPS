"""A50 preflight primitives, not a deployed controller or posterior guarantee."""
import math
import numpy as np


def skip_coefficient(alpha_t, alpha_s):
    """Conditional-score displacement units; alpha_s=1 is deterministic output."""
    a, b = float(alpha_t), float(alpha_s)
    if not (0. < a < b <= 1.):
        raise ValueError("Require 0<alpha_t<alpha_s<=1")
    ratio = a / b
    return (1. - ratio) / math.sqrt(ratio)


def nonnegative_projection(q, target):
    """Euclidean argmin_{g>=0} ||g*q-target||; no labels or action search."""
    q, target = (np.asarray(x, dtype=np.float64).reshape(-1) for x in (q, target))
    if not len(q) or q.shape != target.shape or not all(np.isfinite(x).all() for x in (q, target)):
        raise ValueError("Finite nonempty matched vectors required")
    qq = float(q @ q)
    qt = float(q @ target)
    if not math.isfinite(qq) or not math.isfinite(qt):
        raise ValueError("Nonfinite projection inner product")
    if qq == 0.:
        return dict(resolved=False, gamma=None, reason="ZERO_Q")
    unconstrained = qt / qq
    if not math.isfinite(unconstrained):
        raise ValueError("Nonfinite projection")
    g = max(0., unconstrained)
    return dict(resolved=True, gamma=g, reason="NONNEGATIVE_LEAST_SQUARES")


def score_targets(q, v, u, coefficient):
    """Three targets at SAME state; policies on different paths must be re-sensed."""
    q, v, u = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (q, v, u)]
    if not len(q) or q.shape != v.shape or q.shape != u.shape:
        raise ValueError("Matched vectors required")
    if not all(np.isfinite(x).all() for x in (q, v, u)) or not math.isfinite(coefficient) or coefficient <= 0.:
        raise ValueError("Finite vectors and positive coefficient required")
    uu, vv, vu = float(u @ u), float(v @ v), float(v @ u)
    if not all(math.isfinite(x) for x in (uu, vv, vu)):
        raise ValueError("Nonfinite geometry inner product")
    aligned = vu / uu if uu > 0. else None
    decisions = {
        "PROX_Q": nonnegative_projection(q, coefficient * v),
        "DIRECT_Q": nonnegative_projection(q, coefficient * u),
        "ALIGN_Q": (nonnegative_projection(q, coefficient * aligned * u)
                    if aligned is not None else dict(resolved=False, gamma=None, reason="ZERO_U")),
    }
    return dict(decisions=decisions, alignment_scale=aligned,
                cosine_v_u=vu/math.sqrt(uu)/math.sqrt(vv) if uu > 0. and vv > 0. else None,
                v_dot_u=vu, negative_alignment=vu < 0.,
                q_RMS=float(np.sqrt(np.mean(q*q))), v_RMS=float(np.sqrt(np.mean(v*v))),
                u_RMS=float(np.sqrt(np.mean(u*u))),
                scope="Algebra only. Projection cancels q's positive scalar amplitude; no quality guarantee.")


def conjugate_gradient(action, rhs, *, rtol=1e-5, maxiter=128):
    """Torch SPD solve; failure remains explicit. Imports Torch only when called."""
    import torch
    if not rhs.numel() or rhs.dtype not in (torch.float32, torch.float64) or not torch.isfinite(rhs).all():
        raise ValueError("Finite float32/64 rhs required")
    if not (0. < rtol < 1.) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("Invalid CG contract")
    dot = lambda a, b: float((a.double()*b.double()).sum())
    with torch.no_grad():
        x = torch.zeros_like(rhs)
        residual, direction = rhs.clone(), rhs.clone()
        rr = dot(residual, residual)
        scale = math.sqrt(rr)
        if scale == 0.:
            return x, dict(passed=True, iterations=0, operator_calls=0, relative_residual=0., reason="ZERO_RHS")
        calls = 0
        for i in range(maxiter):
            ap = action(direction)
            calls += 1
            if ap.shape != rhs.shape or not torch.isfinite(ap).all():
                raise ValueError("Invalid operator output")
            pap = dot(direction, ap)
            if not math.isfinite(pap) or pap <= 0.:
                raise ValueError("Nonpositive CG curvature; SPD/adjoint gate required")
            step = rr / pap
            x.add_(direction, alpha=step)
            residual.sub_(ap, alpha=step)
            new_rr = dot(residual, residual)
            if math.sqrt(new_rr) <= rtol * scale:
                # Never certify from recursive residual alone.
                actual = rhs - action(x)
                calls += 1
                relative = math.sqrt(dot(actual, actual)) / scale
                return x, dict(passed=relative <= rtol, iterations=i+1, operator_calls=calls,
                               relative_residual=relative, reason="TRUE_RESIDUAL_CHECK")
            direction = residual + (new_rr / rr) * direction
            rr = new_rr
        actual = rhs - action(x)
        calls += 1
        relative = math.sqrt(dot(actual, actual)) / scale
        return x, dict(passed=False, iterations=maxiter, operator_calls=calls,
                       relative_residual=relative, reason="ITERATION_BUDGET_EXHAUSTED")


def proximal_vector(linear, adjoint, residual, alpha_t, sigma_y, *, rtol=1e-5, maxiter=128):
    """Unit-domain affine offset is in residual; linear/adjoint must exclude it."""
    import torch
    alpha_t, sigma_y = float(alpha_t), float(sigma_y)
    if not (0. < alpha_t < 1.) or not math.isfinite(sigma_y) or sigma_y <= 0.:
        raise ValueError("Positive known measurement noise and valid alpha required")
    a, b2 = math.sqrt(alpha_t), 1. - alpha_t
    eta = sigma_y*sigma_y*a/b2
    with torch.no_grad():
        w, gate = conjugate_gradient(lambda z: linear(adjoint(z)) + eta*z,
                                     residual.detach(), rtol=rtol, maxiter=maxiter)
        v = (a/b2)*adjoint(w) if gate["passed"] else None
        if v is not None and not torch.isfinite(v).all():
            raise ValueError("Nonfinite proximal vector")
    return v, dict(**gate, eta=eta, conditional_variance_proxy=b2/a,
                   linear_calls=gate["operator_calls"],
                   adjoint_calls=gate["operator_calls"]+int(v is not None))


def input_vjp(mean, inputs, vector, *, retain_graph=False):
    """True VJP with detached cotangent; no parameter .grad accumulation."""
    import torch
    if mean.shape != vector.shape or not inputs.requires_grad or not mean.requires_grad:
        raise ValueError("Graph-bearing mean/input and matched cotangent required")
    if not torch.isfinite(vector).all():
        raise ValueError("Nonfinite VJP cotangent")
    result = torch.autograd.grad(mean, inputs, grad_outputs=vector.detach(),
                                 create_graph=False, retain_graph=retain_graph)[0]
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite input VJP")
    return result.detach()
