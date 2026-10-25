"""A50 single-forward sensor plumbing. No driver, policy launch, or efficacy claim."""
from contextlib import contextmanager
import math
from utils import current_score_geometry as algebra

MODES = ("PROX_Q", "DIRECT_Q", "ALIGN_Q")
CG_RTOL = 1e-5
CG_MAXITER = 128
NUMERICAL_ULPS = 32
VJP_FD_SCALES = (1e-3, 2e-3)


@contextmanager
def frozen_model_parameters(model):
    """Weights unchanged; restore requires_grad flags even on failure."""
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    if any(p.grad is not None for p in parameters):
        raise ValueError("Preexisting model gradients are not overwritten")
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
        if any(p.grad is not None for p in parameters):
            raise RuntimeError("Input-only VJP populated a model gradient")
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def raw_mean(dm, x, t, *, graph):
    """Exactly one eps forward; detached input prevents cross-step backprop."""
    import torch
    with torch.set_grad_enabled(graph):
        inp = x.detach().requires_grad_(graph)
        tb = torch.full((inp.shape[0],), int(t), device=inp.device, dtype=torch.long)
        eps = dm._predict_eps(inp, tb)
        ab = dm.alphas_cumprod[t]
        raw = (inp-(1-ab).sqrt()*eps)/ab.sqrt().clamp(min=1e-8)
    if not torch.isfinite(raw).all():
        raise RuntimeError("Nonfinite raw mean")
    return inp, raw


def field_from_mean(z, y, raw, t, positions):
    """Original q arithmetic; no extra UNet and no ζ/D differentiation."""
    import torch
    from utils.diag_fixed_count_state_positions import policy
    with torch.no_grad():
        clean = raw.detach().clamp(-1, 1)
        ab = z.dm.alphas_cumprod[t]
        zeta, diagonal, weights = policy.parameters(z, t, positions)
        res = y-z.A.H(clean)
        vector = z.A.transpose(res)
        hv = z.dwt.synthesis(diagonal*z.dwt.analysis(vector))
        q = zeta*((vector+(1-ab)*hv)/ab.sqrt().clamp(min=1e-8))
        if not torch.isfinite(q).all():
            raise RuntimeError("Nonfinite original q")
    return dict(clean=clean, q=q, weights=weights)


def informative_rms(residual, mask=None):
    import torch
    with torch.no_grad():
        if mask is None:
            selected = residual
        else:
            active = mask.bool().expand_as(residual)
            if not active.any():
                raise ValueError("No informative observations")
            selected = residual[active]
        return float(selected.double().square().mean().sqrt())


def gram_readout(q, v, u, coefficient, x_rms):
    """GPU reductions only; avoid transferring full high-dimensional vectors to NumPy."""
    import torch
    if not math.isfinite(coefficient) or coefficient <= 0.:
        raise ValueError("Positive displacement coefficient required")
    tensors = [q, v] + ([] if u is None else [u])
    if any(t.shape != q.shape or not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Finite matched geometry tensors required")
    with torch.no_grad():
        qq, vv = q.detach().double(), v.detach().double()
        values = [(qq*qq).sum(), (vv*vv).sum(), (qq*vv).sum()]
        if u is not None:
            uu = u.detach().double()
            values += [(uu*uu).sum(), (qq*uu).sum(), (vv*uu).sum()]
        numbers = torch.stack(values).cpu().tolist()
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("Nonfinite Gram matrix")
    q2, v2, qv = numbers[:3]
    n = q.numel()
    qr, vr = math.sqrt(q2/n), math.sqrt(v2/n)
    floor_q = NUMERICAL_ULPS*torch.finfo(q.dtype).eps*max(1., x_rms)
    gamma = dict(PROX_Q=1., DIRECT_Q=1., ALIGN_Q=1.)
    result = dict(gamma=gamma, q_RMS=qr, v_RMS=vr, q_floor=floor_q,
                  resolved=False, reason="NUMERICALLY_UNRESOLVED_Q")
    if qr <= floor_q:
        return result
    gamma["PROX_Q"] = max(0., coefficient*qv/q2)
    result.update(resolved=True, reason="PROX_ONLY", cosine_q_v=qv/math.sqrt(q2)/math.sqrt(v2) if v2 else None)
    if u is not None:
        u2, qu, vu = numbers[3:]
        ur = math.sqrt(u2/n)
        floor_u = NUMERICAL_ULPS*torch.finfo(q.dtype).eps*max(1., vr)
        result.update(u_RMS=ur, u_floor=floor_u, negative_alignment=vu < 0.)
        if ur <= floor_u:
            result.update(resolved=False, reason="NUMERICALLY_UNRESOLVED_U")
        else:
            alignment = vu/u2
            gamma.update(DIRECT_Q=max(0., coefficient*qu/q2),
                         ALIGN_Q=max(0., coefficient*alignment*qu/q2))
            result.update(alignment_scale=alignment,
                          cosine_v_u=vu/math.sqrt(v2)/math.sqrt(u2) if v2 else None,
                          cosine_q_u=qu/math.sqrt(q2)/math.sqrt(u2),
                          resolved=True, reason="CURRENT_SCORE_GEOMETRY")
    if not all(math.isfinite(g) for g in gamma.values()):
        raise ValueError("Nonfinite multiplier; no hidden cap or GT rescue")
    return result


def sensed_field(z, y, x, t, next_t, positions, sigma, mode, mask=None, before_backward=None):
    """One graph-enabled forward for ALL IEEE arms; u only for DIRECT/ALIGN."""
    import torch
    if mode not in MODES+( "C_IEEE",):
        raise ValueError("Unknown current-score mode")
    inp, raw = raw_mean(z.dm, x, t, graph=True)
    field = field_from_mean(z, y, raw, t, positions)
    with torch.no_grad():
        residual = y-z.A.H(raw.detach())
        if mask is not None:
            residual = residual*mask.to(dtype=residual.dtype)
        row = dict(R_raw=informative_rms(y-z.A.H(raw.detach()), mask)/sigma,
                   R_clean=informative_rms(y-z.A.H(field["clean"]), mask)/sigma,
                   clip_fraction=float((raw.detach().abs()>1).float().mean()), backward_calls=0)
    if mode == "C_IEEE":
        return field, 1., row
    v, gate = algebra.proximal_vector(lambda v:.5*z.A.base.H(v), z.A.transpose,
                                     residual, float(z.dm.alphas_cumprod[t]), sigma,
                                     rtol=CG_RTOL, maxiter=CG_MAXITER)
    row["cg"] = gate
    if v is None:
        row.update(resolved=False, reason="CG_UNRESOLVED_FALLBACK")
        return field, 1., row
    u = None
    if mode in ("DIRECT_Q", "ALIGN_Q"):
        if before_backward is not None:
            before_backward()
        row["backward_calls"] = 1
        u = algebra.input_vjp(raw, inp, v)
    alpha_s = float(z.dm.alphas_cumprod[next_t]) if next_t >= 0 else 1.
    coefficient = algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]), alpha_s)
    reading = gram_readout(field["q"], v, u, coefficient,
                           float(x.detach().double().square().mean().sqrt()))
    row.update(reading, coefficient=coefficient)
    return field, row["gamma"][mode], row


def vjp_metrology(dm, x, t, vector, before_backward=None):
    """Six UNet forwards, one input backward. No action or final-quality label."""
    import torch
    inp, raw = raw_mean(dm, x, t, graph=True)
    if before_backward is not None:
        before_backward()
    u = algebra.input_vjp(raw, inp, vector)
    _, repeated = raw_mean(dm, x, t, graph=False)
    scale = max(float(raw.detach().double().norm()), float(repeated.double().norm()), 1e-30)
    repeat = float((raw.detach().double()-repeated.double()).norm())/scale
    rms = float(u.double().square().mean().sqrt())
    if rms == 0.:
        raise RuntimeError("Unresolved input VJP")
    direction = u/rms
    reference = float((u.double()*direction.double()).sum())
    base = max(1., float(x.detach().double().square().mean().sqrt()))
    rows = []
    for relative in VJP_FD_SCALES:
        h = relative*base
        _, plus = raw_mean(dm, x+h*direction, t, graph=False)
        _, minus = raw_mean(dm, x-h*direction, t, graph=False)
        difference = float(((plus.double()-minus.double())*vector.detach().double()).sum())/(2*h)
        error = abs(difference-reference)/max(abs(reference), 1e-30)
        rows.append(dict(relative_scale=relative, effective_RMS=h,
                         derivative=difference, reference=reference, relative_error=error))
    passed = repeat <= 1e-5 and all(r["relative_error"] <= .05 for r in rows)
    return dict(passed=bool(passed), grad_mode_repeat_relative_error=repeat, scales=rows,
                forward_calls=6, backward_calls=1,
                scope="Local input derivative only; neither posterior accuracy nor final reconstruction gain")
