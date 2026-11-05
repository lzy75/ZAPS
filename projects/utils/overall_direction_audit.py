"""A92: single-intervention whole-vector orientation/length diagnosis, not a new policy."""
import numpy as np

KINDS = ('NATIVE', 'DIAG_ALIGN', 'PHYS_DIRECT')
SCALES = (.5, 1., 1.5)
ARMS = ('NATIVE_100',)+tuple(f'{kind}_{int(scale*100):03d}' for kind in KINDS for scale in SCALES
                            if (kind, scale) != ('NATIVE', 1.))
INTERVENTION = 15  # zero-based last-unroll location; original grid stays fixed


def cone_coefficients(energies, dots):
    e, d = np.asarray(energies, dtype=np.float64), np.asarray(dots, dtype=np.float64)
    if e.shape != d.shape or (e < 0).any() or not np.isfinite(e).all() or not np.isfinite(d).all():
        raise ValueError('Invalid orthogonal-component moments')
    return np.maximum(0., np.divide(d, e, out=np.zeros_like(d), where=e > 0))


def specification(arm):
    if arm not in ARMS:
        raise ValueError('Unknown pre-registered arm')
    kind, scale = arm.rsplit('_', 1)
    return kind, int(scale)/100


def make_z(model, op, entry, task, size=256, level=3):
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.shrink_multitask import configuration
    z = ZAPS(model, op, **configuration(task, size, level))
    with torch.no_grad():
        z.zeta.copy_(entry['zeta'].to(model.device))
        z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(), entry['tau']):
        raise ValueError('Original grid changed')
    return z


def restore_rng(state, device):
    import torch
    torch.set_rng_state(state['cpu_rng'])
    if str(device).startswith('cuda'):
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def rng_state(device):
    import torch
    return dict(cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [])


def native_parts(model, z, x, y, i):
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    t = int(z.tau[i]); s = int(z.tau[i-1]) if i else -1
    ab = model.alphas_cumprod
    eps = model._predict_eps(x, torch.full((x.shape[0],), t, device=model.device, dtype=torch.long))
    clean = z._tweedie_estimate(x, eps, ab[t], i)
    uncond = ddpm_posterior_step(x, clean, t, s, ab, eta=1., learned_log_var=None, mode='ddpm')
    v = z.A.transpose(y-z.A.H(clean))
    hv = z.dwt.synthesis(z.D[i]*z.dwt.analysis(v))
    q = z.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
    return uncond, q, clean


def prefix(model, op, y, entry, task, size=256, level=3):
    import torch
    z = make_z(model, op, entry, task, size, level)
    restore_rng(entry, model.device)
    x = entry['initial'].to(model.device)
    with torch.no_grad():
        for step, i in enumerate(range(29, -1, -1)):
            uncond, q, _ = native_parts(model, z, x, y, i)
            if step == INTERVENTION:
                return dict(uncond=uncond.cpu(), q=q.cpu(), next_index=i-1,
                            current_t=int(z.tau[i]), next_t=int(z.tau[i-1]),
                            **rng_state(model.device))
            x = uncond+q
    raise RuntimeError('Intervention not reached')


def physical_reference(model, op, y, state, counts):
    """True input VJP through NEXT clamped denoiser, not legacy detached eps."""
    import torch
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.current_score_runtime import raw_mean
    from utils.shrink_multitask_runtime import informative
    before = rng_state(model.device)
    with torch.enable_grad(), input_vjp_parameters(model.model):
        inp, raw = raw_mean(model, state['uncond'].to(model.device), state['next_t'], graph=True)
        clean = raw.clamp(-1, 1)
        residual = informative(y-op.H(clean), op)
        loss = .5*residual.square().sum()
        counts['input_vjp_attempts'] += 1
        descent = -torch.autograd.grad(loss, inp, create_graph=False)[0]
        counts['input_vjp_completed'] = counts.get('input_vjp_completed', 0)+1
    after = rng_state(model.device)
    if not torch.equal(before['cpu_rng'], after['cpu_rng']) or any(not torch.equal(a,b) for a,b in zip(before['cuda_rng'],after['cuda_rng'])):
        raise RuntimeError('Reference probe consumed sampling RNG')
    if not bool(torch.isfinite(descent).all()):
        raise RuntimeError('Nonfinite physical descent')
    return descent.detach(), clean.detach()


def directions(q, reference, dwt):
    import torch
    from utils.directional_reliability import subbands
    cq, cr = dwt.analysis(q), dwt.analysis(reference)
    bands = subbands(q.shape[-2], q.shape[-1], dwt.level)
    energies, dots = [], []
    for _, (ys, xs) in bands:
        a, b = cq[...,ys,xs].double(), cr[...,ys,xs].double()
        energies.append(float(a.square().sum())); dots.append(float((a*b).sum()))
    coefficients = cone_coefficients(energies, dots)
    cc = cq.clone()
    for coefficient, (_, (ys, xs)) in zip(coefficients, bands):
        cc[...,ys,xs] *= float(coefficient)
    projected = dwt.synthesis(cc)
    magnitude = float(q.double().norm())
    if not np.isfinite(magnitude) or magnitude == 0:
        raise RuntimeError('Zero/nonfinite native correction; cannot match length')
    units, details = {}, {}
    reference_norm = float(reference.double().norm())
    for name, v in (('NATIVE', q), ('DIAG_ALIGN', projected), ('PHYS_DIRECT', reference)):
        norm = float(v.double().norm())
        units[name] = None if norm == 0 else v/norm
        details[name] = dict(resolved=norm > 0, original_norm=norm,
            cosine_to_physical=None if norm == 0 or reference_norm == 0 else float((v.double()*reference.double()).sum())/(norm*reference_norm))
    return units, dict(native_norm=magnitude, directions=details,
                       band_projection_coefficients=coefficients.tolist(), band_labels=[b[0] for b in bands])


def continue_run(model, z, y, state, correction):
    import torch
    restore_rng(state, model.device)
    x = state['uncond'].to(model.device)+correction
    first = None
    with torch.no_grad():
        for i in range(state['next_index'], -1, -1):
            uncond, q, clean = native_parts(model, z, x, y, i)
            if first is None: first = clean.detach().cpu()
            x = uncond+q
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError('Nonfinite continuation; no fallback')
    return x.detach().cpu(), first, rng_state(model.device)


def offline_readout(op, y, gt, reference_clean, candidate_clean, correction, physical):
    """GT only AFTER fixed-arm reconstruction; never used for action selection."""
    import torch
    from utils.shrink_multitask_runtime import informative
    with torch.no_grad():
        base = op.H(reference_clean); predicted = op.H(candidate_clean); truth = op.H(gt)
        r = informative(y-base, op).double(); response = informative(predicted-base, op).double()
        signal = informative(truth-base, op).double(); noise = informative(y-truth, op).double()
        physical_gain = float((r*response).sum()-.5*response.square().sum())
        clean_gain = float((signal*response).sum()-.5*response.square().sum())
        noise_gain = float((noise*response).sum())
        error = abs(physical_gain-clean_gain-noise_gain)
        tolerance = 1e-5*max(1.,abs(physical_gain),abs(clean_gain),abs(noise_gain))
        if error > tolerance: raise RuntimeError('Offline noise/clean gain identity failed')
        return dict(physical_gain=physical_gain, clean_measurement_gain=clean_gain,
            fitted_noise_gain=noise_gain, decomposition_error=error,
            offline_H_actions=3,
            physical_first_order_gain=float((physical.double()*correction.double()).sum()),
            residual_energy_before=float(r.square().sum()),
            residual_energy_after=float(informative(y-predicted,op).double().square().sum()),
            active_measurements=r.numel(),
            next_clean_full_mse=float((candidate_clean.double()-gt.double()).square().mean()),
            reference_next_clean_full_mse=float((reference_clean.double()-gt.double()).square().mean()))
