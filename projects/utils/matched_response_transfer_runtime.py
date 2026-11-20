"""A122: identical next-denoiser target, paired ray/tangent Gauss--Newton trials.

One joint update in a two-dimensional subspace, not a full matrix optimum.
Both arms evaluate the same continuous candidates to match the AD budget.
"""
import ast
import inspect
import math
from utils import paper_clock_alignment as files
from utils.a95_checkpoint_scope_r1 import forward_ad_checkpoint_scope
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.shrink_multitask_runtime import informative


def bump(count, key):
    count[key] = count.get(key, 0) + 1


def dot(a, b):
    return (a.double()*b.double()).sum()


class Provider:
    def __init__(self, network, predict, operator, origin, y, sigma, count):
        import torch
        self.network, self.predict, self.op = network, predict, operator
        self.origin, self.count = origin.detach(), count
        self.base = self.value(torch.zeros_like(origin))
        residual = informative(y, operator).detach() - self.base
        energy = float(dot(residual, residual))
        self.omega = max(0., 1.-residual.numel()*sigma*sigma/max(energy, 1e-30))
        self.target = (self.base+self.omega*residual).detach()

    def observe(self, x):
        bump(self.count, 'response_H')
        return informative(self.op.H(self.predict(x)), self.op)

    def value(self, delta):
        import torch
        bump(self.count, 'response_value_forwards')
        with torch.no_grad():
            value = self.observe(self.origin+delta).detach()
        files.require(bool(torch.isfinite(value).all()), 'Nonfinite actual response')
        return value

    def loss(self, value):
        return float(.5*dot(value-self.target, value-self.target))

    def gradient(self, delta):
        import torch
        bump(self.count, 'response_vjp_forwards')
        with torch.enable_grad(), input_vjp_parameters(self.network), forward_ad_checkpoint_scope(self.network):
            x = (self.origin+delta).detach().requires_grad_(True)
            value = self.observe(x)
            bump(self.count, 'input_vjp_attempts')
            bump(self.count, 'response_vjp_attempts')
            gradient = torch.autograd.grad(value, x, value.detach()-self.target)[0]
        files.require(bool(torch.isfinite(gradient).all()), 'Nonfinite actual response VJP')
        return value.detach(), gradient.detach()

    def jvp(self, delta, direction):
        import torch
        bump(self.count, 'input_jvp_attempts')
        bump(self.count, 'response_jvp_forwards')
        with torch.no_grad(), forward_ad_checkpoint_scope(self.network):
            primal, tangent = torch.func.jvp(self.predict, (self.origin+delta,), (direction,))
            self.count['response_H'] = self.count.get('response_H', 0)+2
            value = informative(self.op.H(primal), self.op).detach()
            response = informative(self.op.H(tangent), self.op).detach()
        files.require(bool(torch.isfinite(response).all()), 'Nonfinite actual response JVP')
        return value, response


def tangent_direction(wavelet, q, gradient, count):
    """Gain Jacobian M=(I-uu^T)W^T diag(Wq), v=-MM^T grad.

    c appears twice: these are multiplicative gains, not free coefficients.
    Normalization fixes the search basis, not the eventual correction length.
    """
    import torch
    norm = float(q.double().norm())
    u = q/max(norm, 1e-30)
    perpendicular = gradient-u*dot(u, gradient).to(q.dtype)
    bump(count, 'module_dwt_analysis'); c = wavelet.analysis(q)
    bump(count, 'module_dwt_analysis'); g = c*wavelet.analysis(perpendicular)
    bump(count, 'module_dwt_synthesis'); raw = -wavelet.synthesis(c*g)
    raw = raw-u*dot(u, raw).to(q.dtype)
    length = float(raw.double().norm())
    v = raw/max(length, 1e-30)
    files.require(bool(torch.isfinite(v).all()), 'Nonfinite gain-space direction')
    return u, v, length


def solve(provider, wavelet, q, legacy_gamma, arm, count):
    """Exactly 8 predictor calls, 1 VJP, 2 JVPs. No finite differences/search."""
    import torch
    files.require(arm in ('RESPONSE_RAY', 'RESPONSE_DIAG'), 'Unknown response arm')
    native = q.detach()
    legacy = native*legacy_gamma
    native_value, legacy_value = provider.value(native), provider.value(legacy)
    native_loss, legacy_loss = provider.loss(native_value), provider.loss(legacy_value)
    gamma = legacy_gamma if legacy_loss < native_loss else 1.
    initial = native*gamma
    value, gradient = provider.gradient(initial)
    u, v, tangent_norm = tangent_direction(wavelet, native, gradient, count)
    primal_u, ju = provider.jvp(initial, u)
    primal_v, jv = provider.jvp(initial, v)
    # Bypassing legacy checkpoint must preserve the clean-predictor values.
    for primal in (primal_u, primal_v):
        error = float((primal.double()-value.double()).norm()/value.double().norm().clamp_min(1e-30))
        files.require(error <= 1e-4, 'Forward AD changed native clean prediction')
    initial_loss = provider.loss(value)
    files.require(math.isclose(initial_loss, min(native_loss, legacy_loss), rel_tol=2e-4, abs_tol=1e-6),
                  'Checkpoint scope changed initial objective')
    residual = (value-provider.target).double().reshape(-1)
    a, b = ju.double().reshape(-1), jv.double().reshape(-1)
    rho = gamma*float(native.double().norm())
    aa = float(dot(a, a))
    ray_alpha = max(-rho, -float(dot(a, residual))/aa) if aa > 1e-30 else 0.
    ray = initial+ray_alpha*u
    # CPU float64 SVD of two columns; no ridge or hand-tuned damping parameter.
    # Only the 2x2 eigensystem is transferred to CPU, not image-sized matrices.
    gram = torch.stack((dot(a,a),dot(a,b),dot(a,b),dot(b,b))).reshape(2,2).cpu()
    rhs = -torch.stack((dot(a,residual),dot(b,residual))).cpu()
    eigen, vectors = torch.linalg.eigh(gram)
    cutoff = 64*torch.finfo(torch.float64).eps*max(float(eigen.max()), 1e-30)
    inverse = torch.where(eigen > cutoff, eigen.clamp_min(cutoff).reciprocal(), torch.zeros_like(eigen))
    coefficients = vectors @ (inverse*(vectors.T @ rhs))
    alpha, beta = coefficients.tolist()
    diagonal = initial+alpha*u+beta*v
    ray_loss = provider.loss(provider.value(ray))
    diag_loss = provider.loss(provider.value(diagonal))
    candidates = [(initial_loss, initial, 'INITIAL'), (ray_loss, ray, 'CONTINUOUS_RAY')]
    ray_best = min(candidates, key=lambda item:item[0])
    diagonal_best = min(candidates+[(diag_loss, diagonal, 'CONTINUOUS_JOINT')], key=lambda item:item[0])
    chosen = ray_best if arm == 'RESPONSE_RAY' else diagonal_best
    files.require(diagonal_best[0] <= ray_best[0]+1e-8, 'Same-state nested response objective increased')
    detail = dict(omega=provider.omega, legacy_gamma=legacy_gamma, initial_gamma=gamma,
                  native_loss=native_loss, legacy_loss=legacy_loss, initial_loss=initial_loss,
                  ray_trial_loss=ray_loss, diagonal_trial_loss=diag_loss,
                  ray_accepted_loss=ray_best[0], diagonal_accepted_loss=diagonal_best[0],
                  accepted=chosen[2], ray_alpha=ray_alpha, diagonal_alpha=alpha, diagonal_beta=beta,
                  retained_rank=int((eigen > cutoff).sum()), tangent_norm=tangent_norm,
                  q_norm=float(native.double().norm()), output_norm=float(chosen[1].double().norm()),
                  guarantee='same-state actual-response objective only; not final reconstruction quality')
    return chosen[1].detach(), detail


def dps_predict(sampler, model, x, index):
    import torch
    if index < 0:
        return x
    time = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
    return sampler.p_mean_variance(model.model, x, time)['pred_xstart']


def pigdm_predict(model, diffusion, x, s):
    import torch
    if s < 0:
        return x
    time = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
    # Match the original loop's scalar-double coefficients and raw prediction.
    a = float(diffusion.alpha(time)[0])
    return (x-math.sqrt(1-a)*model.model(x,time)[:,:x.shape[1]])/math.sqrt(a)


def controller(model, wavelet, arm, count, positions, trace):
    def apply(op, origin, q, y, sigma, old, step, s, predict):
        if step+1 not in positions:
            return None
        import torch
        before = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if q.is_cuda else []
        provider = Provider(model.model, predict, op, origin, y, sigma, count)
        delta, detail = solve(provider, wavelet, q, float(old['gamma']), arm, count)
        files.require(torch.equal(before[0], torch.get_rng_state()) and
                      all(torch.equal(a,b) for a,b in zip(before[1], torch.cuda.get_rng_state_all() if q.is_cuda else [])),
                      'Response module consumed sampling RNG')
        trace.append(dict(step=step+1, next_time=s, **detail))
        return delta
    return apply


def instrument(runtime, apply, host):
    source = inspect.getsource(runtime.run)
    if host == 'pigdm':
        before = "x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
        after = ("origin=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()\n"
                 "   delta=apply(op,origin,q,y,case['sigma'],row,step,s,"
                 "lambda xx:pigdm_predict(model,diffusion,xx,s))\n"
                 "   x=origin+row['gamma']*q if delta is None else origin+delta")
    else:
        amplitude = 'rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
        before = f'result=x_t-grad*{amplitude} if gamma==1. else x_t-gamma*{amplitude}*grad'
        after = ("delta=apply(op,x_t.detach(),q,measurement,case['sigma'],row,step,"
                 "sampler.timestep_map[index-1] if index else -1,"
                 "lambda xx:dps_predict(sampler,model,xx,index-1))\n"
                 f"            result=({before.split('=',1)[1]}) if delta is None else x_t.detach()+delta")
    files.require(source.count(before) == 1, 'Native correction expression changed')
    namespace = {**vars(runtime), 'apply': apply, 'dps_predict': dps_predict, 'pigdm_predict': pigdm_predict}
    exec(compile(ast.parse(source.replace(before, after)), runtime.__file__, 'exec'), namespace)
    return namespace['run']
