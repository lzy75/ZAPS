"""A126: A118 coefficient support with the actual next clean-predictor response.

The scalar anchor is preserved. The nonlinear objective and target are frozen
within a sampling position. No two-dimensional gain subspace or relative cap.
"""
import math
from utils import paper_clock_alignment as files
from utils import matched_response_transfer_runtime as response

RAY_ITERATIONS = 1
DIAGONAL_ITERATIONS = 2
MAX_TRIALS = 3
ARMIJO = 1e-4


def minimize(provider, wavelet, q, initial, mode, iterations, count):
    import torch
    files.require(mode in ('ray', 'diagonal'), 'Unknown coefficient space')
    delta = initial.detach().clone()
    response.bump(count, 'module_dwt_analysis')
    coefficients = wavelet.analysis(q)
    support = coefficients != 0
    qnorm = float(q.double().norm())
    u = q / max(qnorm, 1e-30)
    history = []
    reason = 'iteration_budget'
    for iteration in range(iterations):
        value, gradient = provider.gradient(delta)
        loss = provider.loss(value)
        if mode == 'ray':
            projected = u * response.dot(u, gradient).to(q.dtype)
        else:
            response.bump(count, 'module_dwt_analysis')
            response.bump(count, 'module_dwt_synthesis')
            projected = wavelet.synthesis(wavelet.analysis(gradient) * support)
        descent = float(response.dot(gradient, projected))
        files.require(math.isfinite(loss) and math.isfinite(descent), 'Nonfinite response gradient')
        if descent <= 1e-24:
            reason = 'projected_gradient_zero'
            history.append(dict(iteration=iteration, before=loss, after=loss, reason=reason))
            break
        direction = -projected
        primal, change = provider.jvp(delta, direction)
        error = float((primal.double()-value.double()).norm()/value.double().norm().clamp_min(1e-30))
        files.require(error <= 1e-4, 'Forward AD changed the next clean prediction')
        curvature = float(response.dot(change, change))
        files.require(math.isfinite(curvature) and curvature > 0, 'Unresolved nonlinear response curvature')
        step = descent / curvature
        if mode == 'ray':
            # A ray has nonnegative length; the full diagonal may have signed gains.
            rho = float(response.dot(u, delta))
            drho = float(response.dot(u, direction))
            if drho < 0:
                step = min(step, max(0., rho) / -drho)
        accepted = False
        trials = []
        for attempt in range(MAX_TRIALS):
            candidate = delta + step * direction
            trial = provider.loss(provider.value(candidate))
            tolerance = 1e-9 * max(1., loss)
            accepted = trial <= loss - ARMIJO * step * descent + tolerance
            trials.append(dict(step=step, loss=trial, accepted=accepted))
            if accepted:
                delta = candidate.detach()
                break
            step *= .5
        history.append(dict(iteration=iteration, before=loss, after=trial if accepted else loss,
                            descent=descent, curvature=curvature, trials=trials,
                            accepted=accepted))
        if not accepted:
            reason = 'line_search_budget'
            break
    return delta, dict(mode=mode, termination=reason, iterations=history)


def solve(provider, wavelet, q, legacy_gamma, arm, count):
    files.require(arm in ('RESPONSE_RAY', 'RESPONSE_DIAG'), 'Unknown response arm')
    anchor = q.detach() * legacy_gamma
    ray, ray_detail = minimize(provider, wavelet, q, anchor, 'ray', RAY_ITERATIONS, count)
    if arm == 'RESPONSE_RAY':
        return ray, dict(ray=ray_detail, q_norm=float(q.double().norm()), gamma=legacy_gamma)
    # Start from exactly the same continuous ray solve at this state.
    diagonal, detail = minimize(provider, wavelet, q, ray, 'diagonal', DIAGONAL_ITERATIONS, count)
    rows = ray_detail['iterations']
    ray_loss = rows[-1]['after'] if rows else None
    diag_rows = detail['iterations']
    if diag_rows and ray_loss is not None:
        files.require(diag_rows[0]['before'] <= ray_loss+2e-5*max(1.,ray_loss), 'Ray anchor was not preserved')
        files.require(diag_rows[-1]['after'] <= ray_loss+2e-5*max(1.,ray_loss), 'Full coefficient objective increased')
    return diagonal, dict(ray=ray_detail, diagonal=detail, gamma=legacy_gamma,
                          q_norm=float(q.double().norm()), output_norm=float(diagonal.double().norm()),
                          coordinates='A118 free coefficients on Wq != 0; signed diagonal gains',
                          guarantee='same-state frozen nonlinear response loss; no final metric ordering')


def controller(model, wavelet, arm, count, times, trace):
    active = set(times)
    def apply(op, origin, q, y, sigma, old, step, s, predict):
        # Native source adapters pass both the actual current t and next index.
        if step not in active:
            return None
        import torch
        before = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if q.is_cuda else []
        forwards = count.get('forwards', 0)
        vjp = count.get('response_vjp_attempts', 0)
        jvp = count.get('response_jvp_forwards', 0)
        provider = response.Provider(model.model, predict, op, origin, y, sigma, count)
        delta, detail = solve(provider, wavelet, q, float(old['gamma']), arm, count)
        count['response_network_forwards'] = count.get('response_network_forwards', 0) + count.get('forwards', 0) - forwards
        if s < 0:
            count['terminal_response_vjp'] = count.get('terminal_response_vjp', 0) + count.get('response_vjp_attempts', 0) - vjp
            count['terminal_response_jvp'] = count.get('terminal_response_jvp', 0) + count.get('response_jvp_forwards', 0) - jvp
        files.require(torch.equal(before[0], torch.get_rng_state()) and
                      all(torch.equal(a,b) for a,b in zip(before[1],torch.cuda.get_rng_state_all() if q.is_cuda else [])),
                      'Response solver consumed sampling RNG')
        trace.append(dict(t=step, next_time=s, **detail))
        return delta
    return apply


def instrument(runtime, host, arm, times, apply=None):
    """Keep native q, sampler, scalar rule and Gaussian draws; intervene at 30 t's."""
    import inspect
    source = inspect.getsource(runtime.run)
    def replace(before, after):
        nonlocal source
        files.require(source.count(before)==1, 'Original correction interface changed: '+before)
        source = source.replace(before, after, 1)
    if host == 'pigdm':
        replace("if arm=='SHRINK':", "if arm=='SHRINK' and t in active_times:")
        if arm != 'SCALAR':
            replace("x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q",
                    "origin=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()\n"
                    "   delta=apply(op,origin,q,y,case['sigma'],row,t,s,lambda xx:pigdm_predict(model,diffusion,xx,s)) if t in active_times else None\n"
                    "   x=origin+row['gamma']*q if delta is None else origin+delta")
    else:
        replace("if arm=='DPS':", "if arm=='DPS' or t not in active_times:")
        if arm != 'SCALAR':
            scale='rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
            before=f'result=x_t-grad*{scale} if gamma==1. else x_t-gamma*{scale}*grad'
            replace(before,
                    "delta=apply(op,x_t.detach(),q,measurement,case['sigma'],row,t,sampler.timestep_map[index-1] if index else -1,lambda xx:dps_predict(sampler,model,xx,index-1))\n"
                    f"            result=({before.split('=',1)[1]}) if delta is None else x_t.detach()+delta")
    namespace={**vars(runtime),'active_times':set(times),'apply':apply,
               'pigdm_predict':response.pigdm_predict,'dps_predict':response.dps_predict}
    exec(compile(source,runtime.__file__,'exec'),namespace)
    return namespace['run']
