"""Apply the A118 closed update inside the unchanged native blur samplers."""
import inspect
import math

from utils import paper_clock_alignment as files
from utils import transfer_diagonal_runtime as transfer


ARMS = ('BASE', 'TAIL_SCALAR', 'A118_DIAGONAL', 'BOUNDED_DIAGONAL')
LAST_ACTIVE_T = 299
RELATIVE_BOUND = .5


def step_fraction(anchor_norm, extra_norm):
    return min(1., RELATIVE_BOUND * anchor_norm / extra_norm) if extra_norm > 0 else 1.


def refinement(wavelet, arm, count, trace):
    """A118 line search on the host scalar anchor and the linear H response."""
    from utils.equal_nfe_feedback_runtime import _dot
    dwt = getattr(wavelet, 'dwt', wavelet)
    files.require(callable(getattr(dwt, 'analysis', None)) and callable(getattr(dwt, 'synthesis', None)),
                  'Expected a wavelet transform with analysis/synthesis')

    def apply(operator, q, residual, alpha, scalar, t):
        view = transfer.LinearView(operator, count)
        gamma = scalar['gamma']
        scale = math.sqrt(alpha)
        files.require(math.isfinite(scale) and scale > 0, 'Invalid diffusion coefficient')
        anchor = gamma * q
        anchor_response = view.H(anchor / scale)
        target = scalar['weight'] * residual.double().reshape_as(anchor_response)
        error = anchor_response.double() - target
        initial = .5 * _dot(error, error)

        support = dwt.analysis(q) != 0
        adjoint = view.transpose(error.to(anchor_response.dtype)) / scale
        gradient = dwt.analysis(adjoint) * support
        count['module_adjoint_actions'] += 1
        count['module_dwt_analysis'] += 2
        energy = _dot(gradient, gradient)
        files.require(math.isfinite(initial) and math.isfinite(energy), 'Nonfinite diagonal objective or gradient')
        extra = anchor.new_zeros(anchor.shape)
        extra_response = error.new_zeros(error.shape)
        if energy > 0:
            direction = dwt.synthesis(gradient)
            count['module_dwt_synthesis'] += 1
            direction_response = view.H(direction / scale).double()
            curvature = _dot(direction_response, direction_response)
            files.require(math.isfinite(curvature) and curvature > 0, 'Unresolved diagonal line-search curvature')
            eta = energy / curvature
            extra = -eta * direction
            extra_response = -eta * direction_response
        anchor_norm = float(anchor.double().norm())
        extra_norm = float(extra.double().norm())
        fraction = step_fraction(anchor_norm, extra_norm) if arm == 'BOUNDED_DIAGONAL' else 1.
        delta = anchor + fraction * extra
        final_error = error + fraction * extra_response
        value = .5 * _dot(final_error, final_error)
        files.require(math.isfinite(value) and value <= initial + 2e-5 * max(1., initial),
                      'Diagonal proxy increased')
        trace.append(dict(t=t, gamma=gamma, fraction=fraction, proxy_initial=initial,
                          proxy_final=value, scalar_norm=anchor_norm, proposed_extra_norm=extra_norm,
                          applied_extra_norm=float((delta-anchor).double().norm())))
        return delta
    return apply


def instrument(runtime, host, arm, cutoff, refine):
    """Gate only the feedback branch. The posterior, RNG and original q stay native."""
    files.require(arm in ARMS, 'Unknown comparison arm')
    if arm == 'BASE':
        return runtime.run
    source = inspect.getsource(runtime.run)
    def replace_once(before, after):
        nonlocal source
        files.require(source.count(before) == 1, 'Original sampling interface changed: ' + before)
        source = source.replace(before, after, 1)

    diagonal = arm != 'TAIL_SCALAR'
    if host == 'dps':
        replace_once("if arm=='DPS':", "if arm=='DPS' or t > cutoff:")
        if diagonal:
            amplitude = 'rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
            before = f'result=x_t-grad*{amplitude} if gamma==1. else x_t-gamma*{amplitude}*grad'
            replace_once(before, "with torch.no_grad():\n"
                         "                delta=refine(op,q,residual,a*a,row,t)\n"
                         "            result=x_t+delta")
    else:
        replace_once("if arm=='SHRINK':", "if arm=='SHRINK' and t <= cutoff:")
        if diagonal:
            before = "x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
            replace_once(before, "delta=refine(op,q,r,a,row,t) if t <= cutoff else q\n"
                         "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    namespace = {**vars(runtime), 'cutoff': cutoff, 'refine': refine}
    exec(compile(source, runtime.__file__, 'exec'), namespace)
    return namespace['run']
