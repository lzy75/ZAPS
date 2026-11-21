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
    from utils.equal_nfe_feedback_runtime import diagonal_update

    def apply(operator, clean, q, residual, alpha, sigma, scalar, t):
        view = transfer.LinearView(operator, count)
        delta, detail, _ = diagonal_update(wavelet, view, clean, q, residual,
                                           q.new_tensor(alpha), sigma, count)
        gamma = scalar['gamma']
        files.require(math.isclose(gamma, detail['scalar']['gamma'], rel_tol=2e-5, abs_tol=1e-6),
                      'A118 and native scalar initializers differ')
        anchor = gamma * q
        extra = delta - anchor
        anchor_norm = float(anchor.double().norm())
        extra_norm = float(extra.double().norm())
        fraction = step_fraction(anchor_norm, extra_norm) if arm == 'BOUNDED_DIAGONAL' else 1.
        if fraction < 1.:
            delta = anchor + fraction * extra
            # Evaluate the accepted update using the identical A118 proxy.
            a = q.new_tensor(alpha).sqrt().clamp(min=1e-8)
            response = (view.H(clean + delta / a) - view.H(clean)).double()
            error = response - detail['scalar']['weight'] * residual.double()
            value = float(.5 * error.square().sum())
            files.require(math.isfinite(value) and value <= detail['proxy_initial'] + 2e-5 * max(1., detail['proxy_initial']),
                          'Bounded diagonal proxy increased')
        else:
            value = detail['proxy_final']
        trace.append(dict(t=t, gamma=gamma, fraction=fraction, proxy_initial=detail['proxy_initial'],
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
                         "                delta=refine(op,clean,q,residual,a*a,case['sigma'],row,t)\n"
                         "            result=x_t+delta")
    else:
        replace_once("if arm=='SHRINK':", "if arm=='SHRINK' and t <= cutoff:")
        if diagonal:
            before = "x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
            replace_once(before, "delta=refine(op,m.detach(),q,r,a,case['sigma'],row,t) if t <= cutoff else q\n"
                         "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    namespace = {**vars(runtime), 'cutoff': cutoff, 'refine': refine}
    exec(compile(source, runtime.__file__, 'exec'), namespace)
    return namespace['run']
