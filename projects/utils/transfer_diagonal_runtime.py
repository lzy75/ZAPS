"""Same frozen-observation diagonal step on existing DPS and PiGDM loops."""
import ast
import inspect
import math
from utils import paper_clock_alignment as files
from utils.equal_nfe_feedback_runtime import diagonal_update


class LinearView:
    def __init__(self, operator, count):
        self.operator, self.count = operator, count
        self._mask_cache = getattr(operator, '_mask_cache', None)

    def H(self, value):
        self.count['refinement_H'] += 1
        return self.operator.H(value)

    def transpose(self, value):
        self.count['refinement_Ht'] += 1
        if hasattr(self.operator, 'Ht'):
            return self.operator.Ht(value)
        # The historical SR wrapper exposes an empirical bicubic transpose.
        # The new quadratic uses the raw physical operator's actual adjoint.
        base = getattr(self.operator, 'base', self.operator)
        return base.transpose(value)


def feedback(wavelet, count, trace):
    def apply(operator, clean, q, residual, alpha, sigma, old=None):
        delta, detail, _ = diagonal_update(wavelet, LinearView(operator, count), clean, q,
                                           residual, q.new_tensor(alpha), sigma, count)
        if old is not None:
            a, b = old['gamma'], detail['scalar']['gamma']
            files.require(math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-6),
                          'Original scalar and diagonal initializer differ')
        row = dict(gamma=detail['scalar']['gamma'], **detail)
        trace.append(row)
        return delta, row
    return apply


def instrument(runtime, apply, host):
    source = inspect.getsource(runtime.run)
    if host == 'pigdm':
        before = "x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q"
        after = ("delta,row=apply(op,m.detach(),q,r,a,case['sigma'])\n"
                 "   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    else:
        amplitude = 'rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
        before = f'result=x_t-grad*{amplitude} if gamma==1. else x_t-gamma*{amplitude}*grad'
        after = ("delta,refinement=apply(op,clean,q,measurement-op.H(clean),a*a,case['sigma'],row)\n"
                 "            result=x_t+delta\n"
                 "            row=dict(row,refinement=refinement)")
    files.require(source.count(before) == 1, 'Original correction interface changed')
    namespace = {**vars(runtime), 'apply': apply}
    exec(compile(ast.parse(source.replace(before, after)), runtime.__file__, 'exec'), namespace)
    return namespace['run']
