"""A74 original A71 controller, with newly initialized per-grid ZAPS parameters."""
from utils import shrink_multitask_runtime as parent
from utils import step_count_shrink as rules


def execute(model,op,y,case,arm,size=256,level=3):
    from utils.diag_learning_rate_ablation import set_seed
    if arm not in (*rules.ARMS,'NULL'):raise ValueError('Unknown A74 arm')
    set_seed(case['seed'])
    z=parent.controller_class()(model,op,**rules.configuration(case['task'],case['steps'],size,level))
    z.arm,z.sigma=arm,case['sigma']
    loss=z.optimize(y,verbose=False,x0_gt=None)
    if z.unrolls!=10 or z.tau.cpu().tolist()!=rules.grid(case['steps']):raise ValueError('Grid/unroll differs')
    if arm!='BASE' and len(z.trace)!=case['steps']:raise ValueError('Only final unroll must be controlled')
    import math
    import torch
    if any(not math.isfinite(float(v)) for v in loss) or any(not torch.isfinite(p).all() for p in (z.zeta,z.D)):
        raise RuntimeError('Nonfinite A74 optimization; no fallback')
    return z._last_opt_x0.detach().cpu(),dict(entry=z.entry,trace=z.trace,losses=loss,
        zeta=z.zeta.detach().cpu(),D=z.D.detach().cpu())
