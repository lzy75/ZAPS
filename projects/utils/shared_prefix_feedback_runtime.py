"""A102: one learned ZAPS prefix and three exactly paired final suffixes."""
import copy
from utils import shared_prefix_feedback as rules
from utils import paper_clock_runtime as clock
from utils import noise_signal_shrink as scalar_rule
from utils import equal_nfe_feedback_runtime as closed
from utils import equal_nfe_feedback_runtime_r3 as repaired


def make_operator(case, device, count=None, size=256, kernel=None):
    from utils.shrink_multitask_runtime import make_operator as build
    return build(case, device, count=count, size=size, kernel=kernel)


def train_prefix(model, operator, y, case):
    """Run the common nine ZAPS epochs and retain its Adam state."""
    import torch
    import torch.nn.functional as F
    from modules.zaps_algorithm import ZAPS
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(case['seed'])
    zaps = ZAPS(model, operator, **rules.configuration(case['task']))
    optimizer = torch.optim.Adam([zaps.zeta, zaps.D], lr=zaps.lr)
    initial = torch.randn(y.shape[0], zaps.channels, zaps.img_size, zaps.img_size,
                          device=zaps.device)
    losses = []; nfe = [0]
    for _ in range(9):
        optimizer.zero_grad()
        output = zaps._reverse_diffusion(y, nfe_counter=nfe, eta_override=zaps.eta,
                                         init_noise=initial)
        loss = F.mse_loss(zaps.A.H(output), y); loss.backward()
        torch.nn.utils.clip_grad_norm_([zaps.zeta, zaps.D], max_norm=1.)
        optimizer.step()
        with torch.no_grad(): zaps.zeta.data.clamp_(min=1e-6)
        losses.append(float(loss.detach()))
    return dict(entry=clock.snapshot(zaps, initial), optimizer=copy.deepcopy(optimizer.state_dict()),
                losses=losses, nfe_counter=nfe[0])


def _custom_suffix(zaps, y, initial, arm):
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    x = initial.to(zaps.device); ab = zaps.dm.alphas_cumprod; trace = []
    module = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
    for k, i in enumerate(range(len(zaps.tau)-1, -1, -1)):
        t = int(zaps.tau[i]); s = int(zaps.tau[i-1]) if i else -1
        with torch.no_grad():
            eps = zaps.dm._predict_eps(x, torch.full((y.shape[0],), t, device=zaps.device, dtype=torch.long))
        clean = zaps._tweedie_estimate(x, eps, ab[t], i)
        uncond = ddpm_posterior_step(x, clean, t, s, ab, eta=zaps.eta,
                                     learned_log_var=None, mode='ddpm')
        residual = y-zaps.A.H(clean); v = zaps.A.transpose(residual)
        hv = zaps.dwt.synthesis(zaps.D[i]*zaps.dwt.analysis(v))
        q = zaps.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
        if arm == 'CLOSED_SCALAR_SUFFIX':
            with torch.no_grad():
                base = zaps.A.H(clean); r = closed._informative(residual, zaps.A).double()
                d = closed._informative(zaps.A.H(clean+q/ab[t].sqrt().clamp(min=1e-8))-base,
                                        zaps.A).double()
                row = scalar_rule.select(closed._dot(r,d), closed._dot(r,r), closed._dot(d,d),
                                         r.numel(), rules.SIGMA)
            delta = row['gamma']*q; detail = dict(status='CLOSED_SCALAR', scalar=row)
        elif arm == 'CLOSED_DIAG_SUFFIX':
            delta, detail, module = repaired.exact_diagonal_with_native_straight_through(
                zaps, zaps.A, clean, q, residual, ab[t], rules.SIGMA, module)
        else: raise ValueError('Unknown custom suffix')
        x = uncond+delta
        if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite A102 suffix')
        trace.append(dict(step=k,t=t,next_t=s,**detail))
    return x, trace, module


def run_suffix(model, operator, y, case, arm, prefix):
    """Replay one final epoch from the exact same learned entry and RNG."""
    import torch
    import torch.nn.functional as F
    from modules.zaps_algorithm import ZAPS
    zaps = ZAPS(model, operator, **rules.configuration(case['task']))
    with torch.no_grad():
        zaps.zeta.copy_(prefix['entry']['zeta'].to(zaps.device))
        zaps.D.copy_(prefix['entry']['D'].to(zaps.device))
    optimizer = torch.optim.Adam([zaps.zeta, zaps.D], lr=zaps.lr)
    optimizer.load_state_dict(copy.deepcopy(prefix['optimizer']))
    torch.set_rng_state(prefix['entry']['cpu_rng'])
    if prefix['entry']['cuda_rng']: torch.cuda.set_rng_state_all(prefix['entry']['cuda_rng'])
    optimizer.zero_grad(); initial = prefix['entry']['initial'].to(zaps.device)
    if arm == 'ZAPS_SUFFIX':
        output = zaps._reverse_diffusion(y, nfe_counter=[0], eta_override=zaps.eta,
                                         init_noise=initial)
        trace, module = [], {}
    else: output, trace, module = _custom_suffix(zaps, y, initial, arm)
    loss = F.mse_loss(zaps.A.H(output), y); loss.backward()
    torch.nn.utils.clip_grad_norm_([zaps.zeta,zaps.D], max_norm=1.); optimizer.step()
    with torch.no_grad(): zaps.zeta.data.clamp_(min=1e-6)
    final_rng = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state_all())
    return output.detach().cpu(), dict(loss=float(loss.detach()), trace=trace,
        module_counts=module, final_rng=final_rng)
