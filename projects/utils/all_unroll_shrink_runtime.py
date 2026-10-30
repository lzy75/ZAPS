"""A73: unchanged A71 scalar decision in every original ZAPS unroll."""
from utils import shrink_multitask_runtime as parent
from utils import shrink_multitask as config
from utils import noise_signal_shrink as policy
from utils import paper_clock_runtime as clock
from utils.all_unroll_shrink import ARMS


def controller_class():
    import torch
    from modules.zaps_algorithm import ZAPS, ddpm_posterior_step
    class Controlled(ZAPS):
        def _reverse_diffusion(self, y, nfe_counter=None, eta_override=None,
                               init_noise=None, record_indicators=False):
            self.unrolls = getattr(self, 'unrolls', 0) + 1
            if self.unrolls == 1:
                self.first_entry = clock.snapshot(self, init_noise)
                self.trace = []
            if self.arm == 'BASE':
                return super()._reverse_diffusion(y, nfe_counter, eta_override, init_noise, record_indicators)
            x = init_noise.to(self.device); ab = self.dm.alphas_cumprod
            for k, i in enumerate(range(len(self.tau)-1, -1, -1)):
                t = int(self.tau[i]); s = int(self.tau[i-1]) if i else -1
                with torch.no_grad():
                    eps = self.dm._predict_eps(x, torch.full((y.shape[0],), t, device=self.device, dtype=torch.long))
                if nfe_counter is not None:
                    nfe_counter[0] += 1
                clean = self._tweedie_estimate(x, eps, ab[t], i)
                uncond = ddpm_posterior_step(x, clean, t, s, ab,
                    eta=self.eta if eta_override is None else eta_override,
                    learned_log_var=None, mode='ddpm')
                residual = y-self.A.H(clean); v = self.A.transpose(residual)
                hv = self.dwt.synthesis(self.D[i]*self.dwt.analysis(v))
                q = self.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
                with torch.no_grad():
                    if self.arm == 'NULL_ALL':
                        row = dict(gamma=1., reason='NULL_ALL')
                    else:
                        r = parent.informative(residual.detach().double(), self.A)
                        d = parent.informative((self.A.H(clean.detach()+q.detach()/ab[t].sqrt().clamp(min=1e-8))
                                               - self.A.H(clean.detach())).double(), self.A)
                        row = policy.select(float((r*d).sum()), float(r.square().sum()),
                                            float(d.square().sum()), r.numel(), self.sigma)
                        row['active_measurements'] = r.numel()
                gamma = row['gamma']
                # Same stop-gradient scalar convention as A71; q retains the ζ/D graph.
                x = uncond+q if gamma == 1. else uncond+gamma*q
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite A73 trajectory; no fallback or failed-case retry')
                self.trace.append(dict(unroll=self.unrolls, step=k, t=t, next_t=s, **row))
            return x
    return Controlled


def execute(model, op, y, case, arm, size=256, level=3):
    from utils.diag_learning_rate_ablation import set_seed
    if arm not in (*ARMS, 'NULL_ALL'):
        raise ValueError('Unknown A73 arm')
    set_seed(case['seed'])
    z = controller_class()(model, op, **config.configuration(case['task'], size, level))
    z.arm, z.sigma = arm, case['sigma']
    losses = z.optimize(y, verbose=False, x0_gt=None)
    import math
    import torch
    if any(not math.isfinite(float(v)) for v in losses) or any(
        not torch.isfinite(p).all() or p.grad is not None and not torch.isfinite(p.grad).all()
        for p in (z.zeta, z.D)):
        raise RuntimeError('Nonfinite A73 optimization loss/parameter/gradient')
    if z.unrolls != 10 or (arm != 'BASE' and len(z.trace) != 300):
        raise ValueError('All10 unrolls must finish; output is tenth-unroll last_opt')
    return z._last_opt_x0.detach().cpu(), dict(first_entry=z.first_entry, trace=z.trace, losses=losses,
        zeta=z.zeta.detach().cpu(), D=z.D.detach().cpu())
