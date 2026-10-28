"""Independent 300-forward executions; only tenth unroll uses scalar control."""
import time
from utils import noise_feasible_multiplier as control
from utils import historical_ffhq_gain as strong
from utils import paper_clock_runtime as clock


def controller_class():
    from modules.zaps_algorithm import ZAPS, ddpm_posterior_step
    import torch
    class Controlled(ZAPS):
        def _reverse_diffusion(self, y, nfe_counter=None, eta_override=None, init_noise=None, record_indicators=False):
            self.unrolls = getattr(self, 'unrolls', 0)+1
            if self.unrolls != 10:
                return super()._reverse_diffusion(y, nfe_counter, eta_override, init_noise, record_indicators)
            self.entry = clock.snapshot(self, init_noise)
            self.prefix_seconds = time.perf_counter()-self.pipeline_start
            if self.arm == 'BASE':
                self.trace = []
                return super()._reverse_diffusion(y, nfe_counter, eta_override, init_noise, record_indicators)
            if self.use_learned_var or self.sampler_mode != 'ddpm':
                raise ValueError('Frozen fixed-small DDPM required')
            x = init_noise.to(self.device)
            ab, self.trace = self.dm.alphas_cumprod, []
            for k, i in enumerate(range(len(self.tau)-1, -1, -1)):
                t = int(self.tau[i]); s = int(self.tau[i-1]) if i else -1
                with torch.no_grad():
                    eps = self.dm._predict_eps(x, torch.full((y.shape[0],),t,device=self.device,dtype=torch.long))
                if nfe_counter is not None: nfe_counter[0] += 1
                clean = self._tweedie_estimate(x, eps, ab[t], i)
                uncond = ddpm_posterior_step(x,clean,t,s,ab,eta=self.eta if eta_override is None else eta_override,
                                            learned_log_var=None,mode='ddpm')
                residual = y-self.A.H(clean)
                v = self.A.transpose(residual)
                hv = self.dwt.synthesis(self.D[i]*self.dwt.analysis(v))
                guided = (v+(1.-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8)
                q = self.zeta[i]*guided
                with torch.no_grad():
                    if self.arm in ('NULL', 'TIME_NEAREST'):
                        gamma = 1. if self.arm == 'NULL' else self.profile[k]
                        row = dict(gamma=gamma, reason=self.arm)
                    else:
                        r = residual.detach().double()
                        d = (self.A.H(clean.detach()+q.detach()/ab[t].sqrt().clamp(min=1e-8))
                             -self.A.H(clean.detach())).double()
                        row = control.select(float((r*d).sum()),float(r.square().sum()),float(d.square().sum()),
                                             r.numel(),self.sigma,self.arm)
                        gamma = row['gamma']
                # Exact original addition at gamma1; no extra (gamma-1)*q rounding.
                x = uncond+q if gamma == 1. else uncond+gamma*q
                if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite controlled output')
                self.trace.append(dict(step=k,t=t,next_t=s,**row))
            return x
    return Controlled


def execute(model, operator, y, seed, arm, sigma, profile=None, size=256, level=3):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(seed)
    cfg = strong.config(); cfg.update(img_size=size,level=level)
    z = controller_class()(model,operator,**cfg)
    z.arm, z.sigma, z.profile, z.pipeline_start = arm, sigma, profile, time.perf_counter()
    if z.tau.detach().cpu().tolist() != strong.grid(): raise ValueError('Strong-reference grid differs')
    losses = z.optimize(y,verbose=False,x0_gt=None)
    return z._last_opt_x0.detach().cpu(), dict(entry=z.entry,trace=z.trace,losses=losses,
                                              prefix_seconds=z.prefix_seconds)


def replay(model, operator, y, entry, arm, sigma, profile=None, size=256, level=3):
    """PRIMARY paired quality: same learned prefix and exact noise; paid30 diagnostic."""
    import torch
    cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    with torch.no_grad():
        z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Replay grid differs')
    torch.set_rng_state(entry['cpu_rng']);
    if entry['cuda_rng']:torch.cuda.set_rng_state_all(entry['cuda_rng'])
    z.arm,z.sigma,z.profile,z.unrolls,z.pipeline_start=arm,sigma,profile,9,time.perf_counter()
    # NULL exercises the copied final loop; BASE exercises the original core.
    out=z._reverse_diffusion(y,init_noise=entry['initial'].to(model.device))
    return out.detach().cpu(),z.trace
