"""A70 final-unroll scalar-only intervention; no score probes/input VJP."""
import time
from utils import noise_signal_shrink as rule
from utils import noise_feasible_multiplier as first
from utils import noise_feasible_runtime as previous
from utils import historical_ffhq_gain as strong
from utils import paper_clock_runtime as clock

def controller_class():
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    class Controlled(previous.controller_class()):
        def _reverse_diffusion(self,y,nfe_counter=None,eta_override=None,init_noise=None,record_indicators=False):
            if self.arm in ('BASE','NULL','FIRST_NOISE') or getattr(self,'unrolls',0)<9:
                return super()._reverse_diffusion(y,nfe_counter,eta_override,init_noise,record_indicators)
            self.unrolls=10;self.entry=clock.snapshot(self,init_noise)
            self.prefix_seconds=time.perf_counter()-self.pipeline_start
            x=init_noise.to(self.device);ab=self.dm.alphas_cumprod;self.trace=[]
            for k,i in enumerate(range(len(self.tau)-1,-1,-1)):
                t=int(self.tau[i]);s=int(self.tau[i-1]) if i else -1
                with torch.no_grad():eps=self.dm._predict_eps(x,torch.full((y.shape[0],),t,device=self.device,dtype=torch.long))
                if nfe_counter is not None:nfe_counter[0]+=1
                clean=self._tweedie_estimate(x,eps,ab[t],i)
                uncond=ddpm_posterior_step(x,clean,t,s,ab,eta=self.eta if eta_override is None else eta_override,
                                          learned_log_var=None,mode='ddpm')
                r=y-self.A.H(clean);v=self.A.transpose(r)
                hv=self.dwt.synthesis(self.D[i]*self.dwt.analysis(v))
                q=self.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
                with torch.no_grad():
                    if self.arm=='SHRINK':
                        residual=r.detach().double()
                        d=(self.A.H(clean.detach()+q.detach()/ab[t].sqrt().clamp(min=1e-8))-self.A.H(clean.detach())).double()
                        row=rule.select(float((residual*d).sum()),float(residual.square().sum()),float(d.square().sum()),residual.numel(),self.sigma)
                    else:
                        if self.profile is None or len(self.profile)!=30:raise ValueError('Frozen30 multiplier profile required')
                        row=dict(gamma=float(self.profile[k]),reason=self.arm)
                    gamma=row['gamma']
                x=uncond+q if gamma==1. else uncond+gamma*q
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite A70 trajectory; no outcome-based fallback')
                self.trace.append(dict(step=k,t=t,next_t=s,**row))
            return x
    return Controlled

def execute(model,operator,y,seed,arm,sigma,profile=None,size=256,level=3):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(seed);cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    z.arm,z.sigma,z.profile,z.pipeline_start=arm,sigma,profile,time.perf_counter()
    if z.tau.cpu().tolist()!=strong.grid():raise ValueError('Original grid differs')
    losses=z.optimize(y,verbose=False,x0_gt=None)
    return z._last_opt_x0.detach().cpu(),dict(entry=z.entry,trace=z.trace,losses=losses)

def replay(model,operator,y,entry,arm,sigma,profile=None,size=256,level=3):
    import torch
    cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Replay grid differs')
    torch.set_rng_state(entry['cpu_rng'])
    if entry['cuda_rng']:torch.cuda.set_rng_state_all(entry['cuda_rng'])
    z.arm,z.sigma,z.profile,z.unrolls,z.pipeline_start=arm,sigma,profile,9,time.perf_counter()
    return z._reverse_diffusion(y,init_noise=entry['initial'].to(model.device)).detach().cpu(),z.trace
