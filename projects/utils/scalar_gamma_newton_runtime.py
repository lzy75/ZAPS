"""A69 same A68 Phi/U; scalar AD or model gradient, original wavelet curvature."""
import time
from utils import next_response_bayes_runtime as previous
from utils import scalar_gamma_newton as rule
from utils import historical_ffhq_gain as strong
from utils import paper_clock_runtime as clock
from utils.current_score_runtime_r2 import input_vjp_parameters


def decide(z,y,uncond,q,t,s,index,arm):
    import torch
    queried=0;backs=0;cache={};m0=None;terminal=s<0
    alpha=z.dm.alphas_cumprod[t if terminal else s]
    U=float((1-alpha)/alpha.sqrt())
    with torch.no_grad():
        if terminal:w=q.double()
        else:
            # Original ZAPS surrogate Jacobian; ONLY preconditions scalar search.
            w=((q+(1-alpha)*z.dwt.synthesis(z.D[index-1].detach()*z.dwt.analysis(q)))/
               alpha.sqrt().clamp(min=1e-8)).double()
        hw=z.A.H(w.to(q.dtype)).double()
        h=float(hw.square().sum()/(z.sigma*z.sigma)+w.square().sum()/U)
    def evaluate(gamma,want_gradient,accept_gradient_if):
        nonlocal queried,backs,m0
        use_ad=arm=='AUTO_NEWTON' and not terminal and want_gradient
        # Backward compatibility with the original FFHQ attention checkpoint:
        # parameter flags True temporarily; autograd.grad only requests gamma.
        from contextlib import nullcontext
        context=input_vjp_parameters(z.dm.model) if use_ad else nullcontext()
        with context,torch.set_grad_enabled(use_ad):
            g=torch.tensor(float(gamma),device=q.device,dtype=q.dtype,requires_grad=use_ad)
            latent=uncond+g*q
            if terminal:raw=latent;eps=None
            else:
                queried+=1
                eps=z.dm._predict_eps(latent,torch.full((latent.shape[0],),s,device=latent.device,dtype=torch.long))
                raw=(latent-(1-alpha).sqrt()*eps)/alpha.sqrt().clamp(min=1e-8)
            if m0 is None:m0=raw.detach()
            selected_finite=bool(torch.isfinite(raw).all())
            loss=None;gradient=None
            if selected_finite:
                r=(y-z.A.H(raw)).double();delta=raw.double()-m0.double()
                phi=r.square().sum()/(2*z.sigma*z.sigma)+delta.square().sum()/(2*U)
                loss=float(phi.detach())
                if want_gradient and (accept_gradient_if is None or accept_gradient_if(loss)):
                    if use_ad:
                        backs+=1;z.cost['input_vjp_attempts']+=1
                        gradient=float(torch.autograd.grad(phi,g,create_graph=False)[0].detach())
                    else:
                        gradient=float(-(hw*r).sum()/(z.sigma*z.sigma)+(w*delta).sum()/U)
            cache[float(gamma)]=(raw.detach(),None if eps is None else eps.detach(),latent.detach())
        return dict(value=loss,gradient=gradient)
    if h<=1e-24:
        evaluate(0.,False,None)
        result=dict(gamma=0.,stationarity_met=False,final_gradient_measured=False,
                    projected_step_residual=None,stop_reason='UNRESOLVED_SURROGATE_CURVATURE',
                    value_points=1,gradient_points=0,updates=0,trials=[],etas=[])
    else:result=rule.solve(evaluate,h)
    gamma=result['gamma'];raw,eps,latent=cache[gamma]
    if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite selected scalar-search output')
    result.update(probe_forwards=queried,input_backwards=backs,curvature=h,
                  effective_variance=U,terminal=terminal,
                  gradient_kind='ACTUAL_SCALAR_AD' if arm=='AUTO_NEWTON' else 'WAVELET_MODEL_PROXY',
                  stationarity_scope='actual A68 objective' if arm=='AUTO_NEWTON' else 'surrogate-gradient diagnostic only',
                  resolved=h>1e-24)
    return gamma,None if terminal else (s,latent,eps),result


def controller_class():
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    class Controlled(previous.controller_class()):
        def _reverse_diffusion(self,y,nfe_counter=None,eta_override=None,init_noise=None,record_indicators=False):
            if self.arm not in ('MODEL_NEWTON','AUTO_NEWTON') or getattr(self,'unrolls',0)<9:
                return super()._reverse_diffusion(y,nfe_counter,eta_override,init_noise,record_indicators)
            self.unrolls=10;self.entry=clock.snapshot(self,init_noise)
            x=init_noise.to(self.device);ab=self.dm.alphas_cumprod;self.trace=[];cached=None
            self.last_native_calls=0;self.last_probe_calls=0
            for k,i in enumerate(range(len(self.tau)-1,-1,-1)):
                t=int(self.tau[i]);s=int(self.tau[i-1]) if i else -1
                hit=cached is not None and cached[0]==t and torch.equal(cached[1],x.detach())
                with torch.no_grad():
                    if hit:eps=cached[2]
                    else:
                        eps=self.dm._predict_eps(x,torch.full((y.shape[0],),t,device=self.device,dtype=torch.long))
                        self.last_native_calls+=1
                if nfe_counter is not None and not hit:nfe_counter[0]+=1
                clean=self._tweedie_estimate(x,eps,ab[t],i)
                uncond=ddpm_posterior_step(x,clean,t,s,ab,eta=self.eta if eta_override is None else eta_override,
                                          learned_log_var=None,mode='ddpm')
                r=y-self.A.H(clean);v=self.A.transpose(r)
                hv=self.dwt.synthesis(self.D[i]*self.dwt.analysis(v))
                q=self.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
                with torch.no_grad():gamma,cached,row=decide(self,y,uncond.detach(),q.detach(),t,s,i,self.arm)
                self.last_probe_calls+=row['probe_forwards']
                if nfe_counter is not None:nfe_counter[0]+=row['probe_forwards']
                x=uncond+q if gamma==1. else uncond+gamma*q
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite scalar-search trajectory')
                self.trace.append(dict(step=k,t=t,next_t=s,cached_prediction_reused=bool(hit),**row))
            return x
    return Controlled


def execute(model,operator,y,seed,arm,sigma,count,size=256,level=3):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(seed);cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    z.arm,z.sigma,z.profile,z.pipeline_start,z.cost=arm,sigma,None,time.perf_counter(),count
    if z.tau.cpu().tolist()!=strong.grid():raise ValueError('Strong grid differs')
    losses=z.optimize(y,verbose=False,x0_gt=None)
    return z._last_opt_x0.cpu(),dict(entry=z.entry,trace=z.trace,losses=losses,
        native_last_calls=getattr(z,'last_native_calls',30),probe_calls=getattr(z,'last_probe_calls',0))


def replay(model,operator,y,entry,arm,sigma,count,size=256,level=3):
    import torch
    cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Replay grid differs')
    torch.set_rng_state(entry['cpu_rng'])
    if entry['cuda_rng']:torch.cuda.set_rng_state_all(entry['cuda_rng'])
    z.arm,z.sigma,z.profile,z.unrolls,z.pipeline_start,z.cost=arm,sigma,None,9,time.perf_counter(),count
    out=z._reverse_diffusion(y,init_noise=entry['initial'].to(model.device))
    return out.detach().cpu(),z.trace
