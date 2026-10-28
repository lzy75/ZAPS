"""A68 real next-denoiser response, directional variance and paid validation."""
import math
import time
from utils import noise_feasible_runtime as original
from utils import next_response_bayes as rule
from utils import historical_ffhq_gain as strong
from utils import paper_clock_runtime as clock


def decide(z,y,uncond,q,t,s,current_x,arm):
    import torch
    queried=0;cache={};terminal=s<0
    def predict(inp,index):
        nonlocal queried
        queried+=1
        eps=z.dm._predict_eps(inp,torch.full((inp.shape[0],),index,device=inp.device,dtype=torch.long))
        alpha=z.dm.alphas_cumprod[index]
        raw=(inp-(1-alpha).sqrt()*eps)/alpha.sqrt().clamp(min=1e-8)
        return raw,eps
    def evaluate(gamma):
        gamma=float(gamma)
        if gamma not in cache:
            latent=uncond+q if gamma==1. else uncond+gamma*q
            raw,eps=(latent,None) if terminal else predict(latent,s)
            cache[gamma]=(raw,eps,latent)
        return cache[gamma]
    row=dict(terminal=terminal,gamma=1.,resolved=False,uncertainty_source='UNRESOLVED')
    def finish(gamma):
        raw,eps,latent=evaluate(gamma)
        if not torch.isfinite(raw).all():raise RuntimeError('Selected original/controlled prediction nonfinite')
        row.update(gamma=float(gamma),probe_forwards=queried)
        return float(gamma),None if terminal else (s,latent,eps),row
    m0,_,_=evaluate(0.)
    if not torch.isfinite(m0).all():
        row['reason']='NONFINITE_REFERENCE_KEEP_ORIGINAL';return finish(1.)
    n=q.numel();qr=float(q.double().square().mean().sqrt())
    scale=max(1.,float((current_x if terminal else uncond).double().square().mean().sqrt()))
    if qr<=32*torch.finfo(q.dtype).eps*scale:
        row['reason']='SMALL_ORIGINAL_DIRECTION';return finish(1.)
    delta=rule.PROBE_RMS*scale/qr
    if terminal:
        w=q.double();local_error=0.
    else:
        plus,_,_=evaluate(delta);minus,_,_=evaluate(-delta)
        if not (torch.isfinite(plus).all() and torch.isfinite(minus).all()):
            row['reason']='NONFINITE_RESPONSE_PROBE';return finish(1.)
        change=plus.double()-minus.double()
        w=change/(2*delta)
        local_error=float((plus.double()+minus.double()-2*m0.double()).norm()/change.norm().clamp(min=1e-30))
    w2=float(w.square().sum())
    row.update(delta_gamma=delta,local_nonlinearity=local_error,w_RMS=math.sqrt(w2/n))
    if w2<=1e-24 or local_error>rule.MAX_NONLINEARITY:
        row['reason']='UNRESOLVED_LOCAL_RESPONSE';return finish(1.)
    alpha=float(z.dm.alphas_cumprod[t if terminal else s])
    c=rule.diffusion_variance(alpha);kappa=1.
    if arm=='REAL_STATE':
        e=(w/math.sqrt(w2)).to(q.dtype)
        h=rule.PROBE_RMS*scale*math.sqrt(n)
        reference=current_x if terminal else uncond
        mp,_=predict(reference+h*e,t if terminal else s)
        mm,_=predict(reference-h*e,t if terminal else s)
        if torch.isfinite(mp).all() and torch.isfinite(mm).all():
            measured=float(((mp.double()-mm.double())*e.double()).sum()/(2*h))
            row['directional_J']=measured
            if math.isfinite(measured) and measured>0:
                kappa=measured;row['uncertainty_source']='DIRECTIONAL_STATE'
            else:row['uncertainty_source']='ISO_FALLBACK_NONPOSITIVE_J'
        else:row['uncertainty_source']='ISO_FALLBACK_NONFINITE_J'
    else:row['uncertainty_source']='SCHEDULE_ONLY'
    variance=c*kappa
    r=(y-z.A.H(m0)).double()
    d=z.A.H(w.to(q.dtype)).double()  # Frozen linear model-domain H; no affine offset.
    gamma=rule.proposal(float((r*d).sum()),float(d.square().sum()),w2,z.sigma,variance)
    row.update(resolved=True,reason='REAL_RESPONSE_GAUSSIAN',diffusion_variance=c,
               effective_variance=variance,kappa=kappa,proposed_gamma=gamma,
               residual_R=float(r.square().mean().sqrt()/z.sigma))
    def objective(raw):
        if not torch.isfinite(raw).all():return None
        residual=(y-z.A.H(raw)).double()
        value=float(residual.square().sum()/(2*z.sigma*z.sigma)+
                    (raw.double()-m0.double()).square().sum()/(2*variance))
        return value if math.isfinite(value) else None
    loss0=objective(m0);m1,_,_=evaluate(1.);loss1=objective(m1)
    if loss0 is None or loss1 is None:raise RuntimeError('Nonfinite original objective')
    selected=0. if loss0<loss1 else 1.;best=min(loss0,loss1);attempts=[]
    # Continuous unbounded proposal; finite paid backtracking is NOT a global solve.
    for divisor in (1.,2.,4.):
        g=gamma/divisor
        if g in (0.,1.):continue
        candidate,_,_=evaluate(g);value=objective(candidate)
        change=candidate.double()-m0.double()
        expected=g*w
        error=float((change-expected).norm()/max(float(change.norm()),float(expected.norm()),1e-30))
        accepted=value is not None and error<=rule.MAX_NONLINEARITY and value<best
        attempts.append(dict(gamma=g,objective=value,nonlinearity=error,accepted=bool(accepted)))
        if accepted:selected,best=g,value;break
    row.update(objective0=loss0,objective1=loss1,selected_objective=best,attempts=attempts,
               selected_more_than_two=selected>2.)
    return finish(selected)


def controller_class():
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    class Controlled(original.controller_class()):
        def _reverse_diffusion(self,y,nfe_counter=None,eta_override=None,init_noise=None,record_indicators=False):
            if self.arm not in ('REAL_ISO','REAL_STATE') or getattr(self,'unrolls',0)<9:
                return super()._reverse_diffusion(y,nfe_counter,eta_override,init_noise,record_indicators)
            self.unrolls=10;self.entry=clock.snapshot(self,init_noise)
            self.prefix_seconds=time.perf_counter()-self.pipeline_start
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
                residual=y-self.A.H(clean);v=self.A.transpose(residual)
                hv=self.dwt.synthesis(self.D[i]*self.dwt.analysis(v))
                guided=(v+(1.-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8)
                q=self.zeta[i]*guided
                with torch.no_grad():
                    gamma,cached,row=decide(self,y,uncond.detach(),q.detach(),t,s,x.detach(),self.arm)
                self.last_probe_calls+=row['probe_forwards']
                if nfe_counter is not None:nfe_counter[0]+=row['probe_forwards']
                x=uncond+q if gamma==1. else uncond+gamma*q
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite selected trajectory')
                self.trace.append(dict(step=k,t=t,next_t=s,cached_prediction_reused=bool(hit),**row))
            return x
    return Controlled


def execute(model,operator,y,seed,arm,sigma,size=256,level=3):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(seed);cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    z.arm,z.sigma,z.profile,z.pipeline_start=arm,sigma,None,time.perf_counter()
    if z.tau.cpu().tolist()!=strong.grid():raise ValueError('Strong grid differs')
    losses=z.optimize(y,verbose=False,x0_gt=None)
    return z._last_opt_x0.cpu(),dict(entry=z.entry,trace=z.trace,losses=losses,
        native_last_calls=getattr(z,'last_native_calls',30),probe_calls=getattr(z,'last_probe_calls',0))


def replay(model,operator,y,entry,arm,sigma,size=256,level=3):
    import torch
    cfg=strong.config();cfg.update(img_size=size,level=level)
    z=controller_class()(model,operator,**cfg)
    with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Replay grid differs')
    torch.set_rng_state(entry['cpu_rng'])
    if entry['cuda_rng']:torch.cuda.set_rng_state_all(entry['cuda_rng'])
    z.arm,z.sigma,z.profile,z.unrolls,z.pipeline_start=arm,sigma,None,9,time.perf_counter()
    out=z._reverse_diffusion(y,init_noise=entry['initial'].to(model.device))
    return out.detach().cpu(),z.trace
