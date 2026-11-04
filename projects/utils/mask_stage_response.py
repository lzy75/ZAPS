"""A87 diagnostic-only interventions: fixed blocks, never an online time policy."""
ARMS=('BASE_CHECK','SHRINK_CHECK','BASE_EARLY','BASE_MIDDLE','BASE_LATE')
PROBES=(4,14,24)

def applied_gamma(arm,step,value):
    if arm not in ARMS or not 0<=step<30:raise ValueError('Invalid frozen diagnostic arm/step')
    forced=arm=='BASE_CHECK' or (arm=='BASE_EARLY' and step<10) or (arm=='BASE_MIDDLE' and 10<=step<20) or (arm=='BASE_LATE' and step>=20)
    return 1. if forced else value

def run(model,mask,y,entry,case,arm,observe,size=256,level=3):
    import torch
    from modules.zaps_algorithm import ZAPS,ddpm_posterior_step
    from utils.shrink_multitask import configuration
    from utils.noise_signal_shrink import select
    actions=dict(H=0,adjoint=0,probe_mask=0)
    class Mask(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('p',mask)
        def H(self,x):actions['H']+=1;return self.p*x
        def transpose(self,x):actions['adjoint']+=1;return self.p*x
    z=ZAPS(model,Mask(),**configuration('inpainting',size,level))
    with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Frozen grid differs')
    torch.set_rng_state(entry['cpu_rng'])
    if str(model.device).startswith('cuda'):torch.cuda.set_rng_state_all(entry['cuda_rng'])
    x=entry['initial'].to(model.device);ab=model.alphas_cumprod;trace=[];probes=[];pending=None;probe_calls=0
    observed=mask.bool().expand_as(x)
    def stats(v):
        v=v.detach().double();return dict(observed=float(v[observed].square().sum()),missing=float(v[~observed].square().sum()))
    with torch.no_grad():
        for k,i in enumerate(range(29,-1,-1)):
            t=int(z.tau[i]);s=int(z.tau[i-1]) if i else -1
            eps=model._predict_eps(x,torch.full((1,),t,device=model.device,dtype=torch.long))
            clean=z._tweedie_estimate(x,eps,ab[t],i)
            if pending is not None:
                actions['probe_mask']+=1
                true=mask*(clean-pending['clean_one']);den=float(true.double().norm());entry_probe=dict(step=pending['step'],t=pending['t'],next_t=t,
                    gamma=pending['gamma'],next_clean_current=observe(clean),next_clean_gamma1=observe(pending['clean_one']),
                    actual_observed_change_norm=den,actual_clean_change_energy=stats(clean-pending['clean_one']))
                for label,pred in pending['predictions'].items():
                    pred=pred.double();true64=true.double();pn=float(pred.norm())
                    entry_probe[label]=dict(relative_response_error=float((pred-true64).norm())/den if den>1e-12 else None,
                        cosine=float((pred*true64).sum())/(pn*den) if min(pn,den)>1e-12 else None)
                probes.append(entry_probe);pending=None
            uncond=ddpm_posterior_step(x,clean,t,s,ab,eta=1.,learned_log_var=None,mode='ddpm')
            r=y-z.A.H(clean);v=z.A.transpose(r);hv=z.dwt.synthesis(z.D[i]*z.dwt.analysis(v))
            q=z.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
            rr=r.double()[observed];d=(z.A.H(clean+q/ab[t].sqrt().clamp(min=1e-8))-z.A.H(clean)).double()[observed]
            rule=select(float((rr*d).sum()),float(rr.square().sum()),float(d.square().sum()),rr.numel(),case['sigma'])
            gamma=applied_gamma(arm,k,rule['gamma']);next_x=uncond+q if gamma==1. else uncond+gamma*q
            if arm=='SHRINK_CHECK' and k in PROBES:
                if s<0:raise ValueError('Probe cannot use terminal endpoint')
                x_one=uncond+q
                eps_one=model._predict_eps(x_one,torch.full((1,),s,device=model.device,dtype=torch.long));probe_calls+=1
                clean_one=z._tweedie_estimate(x_one,eps_one,ab[s],i-1)
                actions['probe_mask']+=2
                pending=dict(step=k,t=t,gamma=gamma,clean_one=clean_one.detach(),predictions={
                    'original_same_time_proxy':mask*((gamma-1)*q/ab[t].sqrt()),
                    'next_time_identity_proxy':mask*((gamma-1)*q/ab[s].sqrt())})
            trace.append(dict(step=k,t=t,next_t=s,computed_gamma=rule['gamma'],applied_gamma=gamma,weight=rule['weight'],
                residual_energy=rule.get('residual_energy'),noise_energy=rule['noise_energy'],q_energy=stats(q),applied_q_energy=stats(gamma*q),
                clean_region_error=observe(clean)))
            if not bool(torch.isfinite(next_x).all()):raise RuntimeError('Nonfinite diagnosis; no fallback')
            x=next_x
    if pending is not None:raise ValueError('Unconsumed next-state probe')
    return x.detach().cpu(),dict(trace=trace,probes=probes,probe_forwards=probe_calls,operator_actions=actions,
        final_cpu_rng=torch.get_rng_state().clone(),final_cuda_rng=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else [])
