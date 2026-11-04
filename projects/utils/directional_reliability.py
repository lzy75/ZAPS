"""A91: bandwise frozen-measurement-noise shrinkage of native ZAPS q.

No global projection multiplier, scalar anchor, scalar norm budget, or policy fallback.
"""
import hashlib
import numpy as np
ARMS=('BASE_CHECK','SCALAR_CHECK','DIRECTIONAL')
SIGMAS=(.05,.10)
PROBES=16

def subbands(height,width,level):
    if level<1 or height%(2**level) or width%(2**level):raise ValueError('Invalid original DWT layout')
    out=[(f'LL{level}',(slice(0,height//2**level),slice(0,width//2**level)))]
    for j in range(level,0,-1):
        h=height//2**(j-1);w=width//2**(j-1);hh=h//2;ww=w//2
        out.extend([(f'LH{j}',(slice(0,hh),slice(ww,w))),
                    (f'HL{j}',(slice(hh,h),slice(0,ww))),
                    (f'HH{j}',(slice(hh,h),slice(ww,w)))])
    return out

def calibration_key(c):
    return c['task']+('_mask'+str(c['mask_seed']) if c['task']=='inpainting' else '_shared')

def calibration_seed(key):return int.from_bytes(hashlib.sha256(('A91|'+key).encode()).digest()[:4],'little')

def select(energy,noise):
    """Positive-part moment Wiener weights; no other coefficient is accepted."""
    e=np.asarray(energy,dtype=np.float64);n=np.asarray(noise,dtype=np.float64)
    if e.shape!=n.shape or not np.isfinite(e).all() or not np.isfinite(n).all() or (e<0).any() or (n<0).any():
        raise ValueError('Invalid per-band signal/noise energy')
    return np.maximum(0,1-np.divide(n,e,out=np.ones_like(e),where=e>0))

def draw_probe(shape,generator,device):
    import torch
    return (torch.randint(0,2,shape,generator=generator,dtype=torch.int8).float()*2-1).to(device)

def calibrate(op,dwt,shape,key,device,progress=None):
    """E[(W T xi)^2] = diag(W T T^T W^T); T is actual native mapping.

    One shared bank per fixed operator/mask, independent of data/noise level/D.
    """
    import torch,time
    before=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []
    seed=calibration_seed(key);generator=torch.Generator(device='cpu').manual_seed(seed)
    total=None;half=None;start=time.perf_counter()
    with torch.no_grad():
        for k in range(PROBES):
            probe=draw_probe(shape,generator,device)
            if progress:progress(k+1)
            sample=dwt.analysis(op.transpose(probe)).double().square()
            total=sample.clone() if total is None else total+sample
            if k==PROBES//2-1:half=total.clone()/(PROBES//2)
    if not torch.equal(before,torch.get_rng_state()) or (cuda and not all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))):
        raise RuntimeError('Calibration consumed sampling RNG')
    second=(total-half*(PROBES//2))/(PROBES//2)
    variance=total/PROBES
    if not bool(torch.isfinite(variance).all()) or bool((variance<0).any()):raise RuntimeError('Nonfinite calibration')
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    return dict(unit_variance=variance.float().cpu(),first_half=half.float().cpu(),second_half=second.float().cpu()),dict(
        key=key,seed=seed,probes=PROBES,native_mapping_actions=PROBES,dwt_analysis_actions=PROBES,
        seconds=time.perf_counter()-start,scope='UNIT measurement noise; frozen-mean native-mapping propagation only')

def qualify(op,dwt,shape,key,entry,alphas,device,progress=None):
    """Check coefficient noise propagation vs original q arithmetic, zero NN."""
    import torch
    generator=torch.Generator(device='cpu').manual_seed(calibration_seed(key))
    probe=draw_probe(shape,generator,device)
    if progress:progress(1)
    v=op.transpose(probe);wv=dwt.analysis(v);errors=[]
    for i in (0,len(entry['tau'])-1):
        t=int(entry['tau'][i]);alpha=alphas[t].to(device);a=alpha.sqrt().clamp(min=1e-8)
        D=entry['D'][i].to(device);zeta=entry['zeta'][i].to(device)
        native=zeta*(v+(1-alpha)*dwt.synthesis(D*wv))/a
        analytic=(zeta/a)*(1+(1-alpha)*D)*wv
        actual=dwt.analysis(native)
        err=float((actual.double()-analytic.double()).norm()/analytic.double().norm().clamp(min=1e-30))
        if err>1e-5:raise RuntimeError('Original-q frozen-noise propagation identity failed')
        errors.append(dict(index=i,t=t,relative_error=err))
    return dict(passed=True,errors=errors,native_mapping_actions=1,dwt_analysis_actions=3,dwt_synthesis_actions=2)

def shrink(q,noise_coeff_scale,dwt,bank,sigma):
    import torch
    coeff=dwt.analysis(q);bands=subbands(q.shape[-2],q.shape[-1],dwt.level)
    e=[];n=[];n1=[];n2=[]
    multiplier=noise_coeff_scale.double().square()*(sigma*sigma)
    for _,(ys,xs) in bands:
        sel=(...,ys,xs)
        e.append(coeff[sel].double().square().sum())
        n.append((multiplier[sel]*bank['unit_variance'][sel]).sum())
        n1.append((multiplier[sel]*bank['first_half'][sel]).sum())
        n2.append((multiplier[sel]*bank['second_half'][sel]).sum())
    E=torch.stack(e);N=torch.stack(n);N1=torch.stack(n1);N2=torch.stack(n2)
    weights=torch.where(E>0,(1-N/E.clamp(min=1e-300)).clamp(0,1),torch.zeros_like(E))
    # Use original q exactly at the identity endpoint and exact zero at the zero endpoint.
    if bool((weights==1).all()):correction=q
    elif bool((weights==0).all()):correction=q*0
    else:
        modified=coeff.clone()
        for k,(_,(ys,xs)) in enumerate(bands):modified[...,ys,xs]*=weights[k].to(modified.dtype)
        correction=dwt.synthesis(modified)
    norm=float(correction.double().norm());native=float(q.double().norm())
    if not bool(torch.isfinite(correction).all()) or norm>native*(1+3e-5):raise RuntimeError('Bandwise shrinkage exceeded native correction norm')
    packed=torch.stack((E,N,weights,N1,N2)).cpu().tolist()
    energy,noise,weight,half1,half2=packed
    split=[abs(x-y)/max((x+y)/2,1e-30) for x,y in zip(half1,half2)]
    return correction,dict(module_dwt_analysis=1,module_dwt_synthesis=int(not bool((weights==1).all()) and not bool((weights==0).all())),band_labels=[b[0] for b in bands],weights=weight,band_energy=energy,
        band_noise_energy=noise,band_signal_energy=[max(0,x-y) for x,y in zip(energy,noise)],
        noise_split_relative_difference=split,correction_norm=norm,native_correction_norm=native,
        zero_bands=sum(w==0 for w in weight),identity_bands=sum(w==1 for w in weight),
        proxy='FROZEN_MEAN_MEASUREMENT_NOISE_ONLY',global_multiplier_used=False)

def run(model,op,y,entry,c,arm,actions,size=256,level=3,bank=None):
    import torch
    if arm not in ARMS:raise ValueError('Unknown A91 arm')
    if arm!='DIRECTIONAL':
        from utils import diagonal_feedback as reference
        return reference.run(model,op,y,entry,c,arm,actions,size,level)
    if bank is None:raise ValueError('Direction-only rule requires an operator noise bank')
    from modules.zaps_algorithm import ZAPS,ddpm_posterior_step
    from utils.shrink_multitask import configuration
    z=ZAPS(model,op,**configuration(c['task'],size,level))
    with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Original tau changed')
    noise_bank={k:bank[k].to(model.device) for k in ('unit_variance','first_half','second_half')}
    if any(v.shape!=(1,3,size,size) for v in noise_bank.values()):raise ValueError('Noise coefficient bank shape changed')
    torch.set_rng_state(entry['cpu_rng'])
    if str(model.device).startswith('cuda'):torch.cuda.set_rng_state_all(entry['cuda_rng'])
    x=entry['initial'].to(model.device);ab=model.alphas_cumprod;trace=[]
    for k,i in enumerate(range(29,-1,-1)):
        t=int(z.tau[i]);s=int(z.tau[i-1]) if i else -1
        eps=model._predict_eps(x,torch.full((1,),t,device=model.device,dtype=torch.long))
        clean=z._tweedie_estimate(x,eps,ab[t],i)
        uncond=ddpm_posterior_step(x,clean,t,s,ab,eta=1.,learned_log_var=None,mode='ddpm')
        r=y-op.H(clean);v=op.transpose(r);wv=z.dwt.analysis(v)
        hv=z.dwt.synthesis(z.D[i]*wv);a=ab[t].sqrt().clamp(min=1e-8)
        q=z.zeta[i]*(v+(1-ab[t])*hv)/a
        factor=(z.zeta[i]/a)*(1+(1-ab[t])*z.D[i])
        correction,row=shrink(q,factor,z.dwt,noise_bank,c['sigma'])
        x=uncond+correction
        if not bool(torch.isfinite(x).all()):raise RuntimeError('Nonfinite directional rule; no fallback')
        mask=getattr(op,'_mask_cache',None)
        if mask is not None:
            observed=mask.bool().expand_as(q)
            row.update(native_observed_energy=float(q.double()[observed].square().sum()),
                native_missing_energy=float(q.double()[~observed].square().sum()),
                applied_observed_energy=float(correction.double()[observed].square().sum()),
                applied_missing_energy=float(correction.double()[~observed].square().sum()))
        trace.append(dict(step=k,t=t,next_t=s,**row))
        actions['module_dwt_analysis']=actions.get('module_dwt_analysis',0)+row['module_dwt_analysis']
        actions['module_dwt_synthesis']=actions.get('module_dwt_synthesis',0)+row['module_dwt_synthesis']
        actions['reliability_steps']=actions.get('reliability_steps',0)+1
    return x.detach().cpu(),dict(trace=trace,final_cpu_rng=torch.get_rng_state().clone(),
        final_cuda_rng=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else [])
