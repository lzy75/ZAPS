"""A89 four orthogonal Haar bands and a coupled nonnegative response fit.

Independent of the UNet/ZAPS learned wavelet parameters. No GT or new NN calls.
"""
import itertools
import time
import numpy as np
BANDS=('LOW','COLUMN_DETAIL','ROW_DETAIL','DIAGONAL_DETAIL')
ARMS=('BASE_CHECK','SCALAR_CHECK','DIAG4')
SIGMAS=(.05,.10)

def haar_components(x):
    """Return orthogonal P_b x, sum_b P_b=I; NumPy or Torch arrays."""
    if x.shape[-2]%2 or x.shape[-1]%2:raise ValueError('Even image dimensions required')
    a,b,c,d=x[...,::2,::2],x[...,::2,1::2],x[...,1::2,::2],x[...,1::2,1::2]
    coeff=((a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2)
    signs=((1,1,1,1),(1,-1,1,-1),(1,1,-1,-1),(1,-1,-1,1))
    out=[]
    for v,s in zip(coeff,signs):
        y=x*0
        y[...,::2,::2]=s[0]*v/2;y[...,::2,1::2]=s[1]*v/2
        y[...,1::2,::2]=s[2]*v/2;y[...,1::2,1::2]=s[3]*v/2
        out.append(y)
    return out

def fit(gram,dot,anchor,target_energy):
    """Exact active-face enumeration, <=4 unknowns; ties closest to anchor.

    Variables are image-space correction amplitudes z_b=gamma_b||P_b q||.
    This normalization makes the tie-break an image correction distance.
    Unobserved null directions retain the anchor, not a zero/minimum-norm guess.
    """
    g=np.asarray(gram,dtype=np.float64);b=np.asarray(dot,dtype=np.float64)
    a=np.asarray(anchor,dtype=np.float64);n=len(a)
    if g.shape!=(n,n) or b.shape!=(n,) or not 1<=n<=4:raise ValueError('Need a <=4D Gram system')
    if not all(np.isfinite(v).all() for v in (g,b,a)) or not np.isfinite(target_energy) or target_energy<0 or (a<0).any():
        raise ValueError('Nonfinite/negative fit inputs')
    if not np.allclose(g,g.T,rtol=1e-10,atol=1e-20):raise ValueError('Nonsymmetric response Gram')
    g=(g+g.T)/2;scale=max(float(np.linalg.norm(g,2)),1e-30)
    if np.linalg.eigvalsh(g).min() < -1e-10*scale:raise ValueError('Response Gram not positive semidefinite')
    best=None;objective_tol=1e-10*max(float(target_energy),1e-20)
    for flags in itertools.product((False,True),repeat=n):
        free=np.flatnonzero(flags);z=np.zeros(n)
        if len(free):
            gg=g[np.ix_(free,free)];aa=a[free]
            z[free]=aa+np.linalg.pinv(gg,rcond=1e-12,hermitian=True)@(b[free]-gg@aa)
        tol=1e-8*max(float(np.linalg.norm(z)),float(np.linalg.norm(a)),1e-20)
        if (z < -tol).any():continue
        z=np.maximum(z,0);gradient=g@z-b
        ktol=1e-7*max(float(np.linalg.norm(b)),float(np.linalg.norm(g@z)),1e-20)
        if len(free) and float(np.abs(gradient[free]).max())>ktol:continue
        fixed=np.flatnonzero(np.logical_not(flags))
        if len(fixed) and float(gradient[fixed].min()) < -ktol:continue
        value=float(target_energy-2*b@z+z@g@z);distance=float(np.sum((z-a)**2))
        if best is None or value < best[0]-objective_tol or (abs(value-best[0])<=objective_tol and distance<best[1]):
            best=(value,distance,z.copy())
    if best is None:raise RuntimeError('No numerically qualified NNLS face; stop, no policy fallback')
    original=float(target_energy-2*b@a+a@g@a)
    if best[0]>original+10*objective_tol:raise RuntimeError('Matrix proxy fit worse than feasible scalar anchor')
    return dict(amplitudes=best[2].tolist(),target_objective=max(0.,best[0]),
        scalar_target_objective=max(0.,original),anchor_distance_squared=best[1],
        rank=int(np.linalg.matrix_rank(g,tol=1e-12*scale)),
        gram_eigenvalues=np.linalg.eigvalsh(g).tolist(),solver='NONNEGATIVE_ACTIVE_FACES_MAX16')

def apply(op,r,q,a,sigma,scalar,actions):
    import torch
    from utils.shrink_multitask_runtime import informative
    parts=haar_components(q);norms=torch.stack([v.double().norm() for v in parts])
    response=[]
    for v,n in zip(parts,norms):
        actions['diagonal_H']+=1
        # H is linear; normalization only affects the small solve, not the correction.
        unit=v/n.to(v.dtype) if float(n)>0 else v*0
        response.append(informative(op.H(unit).double()/float(a),op))
    u=torch.stack(response,dim=1);rr=informative(r.double(),op)
    omega=scalar['weight'];gram=(u.T@u).cpu().numpy();dot=(u.T@(omega*rr)).cpu().numpy()
    nn=norms.cpu().numpy();anchor=scalar['gamma']*nn
    start=time.perf_counter()
    solved=fit(gram,dot,anchor,omega*omega*float(rr.square().sum()))
    solved['small_solve_seconds']=time.perf_counter()-start
    z=np.asarray(solved.pop('amplitudes'));gains=np.divide(z,nn,out=np.full(4,scalar['gamma']),where=nn>0)
    correction=sum(float(g)*v for g,v in zip(gains,parts))
    if not bool(torch.isfinite(correction).all()):raise RuntimeError('Nonfinite diagonal correction')
    return correction,dict(**solved,gamma_bands=gains.tolist(),band_norms=nn.tolist(),
        gamma_scalar=scalar['gamma'],weight=omega,active_measurements=rr.numel(),
        correction_norm=float(correction.double().norm()),scalar_correction_norm=float(scalar['gamma']*norms.norm()),
        scalar_residual_energy=scalar.get('residual_energy'),noise_energy=scalar['noise_energy'])

def run(model,op,y,entry,c,arm,actions,size=256,level=3):
    import torch
    from utils import shrink_multitask_runtime as old
    if arm not in ARMS:raise ValueError('Unknown A89 arm')
    if arm!='DIAG4':
        output,trace=old.replay(model,op,y,c,entry,'NULL' if arm=='BASE_CHECK' else 'SHRINK',size,level)
    else:
        from modules.zaps_algorithm import ZAPS,ddpm_posterior_step
        from utils.shrink_multitask import configuration
        from utils.noise_signal_shrink import select
        z=ZAPS(model,op,**configuration(c['task'],size,level))
        with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
        if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Original grid changed')
        torch.set_rng_state(entry['cpu_rng'])
        if str(model.device).startswith('cuda'):torch.cuda.set_rng_state_all(entry['cuda_rng'])
        x=entry['initial'].to(model.device);ab=model.alphas_cumprod;trace=[]
        for k,i in enumerate(range(29,-1,-1)):
            t=int(z.tau[i]);s=int(z.tau[i-1]) if i else -1
            eps=model._predict_eps(x,torch.full((1,),t,device=model.device,dtype=torch.long))
            clean=z._tweedie_estimate(x,eps,ab[t],i)
            uncond=ddpm_posterior_step(x,clean,t,s,ab,eta=1.,learned_log_var=None,mode='ddpm')
            r=y-op.H(clean);v=op.transpose(r);hv=z.dwt.synthesis(z.D[i]*z.dwt.analysis(v))
            a=ab[t].sqrt().clamp(min=1e-8);q=z.zeta[i]*((v+(1-ab[t])*hv)/a)
            rr=old.informative(r.double(),op)
            d=old.informative((op.H(clean+q/a)-op.H(clean)).double(),op)
            scalar=select(float((rr*d).sum()),float(rr.square().sum()),float(d.square().sum()),rr.numel(),c['sigma'])
            correction,row=apply(op,r,q,a,c['sigma'],scalar,actions)
            x=uncond+correction
            if not bool(torch.isfinite(x).all()):raise RuntimeError('Nonfinite A89 trajectory; no fallback')
            trace.append(dict(step=k,t=t,next_t=s,**row))
        output=x.detach().cpu()
    return output,dict(trace=trace,final_cpu_rng=torch.get_rng_state().clone(),
        final_cuda_rng=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else [])
