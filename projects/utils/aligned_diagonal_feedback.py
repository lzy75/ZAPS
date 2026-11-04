"""A90: original db4 coordinates, norm-bounded nonnegative least squares.

QR compression and established SciPy NNLS; no inversion of a response Gram.
"""
import time
from unittest.mock import patch
import numpy as np
from utils import diagonal_feedback as old
ARMS=('BASE_CHECK','SCALAR_CHECK','HAAR4_TRUST','DB4_4_TRUST','DB4_10_TRUST')
SIGMAS=old.SIGMAS

def subbands(height,width,level):
    if level<1 or height%(2**level) or width%(2**level):raise ValueError('Invalid multiscale layout')
    bands=[(f'LL{level}',(slice(0,height//2**level),slice(0,width//2**level)))]
    for j in range(level,0,-1):
        h=height//2**(j-1);w=width//2**(j-1);hh=h//2;ww=w//2
        bands.extend([(f'LH{j}',(slice(0,hh),slice(ww,w))),
                      (f'HL{j}',(slice(hh,h),slice(0,ww))),
                      (f'HH{j}',(slice(hh,h),slice(ww,w)))])
    return bands

def db4_components(q,dwt,group_level=None):
    coefficients=dwt.analysis(q);parts=[];labels=[]
    for label,(ys,xs) in subbands(q.shape[-2],q.shape[-1],dwt.level if group_level is None else group_level):
        c=coefficients*0;c[...,ys,xs]=coefficients[...,ys,xs]
        parts.append(dwt.synthesis(c));labels.append(label)
    return parts,labels

def fit(matrix,target,anchor):
    """Solve min ||R z-c||², z>=0, ||z||<=||anchor||.

    The scalar anchor sets a correction-energy budget, not a fitted lambda.
    Damping is a nonnegative dual multiplier, located by norm monotonicity.
    All linear-algebra inputs are <=10D QR-compressed arrays.
    """
    from scipy.optimize import nnls
    a=np.asarray(matrix,dtype=np.float64);b=np.asarray(target,dtype=np.float64)
    anchor=np.asarray(anchor,dtype=np.float64);n=anchor.size;radius=float(np.linalg.norm(anchor))
    if a.ndim!=2 or a.shape[1]!=n or b.shape!=(a.shape[0],) or not 1<=n<=10:
        raise ValueError('Need QR-compressed <=10D response')
    if not all(np.isfinite(v).all() for v in (a,b,anchor)) or (anchor<0).any():raise ValueError('Invalid trust-region inputs')
    if radius==0:
        return dict(amplitudes=np.zeros(n).tolist(),dual_lambda=0.,dual_steps=0,nnls_calls=0,
            correction_budget=0.,norm_ratio=0.,projected_kkt_residual=0.,solver_status='ZERO_SCALAR_BUDGET',
            target_objective=float(b@b),scalar_target_objective=float(b@b))
    scale=float(np.linalg.norm(a,2))
    if scale==0:
        return dict(amplitudes=anchor.tolist(),dual_lambda=0.,dual_steps=0,nnls_calls=0,
            correction_budget=radius,norm_ratio=1.,projected_kkt_residual=0.,solver_status='ZERO_RESPONSE_KEEP_SCALAR',
            target_objective=float(b@b),scalar_target_objective=float(b@b))
    aa=a/scale;bb=b/(scale*radius);calls=0
    def solve(lam):
        nonlocal calls
        calls+=1
        if lam:
            mat=np.vstack((aa,np.sqrt(lam)*np.eye(n)));rhs=np.r_[bb,np.zeros(n)]
        else:mat,rhs=aa,bb
        x,_=nnls(mat,rhs,maxiter=100*n)
        return x
    x=solve(0.);lam=0.;iterations=0
    if np.linalg.norm(x)>1:
        lo=0.;hi=max(1.,float(np.linalg.norm(aa.T@bb)));qualified=False
        for _ in range(64):
            x=solve(hi);iterations+=1
            if np.linalg.norm(x)<=1:qualified=True;break
            lo=hi;hi*=2
        if not qualified:raise RuntimeError('Trust-region dual bracket budget exhausted')
        for _ in range(64):
            mid=(lo+hi)/2;trial=solve(mid);iterations+=1
            if np.linalg.norm(trial)>1:lo=mid
            else:hi=mid;x=trial
            if 1-float(np.linalg.norm(x))<=1e-7:break
        else:raise RuntimeError('Trust-region norm criterion not met within64 dual iterations')
        lam=hi
    norm=float(np.linalg.norm(x));gradient=aa.T@(aa@x-bb)+lam*x
    projected=x-np.maximum(0,x-gradient)
    kkt=float(np.linalg.norm(projected)/max(1.,float(np.linalg.norm(aa.T@bb))))
    if norm>1+1e-7 or kkt>1e-6:raise RuntimeError('Trust-region feasibility/KKT failed')
    z=x*radius;value=float(np.linalg.norm(a@z-b)**2);reference=float(np.linalg.norm(a@anchor-b)**2)
    if value>reference+1e-6*max(reference,float(b@b),1e-24):raise RuntimeError('Qualified fit worse than feasible scalar anchor')
    return dict(amplitudes=z.tolist(),dual_lambda=lam*scale*scale,dual_steps=iterations,nnls_calls=calls,
        correction_budget=radius,norm_ratio=norm,projected_kkt_residual=kkt,
        solver_status='KKT_AND_NORM_QUALIFIED',target_objective=value,scalar_target_objective=reference)

def apply(op,r,q,a,sigma,scalar,actions,*,basis,dwt,group_level=None):
    import torch
    from utils.shrink_multitask_runtime import informative
    labels=list(old.BANDS) if basis=='haar' else [v[0] for v in subbands(q.shape[-2],q.shape[-1],dwt.level if group_level is None else group_level)]
    if scalar['gamma']==0:
        return q*0,dict(gamma_bands=[0.]*len(labels),band_labels=labels,gamma_scalar=0.,weight=scalar['weight'],
            solver_status='ZERO_SCALAR_BUDGET',nnls_calls=0,dual_steps=0,dual_lambda=0.,norm_ratio=0.,
            correction_norm=0.,scalar_correction_norm=0.,small_solve_seconds=0.,qr_seconds=0.,
            active_measurements=informative(r,op).numel())
    if basis=='haar':parts=old.haar_components(q)
    else:parts,labels=db4_components(q,dwt,group_level)
    reconstruction=sum(parts)
    error=float((reconstruction.double()-q.double()).norm()/q.double().norm().clamp(min=1e-30))
    if error>1e-5:raise RuntimeError('Original-coordinate decomposition identity failed')
    norms=torch.stack([v.double().norm() for v in parts]);responses=[]
    for v,n in zip(parts,norms):
        actions['diagonal_H']+=1
        unit=v/n.to(v.dtype) if float(n)>0 else v*0
        responses.append(informative(op.H(unit).double()/float(a),op))
    u=torch.stack(responses,dim=1);target=scalar['weight']*informative(r.double(),op)
    # QR avoids explicitly squaring the response condition number through U.T@U.
    start=time.perf_counter();orth,small=torch.linalg.qr(u,mode='reduced')
    compressed=orth.T@target
    rr=small.cpu().numpy();cc=compressed.cpu().numpy();qr_seconds=time.perf_counter()-start
    norm_array=norms.cpu().numpy();anchor=scalar['gamma']*norm_array
    start=time.perf_counter();solved=fit(rr,cc,anchor);solve_seconds=time.perf_counter()-start
    z=np.array(solved.pop('amplitudes'));gains=np.divide(z,norm_array,out=np.full(len(parts),scalar['gamma']),where=norm_array>0)
    correction=sum(float(g)*v for g,v in zip(gains,parts))
    actual=float(correction.double().norm());reference=float(scalar['gamma']*q.double().norm())
    if not bool(torch.isfinite(correction).all()) or actual>reference*(1+3e-5):raise RuntimeError('Actual image correction exceeds scalar energy budget')
    return correction,dict(**solved,gamma_bands=gains.tolist(),band_labels=labels,gamma_scalar=scalar['gamma'],
        weight=scalar['weight'],band_norms=norm_array.tolist(),correction_norm=actual,
        scalar_correction_norm=reference,recomposition_relative_error=error,qr_seconds=qr_seconds,
        small_solve_seconds=solve_seconds,active_measurements=target.numel())

def run(model,op,y,entry,c,arm,actions,size=256,level=3):
    if arm not in ARMS:raise ValueError('Unknown A90 arm')
    if arm in ('BASE_CHECK','SCALAR_CHECK'):return old.run(model,op,y,entry,c,arm,actions,size,level)
    from modules.wavelet import OrthogonalDWT2D
    dwt=OrthogonalDWT2D(wave='db4',level=level).to(model.device) if arm in ('DB4_4_TRUST','DB4_10_TRUST') else None
    def controlled(*args,**kwargs):return apply(*args,**kwargs,basis='db4' if dwt is not None else 'haar',dwt=dwt,group_level=1 if arm=='DB4_4_TRUST' else None)
    with patch.object(old,'apply',controlled):return old.run(model,op,y,entry,c,'DIAG4',actions,size,level)
