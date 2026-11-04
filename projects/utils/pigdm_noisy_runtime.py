"""Paper noisy PiGDM; explicit linear H and exact regularized solves."""
import math
import sys
import types
from unittest.mock import patch
from utils import native_ray_transfer_runtime as shared
from utils import noise_signal_shrink as policy
from utils import pigdm_noisy_transfer as rules

class Operator:
 def __init__(self,task,device,size=256,seed=0,kernel=None,count=None,dtype=None):
  import numpy as np
  import torch
  self.task,self.size,self.count,self.module=task,size,count,False
  self.dtype=dtype or torch.float32;self.device=device
  self.mask=None;self.kernel=None
  if task=='inpainting':
   # Independent local generator does not perturb sampling RNG.
   gen=torch.Generator(device='cpu').manual_seed(seed)
   self.mask=(torch.rand((size,size),generator=gen)>=.7).to(device)
   self.indices=self.mask.flatten().nonzero().flatten()
   if self.indices.numel()==0:raise ValueError('Empty observed mask')
  elif task in ('gaussian_deblur','motion_deblur'):
   if kernel is None:
    if task!='gaussian_deblur':raise ValueError('Frozen motion kernel required')
    axis=np.arange(-30,31,dtype=np.float64);g=np.exp(-.5*(axis/3.)**2);kernel=np.outer(g,g)
   k=torch.as_tensor(kernel,device=device,dtype=self.dtype)
   if k.ndim!=2 or k.shape[0]!=k.shape[1] or k.shape[0]%2!=1 or k.shape[0]>size:raise ValueError('Odd square kernel must fit image')
   if not bool(torch.isfinite(k).all()) or bool((k<0).any()) or float(k.sum())<=0:raise ValueError('Invalid PSF')
   self.kernel=k/k.sum();psf=torch.zeros(size,size,device=device,dtype=self.dtype)
   psf[:k.shape[0],:k.shape[1]]=self.kernel
   psf=torch.roll(psf,(-(k.shape[0]//2),-(k.shape[1]//2)),(0,1))
   self.spectrum=torch.fft.rfft2(psf)
  elif task!='super_resolution':raise ValueError('Unknown task')
  if task=='super_resolution' and size%4:raise ValueError('SR4 needs divisible size')
 def bump(self,name):
  if self.count is not None:
   key=('module_' if self.module else 'base_')+name
   self.count[key]=self.count.get(key,0)+1
 def fft(self,x,adjoint=False):
  import torch
  self.bump('fft_actions');self.bump('ifft_actions')
  k=self.spectrum.conj() if adjoint else self.spectrum
  return torch.fft.irfft2(torch.fft.rfft2(x)*k,s=(self.size,self.size))
 def H(self,x):
  import torch
  self.bump('H_actions')
  if self.task=='super_resolution':return torch.nn.functional.avg_pool2d(x,4)
  if self.task=='inpainting':return x.reshape(x.shape[0],3,-1)[...,self.indices]
  return self.fft(x)
 def Ht(self,y):
  import torch
  self.bump('adjoint_actions')
  if self.task=='super_resolution':return y.repeat_interleave(4,-2).repeat_interleave(4,-1)/16.
  if self.task=='inpainting':
   x=torch.zeros(y.shape[0],3,self.size*self.size,device=y.device,dtype=y.dtype)
   x[...,self.indices]=y;return x.reshape(y.shape[0],3,self.size,self.size)
  return self.fft(y,True)
 def solve(self,r,lam):
  """Exactly Ht(HHt+lam I)^-1 r. No singular-value truncation."""
  import torch
  if not math.isfinite(lam) or lam<=0:raise ValueError('Noisy branch requires positive regularization')
  self.bump('regularized_solves')
  if self.task=='super_resolution':return self.Ht(r/(1./16+lam))
  if self.task=='inpainting':return self.Ht(r/(1.+lam))
  self.bump('adjoint_actions');self.bump('fft_actions');self.bump('ifft_actions')
  return torch.fft.irfft2(self.spectrum.conj()*torch.fft.rfft2(r)/(self.spectrum.abs().square()+lam),s=(self.size,self.size))
 def state(self):
  return {k:v.detach().cpu().clone() for k,v in dict(mask=self.mask,kernel=self.kernel).items() if v is not None}


def original_diffusion():
 ref=shared.reference('A84')
 helper=types.ModuleType('utils.functions');helper.sigmoid=ref.helpers.sigmoid
 with patch.dict(sys.modules,{'utils.functions':helper}):return ref.diffusion.Diffusion()


def run(experiment,model,op,y,case,arm,count,*,progress=None,size=256,test_steps=None,force_one=False):
 import torch
 if arm not in ('BASE','SHRINK'):raise ValueError('No calibration/mismatch arms')
 n=test_steps or rules.STEPS
 if 1000%n:raise ValueError('Uniform integer training-index grid required')
 diffusion=original_diffusion();grid=list(range(0,1000,1000//n))[::-1]
 shared.seed(case['seed']);x=torch.randn(1,3,size,size,device=model.device)
 initial=x.detach().cpu().clone();entry=shared.rng();trace=[]
 for step,t in enumerate(grid):
  s=grid[step+1] if step+1<len(grid) else -1
  a=float(diffusion.alpha(torch.tensor([t],device=model.device))[0]);b=float(diffusion.alpha(torch.tensor([s],device=model.device))[0])
  with torch.enable_grad():
   x=x.detach().requires_grad_();prediction=model.model(x,torch.tensor([t],device=model.device))
   eps=prediction[:,:3];m=(x-math.sqrt(1-a)*eps)/math.sqrt(a)
   with torch.no_grad():
    r=y-op.H(m.detach());v=op.solve(r,case['sigma']**2/(1-a))
   count['input_vjp_attempts']+=1
   g=torch.autograd.grad((m*v.detach()).sum(),x)[0]
  with torch.no_grad():
   q=math.sqrt(a)*g;row=dict(gamma=1.,reason='PAPER_NOISY_PIGDM')
   if arm=='SHRINK':
    op.module=True
    try:d=op.H(q/math.sqrt(a)).double()
    finally:op.module=False
    rr=r.double();row=policy.select(float((rr*d).sum()),float(rr.square().sum()),float(d.square().sum()),rr.numel(),case['sigma'])
    if force_one:row.update(gamma=1.,reason='TEST_FORCED_ONE')
   c1=rules.ETA*math.sqrt(max(0.,(1-a/b)*(1-b)/(1-a)))
   c2=math.sqrt(max(0.,1-b-c1*c1))
   # Draw even at final zero coefficient; same RNG contract in both arms.
   noise=torch.randn_like(x)
   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q
   count['corrections']+=1
   if not bool(torch.isfinite(x).all()):raise RuntimeError('Nonfinite trajectory, no output fallback')
   trace.append(dict(step=step,t=t,s=s,alpha=a,rt_squared=1-a,lambda_noise=case['sigma']**2/(1-a),
     residual_mse=float(r.double().square().mean()),q_norm=float(q.double().norm()),**row))
   if progress and (step+1)%10==0:progress(step+1,count)
  del prediction,eps,m,g,q,v,r
 return x.detach().cpu(),dict(trace=trace,initial=initial,entry_rng=entry,final_rng=shared.rng(),grid=grid)
