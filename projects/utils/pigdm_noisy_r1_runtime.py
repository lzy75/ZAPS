"""A85R1: explicit published update coefficient plus per-step failure diagnostics."""
import math
from utils import pigdm_noisy_r1 as rules
from utils import native_ray_transfer_runtime as shared
from utils import noise_signal_shrink as policy
from utils.pigdm_noisy_runtime import Operator,original_diffusion

class NumericalFailure(RuntimeError):
 def __init__(self,trace):
  self.trace=trace
  super().__init__('Nonfinite trajectory at t='+str(trace[-1]['t'])+'; failure_trace.json retained; no fallback')

def stat(x):
 import torch
 finite=bool(torch.isfinite(x).all())
 if not finite:return dict(finite=False,rms=None,norm=None,max_abs=None)
 y=x.detach().double()
 return dict(finite=True,rms=float(y.square().mean().sqrt()),norm=float(y.norm()),max_abs=float(y.abs().max()))

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
  input_stat=stat(x)
  with torch.enable_grad():
   x=x.detach().requires_grad_();prediction=model.model(x,torch.tensor([t],device=model.device))
   eps=prediction[:,:3];m=(x-math.sqrt(1-a)*eps)/math.sqrt(a)
   with torch.no_grad():
    r=y-op.H(m.detach());v=op.solve(r,case['sigma']**2/(1-a))
   count['input_vjp_attempts']+=1
   g=torch.autograd.grad((m*v.detach()).sum(),x)[0]
  with torch.no_grad():
   q=math.sqrt(a*b)*g;row=dict(gamma=1.,reason='PUBLISHED_COEFFICIENT_NOISY_PIGDM')
   if not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(m).all()):
    trace.append(dict(step=step,t=t,s=s,stage='CLEAN_OR_VJP',input_state=input_stat,epsilon=stat(eps),clean=stat(m),pullback=stat(g),correction=stat(q)))
    raise NumericalFailure(trace)
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
   trace.append(dict(step=step,t=t,s=s,alpha=a,rt_squared=1-a,lambda_noise=case['sigma']**2/(1-a),
     residual_rms=stat(r)['rms'],q_norm=stat(q)['norm'],input_state=input_stat,
     epsilon=stat(eps),clean=stat(m),pullback=stat(g),correction=stat(q),next_state=stat(x),**row))
   if not bool(torch.isfinite(x).all()):raise NumericalFailure(trace)
   if progress and (step+1)%10==0:progress(step+1,count)
  del prediction,eps,m,g,q,v,r
 return x.detach().cpu(),dict(trace=trace,initial=initial,entry_rng=entry,final_rng=shared.rng(),grid=grid)
