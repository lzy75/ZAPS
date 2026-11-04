"""Mandatory allocated-CUDA algebra and Algorithm1 gates. No missing-Torch skips."""
import math
import types
import unittest
import numpy as np
import torch
from utils import pigdm_noisy_r1_runtime as rt
from utils import native_ray_transfer_runtime as shared

class Toy(torch.nn.Module):
 def forward(self,x,t):return torch.cat((.05*x,torch.zeros_like(x)),1)

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not torch.cuda.is_available():raise RuntimeError('A85 requires allocated CUDA numerical gates; no skip')
 def test_dense_operator_adjoint_solve_multichannel(self):
  # Independent dense H includes every channel; catches channel/spectral permutation errors.
  n=8;kernel=np.arange(1,10,dtype=float).reshape(3,3)
  for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
   op=rt.Operator(task,'cuda:0',size=n,seed=17,kernel=kernel,dtype=torch.float64)
   eye=torch.eye(3*n*n,device='cuda:0',dtype=torch.float64).reshape(-1,3,n,n)
   dense=op.H(eye).reshape(3*n*n,-1).T
   x=torch.randn(1,3,n,n,device='cuda:0',dtype=torch.float64)
   y=torch.randn_like(op.H(x))
   self.assertTrue(torch.allclose(op.H(x).flatten(),dense@x.flatten(),atol=1e-11,rtol=1e-11))
   self.assertTrue(torch.allclose(op.Ht(y).flatten(),dense.T@y.flatten(),atol=1e-11,rtol=1e-11))
   lam=.03;exact=dense.T@torch.linalg.solve(dense@dense.T+lam*torch.eye(dense.shape[0],device='cuda:0',dtype=torch.float64),y.flatten())
   self.assertTrue(torch.allclose(op.solve(y,lam).flatten(),exact,atol=1e-10,rtol=1e-10))
   if 'deblur' in task:
    self.assertTrue(torch.allclose(op.H(torch.ones_like(x)),torch.ones_like(x),atol=1e-12))
    # Direct circular cross-correlation with flipped PSF == our circular convolution.
    padded=torch.nn.functional.pad(x,(1,1,1,1),mode='circular')
    k=torch.tensor(kernel/kernel.sum(),device=x.device,dtype=x.dtype).flip((0,1))
    direct=torch.nn.functional.conv2d(padded,k[None,None].repeat(3,1,1,1),groups=3)
    self.assertTrue(torch.allclose(op.H(x),direct,atol=1e-11,rtol=1e-11))
 def test_fixed_state_vjp_is_noisy_paper_gradient(self):
  op=rt.Operator('super_resolution','cuda:0',size=8,dtype=torch.float64)
  x=torch.randn(1,3,8,8,device='cuda:0',dtype=torch.float64,requires_grad=True)
  a=.4;m=(x-math.sqrt(1-a)*(.05*x))/math.sqrt(a);y=torch.randn_like(op.H(x))
  lam=.05**2/(1-a);r=y-op.H(m);v=op.solve(r.detach(),lam)
  pull=torch.autograd.grad((m*v).sum(),x,retain_graph=True)[0]
  loss=.5*(r*(r/(1/16+lam))).sum()
  minus=torch.autograd.grad(loss,x)[0]
  self.assertTrue(torch.allclose(pull,-minus,atol=1e-10,rtol=1e-10))
 def test_published_coefficient_and_forced_one_rng(self):
  model=types.SimpleNamespace(model=Toy().cuda(),device='cuda:0');case=dict(sigma=.05,seed=4321)
  op=rt.Operator('super_resolution','cuda:0',size=8);y=torch.linspace(-.7,.7,12,device='cuda:0').reshape(1,3,2,2)
  outputs=[]
  for arm,force in (('BASE',False),('SHRINK',True),('SHRINK',False)):
   with shared.counters(model.model) as count:
    actual,detail=rt.run('A85',model,rt.Operator('super_resolution','cuda:0',size=8,count=count),y,case,arm,count,test_steps=4,size=8,force_one=force)
   outputs.append((actual,detail,count))
  # Independent noisy likelihood with published RED-diff update coefficient.
  diff=rt.original_diffusion();shared.seed(case['seed']);x=torch.randn(1,3,8,8,device='cuda:0')
  for t,s in zip([750,500,250,0],[500,250,0,-1]):
   a=float(diff.alpha(torch.tensor([t],device='cuda:0'))[0]);b=float(diff.alpha(torch.tensor([s],device='cuda:0'))[0])
   x=x.detach().requires_grad_();eps=.05*x;m=(x-math.sqrt(1-a)*eps)/math.sqrt(a)
   r=(y-op.H(m)).detach();v=op.Ht(r/(1/16+.05**2/(1-a)))
   g=torch.autograd.grad((m*v).sum(),x)[0]
   c1=math.sqrt(max(0.,(1-a/b)*(1-b)/(1-a)));c2=math.sqrt(max(0.,1-b-c1*c1))
   with torch.no_grad():x=math.sqrt(b)*m+c1*torch.randn_like(x)+c2*eps+math.sqrt(a*b)*g
  self.assertTrue(torch.equal(x.cpu(),outputs[0][0]))
  self.assertTrue(torch.equal(outputs[0][0],outputs[1][0]))
  for out,detail,c in outputs:
   self.assertEqual(c['forwards'],4);self.assertEqual(c['input_vjp_attempts'],4)
   self.assertTrue(torch.equal(detail['initial'],outputs[0][1]['initial']))
   self.assertTrue(torch.equal(detail['final_rng']['cpu'],outputs[0][1]['final_rng']['cpu']))
   self.assertTrue(all(torch.equal(a,b) for a,b in zip(detail['final_rng']['cuda'],outputs[0][1]['final_rng']['cuda'])))
  self.assertEqual(outputs[0][2]['module_H_actions'],0)
  self.assertEqual(outputs[2][2]['module_H_actions'],4)
 def test_original_author_update_matches_noisy_extension(self):
  raw=Toy().cuda();model=types.SimpleNamespace(model=raw,device='cuda:0')
  case=dict(sigma=.05,seed=4321);y=torch.linspace(-.7,.7,12,device='cuda:0').reshape(1,3,2,2)
  op=rt.Operator('super_resolution','cuda:0',size=8)
  with shared.counters(raw) as count:
   expected,detail=rt.run('A85R1',model,op,y,case,'BASE',count,test_steps=4,size=8)
  ref=shared.reference('A84',False);diff=rt.original_diffusion()
  class NoisyOperator:
   alpha=None
   def H(self,x):return op.H(x)
   def H_pinv(self,r):return op.solve(r,.05**2/(1-self.alpha))
  noisy=NoisyOperator()
  def denoiser(x,unused,t,scale=1.):
   a=diff.alpha(t).reshape(-1,1,1,1);noisy.alpha=float(a.flatten()[0])
   eps=raw(x,t)[:,:3]
   return eps,(x-(1-a).sqrt()*eps)/a.sqrt()
  algo=object.__new__(ref.pgdm.PGDM);algo.diffusion=diff;algo.model=denoiser;algo.H=noisy
  algo.cfg=types.SimpleNamespace(algo=types.SimpleNamespace(deg='sr4'))
  algo.awd=True;algo.cond_awd=False;algo.mcg=False;algo.grad_term_weight=1.;algo.eta=1.
  algo.initialize=lambda x,*args,**kwargs:x
  shared.seed(case['seed']);initial=torch.randn(1,3,8,8,device='cuda:0')
  states,_=algo.sample(initial,None,[0,250,500,750],y_0=y)
  # Published source uses tensor coefficients and two linear solves vs scalar coefficients/one solve.
  self.assertTrue(torch.allclose(states[0],expected,rtol=3e-4,atol=3e-4))
  self.assertTrue(torch.equal(shared.rng()['cpu'],detail['final_rng']['cpu']))
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(shared.rng()['cuda'],detail['final_rng']['cuda'])))
 def test_real_size_operator_state_and_mask_sensor_count(self):
  for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
   k=np.zeros((61,61));k[30,10:51]=1
   op=rt.Operator(task,'cuda:0',seed=99,kernel=k if task=='motion_deblur' else None)
   twin=rt.Operator(task,'cuda:0',seed=99,kernel=k if task=='motion_deblur' else None)
   self.assertTrue(all(torch.equal(v,twin.state()[n]) for n,v in op.state().items()))
   x=torch.ones(1,3,256,256,device='cuda:0');r=op.H(x)
   self.assertTrue(torch.isfinite(op.solve(r,.0025)).all())
   if task=='inpainting':self.assertEqual(r.numel(),3*int(op.mask.sum()))

if __name__=='__main__':unittest.main()
