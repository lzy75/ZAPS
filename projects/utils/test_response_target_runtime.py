"""Mandatory actual Torch: exact JVP, checkpoint scope, coupled LM, native continuation."""
import importlib.util,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from modules.wavelet import OrthogonalDWT2D
from modules.diffusion_model import _precompute_diffusion_params
from modules.zaps_algorithm import ZAPS
from utils import response_target_runtime as rt,response_target_lm as rules,overall_direction_audit as old
from utils.shrink_multitask import configuration
from utils.paper_clock_runtime import counters,snapshot
from utils.test_mask_stage_response_runtime import Mask
from utils.a67_model_freeze import frozen_network

spec=importlib.util.spec_from_file_location('a95_legacy_checkpoint',Path(__file__).resolve().parents[2]/'DPS/guided_diffusion/nn.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy);checkpoint=legacy.checkpoint


class Toy(torch.nn.Module):
 def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.6))
 def forward(self,x,t):
  eps=checkpoint(lambda a:self.scale*a+.05*torch.sin(a),(x,),self.parameters(),True)
  return torch.cat((eps,torch.zeros_like(eps)),1)
class Operator:
 def H(self,x):return .7*x+.3*x.roll(1,-1)


def model():
 m=types.SimpleNamespace(device='cpu',model=Toy(),alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod'],num_steps=1000)
 m._predict_eps=lambda x,t:m.model(x,t)[:,:3]
 return m


class Tests(unittest.TestCase):
 def test_checkpoint_scope_exact_analytic_jvp_and_adjoint_restore(self):
  torch.manual_seed(12);m=model();op=Operator();origin=.01*torch.randn(1,3,8,8);q=.02*torch.randn_like(origin)
  count=dict(input_vjp_attempts=0);dwt=OrthogonalDWT2D('db4',1);before=checkpoint;weight=m.model.scale.detach().clone()
  with frozen_network(m.model):
   for scalar in (False,True):
    geometry=rt.Geometry(q,dwt,scalar,count);p=torch.zeros(1) if scalar else torch.zeros_like(q)
    direction=torch.randn_like(p);output_weight=torch.randn(q.numel());target=output_weight*0
    provider=rt.Provider(m,op,origin,309,geometry,target,count)
    rng=torch.get_rng_state().clone();jv=provider.jvp(p,direction);g=provider.vjp(p,output_weight)
    ab=m.alphas_cumprod[309];jac=(1-(1-ab).sqrt()*(.6+.05*origin.cos()))/ab.sqrt()
    expected=op.H(jac*geometry.image(direction)).reshape(-1)
    torch.testing.assert_close(jv,expected,rtol=3e-5,atol=3e-6)
    self.assertAlmostEqual(rules.dot(jv,output_weight),rules.dot(direction,g),delta=3e-5*max(1.,abs(rules.dot(jv,output_weight))))
    self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertIs(checkpoint,before)
    scaled=rt.Geometry(7*q,dwt,scalar,count)
    torch.testing.assert_close(scaled.image(direction),geometry.image(direction),rtol=3e-5,atol=3e-6)
  self.assertTrue(torch.equal(weight,m.model.scale));self.assertIsNone(m.model.scale.grad);self.assertTrue(m.model.scale.requires_grad)
 def test_real_nonlinear_solve_and_scalar_initializer_share_objective(self):
  torch.manual_seed(16);m=model();op=Operator();origin=.01*torch.randn(1,3,8,8);q=.01*torch.randn_like(origin);dwt=OrthogonalDWT2D('db4',1)
  count=dict(input_vjp_attempts=0)
  with frozen_network(m.model),torch.no_grad():base=op.H(rt.predict_clean(m,origin,309)).reshape(-1);target=base+.01*torch.sin(torch.arange(base.numel()))
  with frozen_network(m.model),counters(m.model) as count:
   geometry=rt.Geometry(q,dwt,True,count);provider=rt.Provider(m,op,origin,309,geometry,target,count)
   scalar,si=rules.lm(provider,torch.zeros(1),geometry.project)
   full_geometry=rt.Geometry(q,dwt,False,count);full_provider=rt.Provider(m,op,origin,309,full_geometry,target,count)
   full,fi=rules.lm(full_provider,full_geometry.c*(float(scalar[0])/full_geometry.qnorm),full_geometry.project)
  self.assertTrue(si['converged']);self.assertTrue(fi['converged']);self.assertLessEqual(fi['final_value'],si['final_value']+1e-8)
  self.assertEqual(count['input_vjp_attempts'],count['input_vjp_completed']);self.assertEqual(count['input_jvp_attempts'],count['input_jvp_completed'])
  self.assertEqual(count['optimizer_backward_attempts'],0)
 def test_shared_current_state_and_forced_one_native_continuation(self):
  torch.manual_seed(63);m=model();mask=(torch.rand(1,1,8,8)>.7).float();op=Mask(mask);y=torch.rand(1,3,8,8)*mask
  z=ZAPS(m,op,**configuration('inpainting',8,1));entry=snapshot(z,torch.randn_like(y))
  with frozen_network(m.model):
   state=old.prefix(m,op,y,entry,'inpainting',8,1);physical,clean=old.physical_reference(m,op,y,state,dict(input_vjp_attempts=0))
   bundle=dict(state=state,physical=physical,reference_clean=clean)
   with counters(m.model) as count:current,gate=rt.shared_prefix(m,op,y,entry,bundle,'inpainting',count,size=8,level=1)
   self.assertEqual(count['forwards'],16)
   expected,_,rng=old.continue_run(m,z,y,state,state['q'])
   with patch('utils.noise_signal_shrink.select',return_value=dict(gamma=1.)),counters(m.model) as count:
    output,_,actual_rng,_,_=rt.run(m,op,y,entry,bundle,current,'inpainting','LEGACY_SCALAR',.05,count,size=8,level=1)
   torch.testing.assert_close(output,expected,rtol=0,atol=0);self.assertTrue(rt.rng_equal(rng,actual_rng));self.assertEqual(count['forwards'],14)

if __name__=='__main__':unittest.main()
