"""Required Torch: actual nonlinear derivatives, endpoints, coarse freezing and RNG."""
import types,unittest
from unittest.mock import patch
import torch
from modules.zaps_algorithm import ZAPS
from modules.wavelet import OrthogonalDWT2D
from utils import fast_response_refinement as r,response_target_runtime as rt,overall_direction_audit as old
from utils.test_overall_direction_audit_runtime import model
from utils.test_full_unroll_response_runtime import Linear
from utils.test_mask_stage_response_runtime import Mask
from utils.shrink_multitask import configuration
from utils.paper_clock_runtime import counters,snapshot
from utils.noise_signal_shrink import select

class Tests(unittest.TestCase):
 def test_closed_form_is_exact_next_proxy_no_extra_derivative_or_rng(self):
  torch.manual_seed(42);m=model();op=Linear();x=torch.zeros(1,3,32,32);q=torch.randn_like(x)*.01;y=torch.ones_like(x)*.2
  z=types.SimpleNamespace(dwt=OrthogonalDWT2D('db4',3));clean=rt.predict_clean(m,x,309);a,b=rt.original_scalar(m,op,y,clean,q,309)
  info=select(float((a*b).sum()),float(a.square().sum()),float(b.square().sum()),a.numel(),.05);rng=old.rng_state('cpu')
  with counters(m.model) as count:delta,detail,_=r.action(m,op,y,z,x,q,x,333,309,'SCALAR_K0',count)
  torch.testing.assert_close(delta,info['gamma']*q);self.assertEqual(count['forwards'],1);self.assertEqual(count['input_vjp_attempts'],0)
  self.assertTrue(rt.rng_equal(rng,old.rng_state('cpu')))
 def test_real_next_response_updates_decrease_target_and_keep_coarse_fine_coefficients(self):
  torch.manual_seed(43);m=model();op=Linear();x=torch.zeros(1,3,32,32);q=torch.randn_like(x)*.01;y=torch.ones_like(x)*.2
  z=types.SimpleNamespace(dwt=OrthogonalDWT2D('db4',3));rng=old.rng_state('cpu')
  for arm in r.ARMS[1:]:
   with counters(m.model) as count:delta,detail,trace=r.action(m,op,y,z,x,q,x,333,309,arm,count)
   d=detail['diagonal'];self.assertLessEqual(d['final_value'],d['initial_value']);self.assertLessEqual(count['forwards'],1+4*r.STEPS[arm])
   self.assertEqual(count['input_vjp_attempts'],count.get('input_vjp_completed',0));self.assertEqual(count['optimizer_backward_attempts'],0)
   self.assertTrue(rt.rng_equal(rng,old.rng_state('cpu')))
   if arm=='COARSE_K3':
    initial=q*detail['scalar']['gamma'];fine=z.dwt.analysis(delta-initial);h,w=fine.shape[-2:]
    high=fine.clone();high[...,:h//4,:w//4]=0
    self.assertLessEqual(float(high.norm()),1e-5*max(1.,float(delta.norm())))
 def test_terminal_all_arms_use_only_operator_and_zero_native_is_zero(self):
  torch.manual_seed(44);m=model();op=Linear();x=torch.zeros(1,3,32,32);q=torch.randn_like(x)*.01;y=torch.ones_like(x)*.2;z=types.SimpleNamespace(dwt=OrthogonalDWT2D('db4',3))
  with patch.object(m,'_predict_eps',side_effect=RuntimeError('Forbidden negative-time NN')):
   for arm in r.ARMS:
    delta,detail,_=r.action(m,op,y,z,x,q,x,0,-1,arm,{})
    self.assertTrue(torch.isfinite(delta).all())
    zero,detail,_=r.action(m,op,y,z,x,q*0,x,0,-1,arm,{})
    self.assertEqual(detail['status'],'ZERO_NATIVE_FEASIBLE_SET');self.assertEqual(float(zero.norm()),0.)
 def test_all30_rng_endpoint_and_registered_small_budget(self):
  torch.manual_seed(45);m=model();n=8;mask=(torch.rand(1,1,n,n)>.7).float();op=Mask(mask);y=torch.rand(1,3,n,n)*mask
  z=ZAPS(m,op,**configuration('inpainting',n,1));entry=snapshot(z,torch.randn_like(y));z=old.make_z(m,op,entry,'inpainting',n,1)
  old.restore_rng(entry,'cpu');x=entry['initial'].clone()
  with torch.no_grad():
   for i in range(29,-1,-1):b,q,_=old.native_parts(m,z,x,y,i);x=b+q
  expected_rng=old.rng_state('cpu');old.restore_rng(entry,'cpu');x=entry['initial'].clone()
  with counters(m.model) as count:
   for k in range(30):x,rng,detail,_,_=r.step(m,z,op,y,x,k,'SCALAR_K0',count)
  self.assertEqual(count['forwards'],59);self.assertTrue(rt.rng_equal(rng,expected_rng));self.assertEqual(detail['next_t'],-1)

if __name__=='__main__':unittest.main()
