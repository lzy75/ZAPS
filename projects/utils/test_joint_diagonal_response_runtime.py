"""Required actual Torch: nonlinear chain, coupling, objective and unchanged continuation."""
import unittest
from unittest.mock import patch
import torch
from modules.wavelet import OrthogonalDWT2D
from utils import joint_diagonal_response as r,overall_direction_audit as old
from utils.test_overall_direction_audit_runtime import model
from utils.test_mask_stage_response_runtime import Mask
from utils.shrink_multitask import configuration
from modules.zaps_algorithm import ZAPS
from utils.paper_clock_runtime import counters,snapshot


class CoupledOperator:
 def H(self,x):return .6*x+.4*x.roll(1,-1)


class Tests(unittest.TestCase):
 def fixture(self):
  torch.manual_seed(12);m=model();op=CoupledOperator();q=.003*torch.randn(1,3,32,32);origin=.01*torch.randn_like(q);y=.2+origin*0
  state=dict(uncond=origin,q=q,next_t=309)
  physical,clean=old.physical_reference(m,op,y,state,dict(input_vjp_attempts=0))
  return m,op,y,dict(state=state,physical=physical,reference_clean=clean),OrthogonalDWT2D('db4',3)
 def test_true_joint_chain_and_nonzero_cross_coefficient_derivative(self):
  m,op,y,bundle,dwt=self.fixture();c=dwt.analysis(bundle['state']['q']);origin=bundle['state']['uncond'];sigma=.05
  torch.testing.assert_close(dwt.synthesis(c),bundle['state']['q'],rtol=3e-5,atol=3e-7)
  self.assertTrue(bool((dwt.synthesis(c*0)==0).all()))
  count=dict(input_vjp_attempts=0)
  value,grad=r.data_value(m,op,y,origin,309,count,True)
  expected=c*dwt.analysis(grad)/(sigma*sigma)
  cached=-c*dwt.analysis(bundle['physical'])/(sigma*sigma)
  torch.testing.assert_close(expected,cached,rtol=2e-5,atol=2e-6)
  direction=torch.randn_like(c);h=.01
  a,_=r.data_value(m,op,y,origin+dwt.synthesis(c*(h*direction)),309,count)
  b,_=r.data_value(m,op,y,origin-dwt.synthesis(c*(h*direction)),309,count)
  fd=(a-b)/(2*h*sigma*sigma);analytic=float((expected.double()*direction.double()).sum())
  self.assertAlmostEqual(fd,analytic,delta=3e-3*max(1.,abs(analytic)))
  probe=torch.zeros_like(c);j=int(c.abs().reshape(-1).argmax());probe.reshape(-1)[j]=.2
  _,grad2=r.data_value(m,op,y,origin+dwt.synthesis(c*probe),309,count,True)
  change=(c*dwt.analysis(grad2)/(sigma*sigma)-expected).reshape(-1);change[j]=0
  self.assertGreater(float(change.abs().max()),1e-9)
 def test_solver_monotonicity_signs_and_rng_without_model_updates(self):
  m,op,y,bundle,dwt=self.fixture()
  for arm in r.ARMS:
   before=torch.get_rng_state().clone()
   with counters(m.model) as count:correction,gamma,detail=r.solve(m,op,y,bundle,dwt,.05,arm,count)
   self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(torch.isfinite(correction).all())
   self.assertLessEqual(count['forwards'],13);self.assertLessEqual(count['input_vjp_attempts'],1)
   for row in detail['trace']:
    if row.get('accepted'):self.assertLessEqual(row['objective'],row['old_objective']+row['tolerance'])
   self.assertLessEqual(detail['final_data']+detail['final_prior'],detail['initial_data']+1e-3)
   if arm=='NONNEG_JOINT':self.assertTrue(bool((gamma>=0).all()))
   if arm=='SIGNED_JOINT':self.assertGreater(detail['negative_fraction'],0.)
   if arm=='SCALAR_TIED':self.assertTrue(bool((gamma==gamma.reshape(-1)[0]).all()))
 def test_continuation_and_rng_are_parent_when_forced_correction_is_identical(self):
  torch.manual_seed(63);m=model();mask=(torch.rand(1,1,32,32)>.7).float();op=Mask(mask)
  y=torch.rand(1,3,32,32)*mask;z=ZAPS(m,op,**configuration('inpainting',32,3));entry=snapshot(z,torch.randn_like(y))
  state=old.prefix(m,op,y,entry,'inpainting',32,3);bundle=dict(state=state)
  expected,first,rng=old.continue_run(m,z,y,bundle['state'],bundle['state']['q'])
  fake=(bundle['state']['q'],torch.ones_like(bundle['state']['q']),dict(trace=[]))
  with patch.object(r,'solve',return_value=fake),counters(m.model) as count:
   output,actual_first,actual_rng,_,_,_=r.run(m,op,y,entry,bundle,'inpainting','SIGNED_JOINT',.1,count,size=32,level=3)
  torch.testing.assert_close(output,expected,rtol=0,atol=0);torch.testing.assert_close(actual_first,first,rtol=0,atol=0)
  self.assertTrue(torch.equal(rng['cpu_rng'],actual_rng['cpu_rng']));self.assertEqual(count['forwards'],14)
  self.assertEqual(count['input_vjp_attempts'],0)

if __name__=='__main__':unittest.main()
