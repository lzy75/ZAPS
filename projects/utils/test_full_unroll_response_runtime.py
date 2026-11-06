"""Required real Torch: full30 native/scalar replay and clean terminal derivatives."""
import types,unittest
from unittest.mock import patch
import torch
from modules.zaps_algorithm import ZAPS
from modules.wavelet import OrthogonalDWT2D
from utils import full_unroll_response as r,overall_direction_audit as old
from utils import response_target_runtime as rt,response_target_lm as lm
from utils.test_overall_direction_audit_runtime import model
from utils.test_mask_stage_response_runtime import Mask
from utils.shrink_multitask import configuration
from utils.shrink_multitask_runtime import replay,informative
from utils.paper_clock_runtime import counters,snapshot

class Linear:
 def H(self,x):return .7*x+.3*x.roll(1,-1)

class Tests(unittest.TestCase):
 def test_all30_exact_native_and_original_scalar_replay_rng_and_cost(self):
  torch.manual_seed(91);n=8;m=model();mask=(torch.rand(1,1,n,n)>.7).float();op=Mask(mask);y=torch.rand(1,3,n,n)*mask
  z=ZAPS(m,op,**configuration('inpainting',n,1));entry=snapshot(z,torch.randn_like(y));case=dict(task='inpainting',sigma=.05)
  for arm,reference_arm in (('BASE','NULL'),('LEGACY_SCALAR','SHRINK')):
   expected,_=replay(m,op,y,case,entry,reference_arm,size=n,level=1);rng=old.rng_state(m.device)
   z=old.make_z(m,op,entry,'inpainting',n,1);old.restore_rng(entry,m.device);x=entry['initial'].clone();details=[]
   with counters(m.model) as count:
    for k in range(30):
     x,actual,detail,_,_=r.step(m,z,op,y,x,k,arm,count);details.append(detail)
   torch.testing.assert_close(x,expected,rtol=1e-6,atol=1e-6)
   self.assertTrue(rt.rng_equal(actual,rng));self.assertEqual(count['forwards'],30)
   self.assertEqual(count['input_vjp_attempts'],0);self.assertEqual(count['optimizer_backward_attempts'],0)
   self.assertEqual(count['calls'],entry['tau'].tolist()[::-1]);self.assertEqual(details[-1]['next_t'],-1)
 def test_terminal_identity_jvp_vjp_adjoint_and_no_network(self):
  torch.manual_seed(23);m=model();op=Linear();x=.1*torch.randn(1,3,8,8);q=.1*torch.randn_like(x);dwt=OrthogonalDWT2D('db4',1)
  for scalar in (True,False):
   count={};g=rt.Geometry(q,dwt,scalar,count);p=torch.ones(1)*.1 if scalar else .1*dwt.analysis(q)
   direction=torch.ones_like(p)*.01;target=op.H(x).reshape(-1);w=torch.randn_like(target);provider=r.Provider(m,op,x,-1,g,target,count)
   with patch.object(m,'_predict_eps',side_effect=RuntimeError('No network at -1')):
    value=provider.value(p);jv=provider.jvp(p,direction);prediction,gradient=provider.linearize(p);adj=provider.vjp(p,w)
   torch.testing.assert_close(value,op.H(x+g.image(p)).reshape(-1))
   torch.testing.assert_close(jv,op.H(g.image(direction)).reshape(-1))
   self.assertAlmostEqual(lm.dot(jv,w),lm.dot(direction,adj),delta=1e-5)
   torch.testing.assert_close(prediction,value)
   self.assertEqual(count.get('input_vjp_attempts',0),0);self.assertEqual(count.get('input_jvp_attempts',0),0)
   self.assertEqual(count['terminal_operator_vjp_completed'],2)
 def test_terminal_real_scalar_joint_and_cross_actions_do_not_denoise(self):
  torch.manual_seed(42);m=model();op=Linear();x=torch.zeros(1,3,8,8);q=torch.ones_like(x)*.01;y=op.H(x+.2)
  z=types.SimpleNamespace(dwt=OrthogonalDWT2D('db4',1));results={}
  with patch.object(m,'_predict_eps',side_effect=RuntimeError('No network at -1')):
   for arm in r.ARMS[2:]:
    delta,detail,_=r.action(m,op,y,z,x,q,x,0,-1,arm,{})
    self.assertTrue(torch.isfinite(delta).all());results[arm]=(delta,detail)
    self.assertGreater(detail['scalar_norm'],0.)
   for arm in r.ARMS[2:]:
    delta,detail,_=r.action(m,op,y,z,x,q*0,x,0,-1,arm,{})
    self.assertEqual(detail['status'],'ZERO_NATIVE_FEASIBLE_SET');self.assertEqual(float(delta.norm()),0.)
  s=results['REAL_SCALAR'][0];d=results['REAL_DIAG'][0]
  for arm in r.ARMS[4:]:torch.testing.assert_close(results[arm][0],r.cross.correction(q,s,d,arm),rtol=1e-5,atol=1e-6)

if __name__=='__main__':unittest.main()
