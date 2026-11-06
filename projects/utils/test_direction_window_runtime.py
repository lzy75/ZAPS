"""Required actual Torch geometry and two refreshed lookahead VJP/window gates."""
import unittest
from unittest.mock import patch
import torch
from utils import direction_window as r,overall_direction_audit as old
from utils.test_overall_direction_audit_runtime import model
from utils.test_mask_stage_response_runtime import Mask
from utils.shrink_multitask import configuration
from modules.zaps_algorithm import ZAPS
from modules.wavelet import OrthogonalDWT2D
from utils.paper_clock_runtime import snapshot,counters


class Tests(unittest.TestCase):
 def fixture(self):
  torch.manual_seed(63);n=32;m=model();mask=(torch.rand(1,1,n,n)>.7).float();op=Mask(mask)
  y=torch.rand(1,3,n,n)*mask;z=ZAPS(m,op,**configuration('inpainting',n,3));entry=snapshot(z,torch.randn_like(y))
  state=old.prefix(m,op,y,entry,'inpainting',n,3)
  physical,clean=old.physical_reference(m,op,y,state,dict(input_vjp_attempts=0))
  units,detail=old.directions(state['q'],physical,z.dwt)
  return m,op,y,entry,z,dict(state=state,physical=physical,reference_clean=clean,units=units,detail=detail)
 def test_geometry_reuses_cached_vectors_without_model_or_grad(self):
  m,op,y,entry,z,bundle=self.fixture()
  before=torch.get_rng_state().clone()
  with patch.object(m.model,'forward',side_effect=RuntimeError('No NN in geometry')),patch('torch.autograd.grad',side_effect=RuntimeError('No grad in geometry')):
   result=r.geometry(bundle,OrthogonalDWT2D('db4',3))
  self.assertEqual(result['UNet'],0);self.assertEqual(result['input_vjp'],0)
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
  self.assertGreaterEqual(result['cosine_coefficient_signed']+1e-5,result['cosine_band_signed'])
 def test_window_first_action_matches_parent_and_refresh_costs_are_exact(self):
  m,op,y,entry,z,bundle=self.fixture()
  for arm in r.ARMS:
   kind=arm.rsplit('_',1)[0]
   correction=bundle['units'][kind]*bundle['detail']['native_norm']
   _,expected_first,expected_rng=old.continue_run(m,z,y,bundle['state'],correction)
   with counters(m.model) as count:out,first,rng,trace=r.run(m,op,y,entry,bundle,'inpainting',arm,count,size=32,level=3)
   self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['input_vjp_completed'],count['optimizer_backward_attempts']),(16,2,2,0))
   self.assertEqual(count['calls'],r.expected_calls(entry['tau'].tolist()[::-1]))
   torch.testing.assert_close(first,expected_first,rtol=0,atol=0)
   self.assertTrue(torch.equal(rng['cpu_rng'],expected_rng['cpu_rng']))
   self.assertEqual(len(trace),3);self.assertEqual([row['reference_reused'] for row in trace],[True,False,False])
   self.assertTrue(torch.isfinite(out).all())

if __name__=='__main__':unittest.main()
