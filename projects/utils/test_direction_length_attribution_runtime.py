"""Required Torch: unchanged native continuation/RNG and exact risk decomposition."""
import unittest
from unittest.mock import patch
import torch
from modules.wavelet import OrthogonalDWT2D
from modules.zaps_algorithm import ZAPS
from utils import direction_length_attribution as r,overall_direction_audit as old
from utils.test_overall_direction_audit_runtime import model
from utils.test_mask_stage_response_runtime import Mask
from utils.shrink_multitask import configuration
from utils.paper_clock_runtime import snapshot,counters


class Tests(unittest.TestCase):
 def test_native_continuation_exact_with_snapshots_and_no_derivatives(self):
  torch.manual_seed(63);m=model();mask=(torch.rand(1,1,32,32)>.7).float();op=Mask(mask);y=torch.rand(1,3,32,32)*mask
  z=ZAPS(m,op,**configuration('inpainting',32,3));entry=snapshot(z,torch.randn_like(y));state=old.prefix(m,op,y,entry,'inpainting',32,3)
  expected,first,rng=old.continue_run(m,z,y,state,state['q'])
  with patch('torch.autograd.grad',side_effect=RuntimeError('No derivatives in A96')),counters(m.model) as count:
   output,stages,actual_rng=r.continue_run(m,z,y,state,state['q'])
  torch.testing.assert_close(output,expected,rtol=0,atol=0);torch.testing.assert_close(stages['16']['clean'],first,rtol=0,atol=0)
  self.assertTrue(torch.equal(rng['cpu_rng'],actual_rng['cpu_rng']));self.assertEqual(count['forwards'],14);self.assertEqual(count['input_vjp_attempts'],0)
  self.assertEqual([stages[str(k)]['t'] for k in r.STAGES],[309,119,0])
 def test_wavelet_error_energy_is_additive_and_separates_coarse_detail(self):
  dwt=OrthogonalDWT2D('db4',3);gt=torch.zeros(1,3,32,32);constant=gt+.1
  row=r.wavelet_risk(constant,gt,dwt)
  self.assertAlmostEqual(row['raw_mse'],sum(row['band_mse_contributions'].values()),delta=1e-7)
  self.assertGreater(row['coarse_LL3_mse'],.999*row['raw_mse'])
  checker=(torch.arange(32).reshape(1,1,32,1)+torch.arange(32).reshape(1,1,1,32))%2
  row=r.wavelet_risk((checker.float()*2-1).expand_as(gt)*.1,gt,dwt)
  self.assertGreater(row['detail_mse'],.999*row['raw_mse'])

if __name__=='__main__':unittest.main()
