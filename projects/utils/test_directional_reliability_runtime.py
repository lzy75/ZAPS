"""Required real Torch tests; no skipping missing Torch or original operators."""
import types,unittest
from unittest.mock import patch
import torch
from utils import directional_reliability as rt
from utils import noise_signal_shrink as global_rule
from utils.test_mask_stage_response_runtime import Toy,Mask
from utils.shrink_multitask import configuration
from modules.zaps_algorithm import ZAPS
from modules.diffusion_model import _precompute_diffusion_params
from modules.wavelet import OrthogonalDWT2D
from utils.paper_clock_runtime import counters,snapshot

class Tests(unittest.TestCase):
 def test_noise_bank_preserves_rng_and_mask_total_variance(self):
  torch.manual_seed(63);n=32;mask=(torch.rand(1,1,n,n)>.7).float()
  op=Mask(mask);dwt=OrthogonalDWT2D('db4',3);before=torch.get_rng_state().clone()
  bank,record=rt.calibrate(op,dwt,(1,3,n,n),'inpainting_test','cpu')
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
  self.assertEqual(record['native_mapping_actions'],16)
  expected=3*float(mask.sum());self.assertAlmostEqual(float(bank['unit_variance'].sum())/expected,1.,places=5)
  self.assertTrue((bank['unit_variance']>=0).all())
 def test_identity_zero_and_zeta_homogeneity_without_projection(self):
  torch.manual_seed(8);q=torch.randn(1,3,32,32);dwt=OrthogonalDWT2D('db4',3)
  bank={k:torch.ones_like(q) for k in ('unit_variance','first_half','second_half')}
  factor=torch.full((3,32,32),.3)
  one,row=rt.shrink(q,factor,dwt,bank,0.)
  torch.testing.assert_close(one,q,rtol=0,atol=0)
  zero,_=rt.shrink(q,factor,dwt,bank,100.)
  self.assertEqual(float(zero.abs().max()),0.)
  a,ra=rt.shrink(q,factor,dwt,bank,.1);b,rb=rt.shrink(3*q,3*factor,dwt,bank,.1)
  torch.testing.assert_close(b,3*a,rtol=1e-5,atol=2e-6)
  torch.testing.assert_close(torch.tensor(ra['weights']),torch.tensor(rb['weights']),rtol=1e-5,atol=1e-7)
 def test_native_and_scalar_controls_and_directional_rng_nfe_no_global_rule(self):
  torch.manual_seed(63);n=32;mask=(torch.rand(1,1,n,n)>.7).float();op=Mask(mask)
  y=torch.rand(1,3,n,n)*mask
  model=types.SimpleNamespace(device='cpu',num_steps=1000,model=Toy(),alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod'])
  model._predict_eps=lambda x,t:model.model(x,t)[:,:3]
  c=dict(sigma=.1,task='inpainting');z=ZAPS(model,op,**configuration('inpainting',n,3))
  entry=snapshot(z,torch.randn(1,3,n,n));bank,_=rt.calibrate(op,z.dwt,y.shape,'inpainting_test','cpu')
  with torch.no_grad():
   qualification=rt.qualify(op,z.dwt,y.shape,'inpainting_test',entry,model.alphas_cumprod,'cpu')
   self.assertTrue(qualification['passed']);details={}
   for arm in rt.ARMS:
    actions={}
    with counters(model.model) as count:
     if arm=='DIRECTIONAL':
      with patch.object(global_rule,'select',side_effect=RuntimeError('Global multiplier is forbidden in new rule')):
       out,detail=rt.run(model,Mask(mask),y,entry,c,arm,actions,size=n,level=3,bank=bank)
     else:out,detail=rt.run(model,Mask(mask),y,entry,c,arm,actions,size=n,level=3)
    self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(30,0,0))
    self.assertTrue(torch.isfinite(out).all());details[arm]=detail
    if arm=='DIRECTIONAL':
     self.assertEqual(actions['reliability_steps'],30)
     for row in detail['trace']:
      self.assertFalse(row['global_multiplier_used'])
      self.assertTrue(all(0<=w<=1 for w in row['weights']))
      self.assertLessEqual(row['correction_norm'],row['native_correction_norm']*(1+3e-5))
   for arm in rt.ARMS:self.assertTrue(torch.equal(details[arm]['final_cpu_rng'],details['BASE_CHECK']['final_cpu_rng']))

if __name__=='__main__':unittest.main()
