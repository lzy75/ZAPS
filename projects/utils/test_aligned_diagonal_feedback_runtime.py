"""Mandatory actual Torch/db4/RNG/energy gates; missing dependencies fail."""
import types,unittest
import torch
from utils import aligned_diagonal_feedback as rt
from utils import shrink_multitask_runtime as original
from utils.shrink_multitask import configuration
from utils.test_mask_stage_response_runtime import Toy,Mask
from modules.zaps_algorithm import ZAPS
from modules.diffusion_model import _precompute_diffusion_params
from utils.paper_clock_runtime import snapshot,counters

class Tests(unittest.TestCase):
 def test_original_db4_multiscale_coordinates_and_scalar_reduction(self):
  from modules.wavelet import OrthogonalDWT2D
  torch.manual_seed(11);x=torch.randn(1,3,32,32);dwt=OrthogonalDWT2D('db4',3)
  parts,labels=rt.db4_components(x,dwt)
  self.assertEqual(len(parts),10)
  grouped,_=rt.db4_components(x,dwt,1)
  self.assertEqual(len(grouped),4)
  torch.testing.assert_close(grouped[0],sum(parts[:7]),rtol=1e-5,atol=2e-6)
  torch.testing.assert_close(sum(parts),x,rtol=1e-5,atol=2e-6)
  torch.testing.assert_close(sum(.7*v for v in parts),.7*x,rtol=1e-5,atol=2e-6)
  flat=torch.stack([v.double().flatten() for v in parts]);g=flat@flat.T
  off=g-torch.diag(torch.diag(g));self.assertLess(float(off.norm()/g.norm()),1e-5)
 def test_same_checkpoint_free_model_grid_rng_and_bounded_matrix_updates(self):
  torch.manual_seed(63);n=32
  mask=(torch.rand(1,1,n,n)>.7).float();y=torch.rand(1,3,n,n)*mask
  model=types.SimpleNamespace(device='cpu',num_steps=1000,model=Toy(),alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod'])
  model._predict_eps=lambda x,t:model.model(x,t)[:,:3]
  c=dict(sigma=.1,task='inpainting');z=ZAPS(model,Mask(mask),**configuration('inpainting',n,3));entry=snapshot(z,torch.randn(1,3,n,n))
  results={}
  from utils import diagonal_feedback as old
  prior_apply=old.apply
  with torch.no_grad():
   for arm in rt.ARMS:
    actions={'diagonal_H':0}
    with counters(model.model) as count:
     output,detail=rt.run(model,Mask(mask),y,entry,c,arm,actions,size=n,level=3)
    self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(30,0,0))
    self.assertTrue(torch.isfinite(output).all());results[arm]=detail
    self.assertIs(old.apply,prior_apply)
    if arm in ('BASE_CHECK','SCALAR_CHECK'):
     reference,_=original.replay(model,Mask(mask),y,c,entry,'NULL' if arm=='BASE_CHECK' else 'SHRINK',size=n,level=3)
     torch.testing.assert_close(output,reference,rtol=0,atol=0)
    else:
     expected=10 if arm=='DB4_10_TRUST' else 4
     self.assertLessEqual(actions['diagonal_H'],30*expected)
     for row in detail['trace']:
      self.assertEqual(len(row['gamma_bands']),expected)
      self.assertLessEqual(row['norm_ratio'],1+1e-7)
      self.assertLessEqual(row['correction_norm'],row['scalar_correction_norm']*(1+3e-5))
      self.assertTrue(all(v>=0 for v in row['gamma_bands']))
  for arm in rt.ARMS:self.assertTrue(torch.equal(results[arm]['final_cpu_rng'],results['BASE_CHECK']['final_cpu_rng']))

if __name__=='__main__':unittest.main()
