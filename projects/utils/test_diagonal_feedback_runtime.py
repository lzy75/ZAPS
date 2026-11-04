"""Mandatory Torch gate; absence of Torch fails, no pretrained model result."""
import types,unittest
import torch
from utils import diagonal_feedback as rt
from utils import shrink_multitask_runtime as old
from utils.shrink_multitask import configuration
from utils.test_mask_stage_response_runtime import Toy,Mask
from modules.zaps_algorithm import ZAPS
from modules.diffusion_model import _precompute_diffusion_params
from utils.paper_clock_runtime import snapshot,counters

class Tests(unittest.TestCase):
 def test_real_parent_loops_rng_nfe_and_zero_additional_backwards(self):
  torch.manual_seed(63);n=8
  mask=(torch.rand(1,1,n,n)>.7).float();y=torch.rand(1,3,n,n)*mask
  model=types.SimpleNamespace(device='cpu',num_steps=1000,model=Toy(),alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod'])
  model._predict_eps=lambda x,t:model.model(x,t)[:,:3]
  c=dict(sigma=.1,task='inpainting');z=ZAPS(model,Mask(mask),**configuration('inpainting',n,1));entry=snapshot(z,torch.randn(1,3,n,n))
  results={}
  with torch.no_grad():
   for arm,parent_arm in (('BASE_CHECK','NULL'),('SCALAR_CHECK','SHRINK'),('DIAG4',None)):
    if parent_arm is not None:
     expected,_=old.replay(model,Mask(mask),y,c,entry,parent_arm,size=n,level=1)
    actions={'diagonal_H':0}
    with counters(model.model) as count:
     output,detail=rt.run(model,Mask(mask),y,entry,c,arm,actions,size=n,level=1)
    if parent_arm is not None:torch.testing.assert_close(output,expected,rtol=0,atol=0)
    self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(30,0,0))
    self.assertEqual(actions['diagonal_H'],120 if arm=='DIAG4' else 0)
    self.assertTrue(torch.isfinite(output).all());results[arm]=detail
    if arm=='DIAG4':
     self.assertEqual(len(detail['trace']),30)
     for row in detail['trace']:
      self.assertTrue(all(g>=0 for g in row['gamma_bands']))
      self.assertLessEqual(row['target_objective'],row['scalar_target_objective']+1e-8)
  for arm in rt.ARMS:self.assertTrue(torch.equal(results[arm]['final_cpu_rng'],results['BASE_CHECK']['final_cpu_rng']))
 def test_torch_band_completeness_and_energy(self):
  torch.manual_seed(5);x=torch.randn(1,3,16,16,dtype=torch.float64)
  parts=rt.haar_components(x)
  torch.testing.assert_close(sum(parts),x,rtol=0,atol=1e-15)
  self.assertAlmostEqual(sum(float(v.square().sum()) for v in parts),float(x.square().sum()),places=10)

if __name__=='__main__':unittest.main()
