"""REQUIRED Torch CPU equivalence with old A71 loop; no weights or CUDA needed."""
import types,unittest
import torch
from utils import mask_stage_response as rt
from utils import shrink_multitask_runtime as old
from utils.shrink_multitask import configuration
from modules.zaps_algorithm import ZAPS
from modules.diffusion_model import _precompute_diffusion_params
from utils.paper_clock_runtime import snapshot,counters
class Toy(torch.nn.Module):
    def forward(self,x,t):return torch.cat((.95*x,torch.zeros_like(x)),1)
class Mask(torch.nn.Module):
    def __init__(self,p):super().__init__();self.register_buffer('p',p);self._mask_cache=p
    def H(self,x):return self.p*x
    def transpose(self,x):return self.H(x)
class Tests(unittest.TestCase):
    def test_matches_parent_and_probes_do_not_change_rng(self):
        torch.manual_seed(63);n=8
        mask=(torch.rand(1,1,n,n)>.7).float();y=torch.rand(1,3,n,n)*mask
        model=types.SimpleNamespace(device='cpu',num_steps=1000,model=Toy(),alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod'])
        model._predict_eps=lambda x,t:model.model(x,t)[:,:3]
        c=dict(sigma=.1,task='inpainting');z=ZAPS(model,Mask(mask),**configuration('inpainting',n,1));entry=snapshot(z,torch.randn(1,3,n,n))
        results={}
        for arm,parent_arm in [('BASE_CHECK','NULL'),('SHRINK_CHECK','SHRINK')]:
            reference,_=old.replay(model,Mask(mask),y,c,entry,parent_arm,size=n,level=1)
            rng=torch.get_rng_state().clone()
            with counters(model.model) as count:
                output,detail=rt.run(model,mask,y,entry,c,arm,lambda x:{},size=n,level=1)
            self.assertTrue(torch.allclose(output,reference,rtol=1e-6,atol=1e-6))
            self.assertTrue(torch.equal(detail['final_cpu_rng'],rng))
            self.assertEqual(count['forwards'],33 if arm=='SHRINK_CHECK' else 30)
            results[arm]=detail
        self.assertEqual(len(results['SHRINK_CHECK']['probes']),3)
        for arm in ['BASE_EARLY','BASE_MIDDLE','BASE_LATE']:
            output,detail=rt.run(model,mask,y,entry,c,arm,lambda x:{},size=n,level=1)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(detail['final_cpu_rng'],results['BASE_CHECK']['final_cpu_rng']))
if __name__=='__main__':unittest.main()
