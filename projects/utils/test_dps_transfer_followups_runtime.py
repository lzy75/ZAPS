"""Mandatory Torch gates; absence of Torch fails instead of skipping."""
from pathlib import Path
import types,unittest
import torch
from utils import dps_multiplier_transfer_runtime as original
from utils import dps_frozen_profile_runtime as frozen
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.paper_clock_runtime import counters
from utils.test_dps_multiplier_transfer_runtime import Network,Operator
class TorchTests(unittest.TestCase):
 def test_frozen_replay_and_gamma1_native_equivalence(self):
  root=Path(__file__).resolve().parents[2]/'DPS'
  config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,rescale_timesteps=False,timestep_respacing='8')
  net=Network();model=types.SimpleNamespace(model=net,device='cpu');y=torch.full((1,3,8,8),.1)
  case=dict(seed=1226,sigma=.05);outputs={};details={};version=net.weight._version
  with input_vjp_parameters(net):
   for arm in ('DPS','SHRINK','REPLAY','ONES'):
    with counters(net) as count:
     if arm in ('DPS','SHRINK'):out,detail=original.run(model,Operator(),y,case,arm,config,root,count,size=8)
     else:
      curve=[r['gamma'] for r in details['SHRINK']['trace']] if arm=='REPLAY' else [1.]*8
      out,detail=frozen.run_profile(model,Operator(),y,case,curve,config,root,count,size=8)
    outputs[arm]=out;details[arm]=detail
    self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(8,8,0))
    self.assertEqual(count['calls'],detail['grid']);self.assertTrue(torch.isfinite(out).all());self.assertIsNone(net.weight.grad)
   torch.testing.assert_close(outputs['DPS'],outputs['ONES'],rtol=1e-6,atol=1e-6)
   torch.testing.assert_close(outputs['SHRINK'],outputs['REPLAY'],rtol=0,atol=0)
   for detail in details.values():
    torch.testing.assert_close(details['DPS']['initial'],detail['initial'],rtol=0,atol=0)
    self.assertTrue(torch.equal(details['DPS']['cpu_rng'],detail['cpu_rng']))
   self.assertEqual(net.weight._version,version)
  self.assertFalse(net.weight.requires_grad)
 def test_imagenet_config_is_different_native_variance_compatible(self):
  from modules.diffusion_model import ImageNetDiffusionModel,FFHQDiffusionModel
  cfg=ImageNetDiffusionModel.MODEL_CONFIG
  self.assertEqual((cfg['num_channels'],cfg['num_res_blocks'],cfg['attention_resolutions']),(256,2,'32,16,8'))
  self.assertTrue(cfg['learn_sigma']);self.assertFalse(cfg['use_fp16']);self.assertNotEqual(cfg,FFHQDiffusionModel.MODEL_CONFIG)
if __name__=='__main__':unittest.main()
