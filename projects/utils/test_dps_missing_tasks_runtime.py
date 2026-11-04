"""Required original DPS / real operators Torch gates. Missing Torch is an error."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from utils import dps_missing_tasks_runtime as runtime
from utils import shrink_multitask_runtime as physical
from utils import noise_signal_shrink as policy
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.paper_clock_runtime import counters
from utils.test_dps_multiplier_transfer_runtime import Network
from utils.diag_learning_rate_ablation import set_seed

CONFIG=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
 model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
 rescale_timesteps=False,timestep_respacing='8')
ROOT=Path(__file__).resolve().parents[2]/'DPS'

class TorchTests(unittest.TestCase):
 def test_real_operators_adjoint_mask_seed_and_original_input_gradient(self):
  for task in ('gaussian_deblur','inpainting'):
   c=dict(task=task,mask_seed=3000);op=physical.make_operator(c,'cpu',size=8)
   set_seed(1);x=torch.randn(1,3,8,8,requires_grad=True);v=torch.randn_like(x)
   y=op.H(x);grad=torch.autograd.grad((y*v).sum(),x)[0]
   torch.testing.assert_close(grad,op.transpose(v),rtol=2e-5,atol=2e-5)
   self.assertTrue(torch.isfinite(grad).all())
   if task=='inpainting':
    second=physical.make_operator(c,'cpu',size=8)
    self.assertTrue(torch.equal(op._mask_cache,second._mask_cache))
    self.assertEqual(physical.informative(y,op).numel(),3*int(op._mask_cache.sum()))
 def test_native_loop_forced_one_rng_counts_and_active_noise_energy(self):
  for task in ('gaussian_deblur','inpainting'):
   net=Network();model=SimpleNamespace(model=net,device='cpu')
   c=dict(task=task,mask_seed=3000,seed=1226,sigma=.05)
   op=physical.make_operator(c,'cpu',size=8)
   set_seed(44);y=op.H(torch.full((1,3,8,8),.1))+.05*torch.randn(1,3,8,8)
   outs={};details={};version=net.weight._version
   with input_vjp_parameters(net):
    # Independent unwrapped author loop/condition, including native task scale.
    dps,conditions,_=runtime.reference(str(ROOT.resolve()));sampler=dps.create_sampler(**CONFIG)
    cond=conditions.get_conditioning_method('ps',operator=runtime.PureOperator(op),
     noiser=SimpleNamespace(__name__='gaussian'),scale=runtime.rules.SCALES[task])
    set_seed(c['seed']);initial=torch.randn_like(y)
    with torch.enable_grad():native=sampler.p_sample_loop(model=net,x_start=initial,measurement=y,
     measurement_cond_fn=cond.conditioning,record=False,save_root='').detach()
    for arm in ('DPS','NULL','SHRINK'):
     with counters(net) as count:
      outs[arm],details[arm]=runtime.run(model,op,y,c,arm,CONFIG,ROOT,count,size=8)
     self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(8,8,0))
     self.assertEqual(count['module_H_actions'],24 if arm=='SHRINK' else 0)
     self.assertEqual(count['calls'],details[arm]['grid'])
     self.assertTrue(torch.isfinite(outs[arm]).all());self.assertIsNone(net.weight.grad)
    torch.testing.assert_close(native,outs['DPS'],rtol=0,atol=0)
    torch.testing.assert_close(outs['DPS'],outs['NULL'],rtol=1e-6,atol=1e-6)
    self.assertEqual(details['DPS']['pairing'],details['SHRINK']['pairing'])
    self.assertEqual(net.weight._version,version)
    active=physical.informative(y,op).numel()
    self.assertTrue(all(r['active_measurements']==active for r in details['SHRINK']['trace']))
   self.assertFalse(net.weight.requires_grad)
 def test_mask_policy_excludes_missing_noise_but_native_dps_does_not_change(self):
  c=dict(task='inpainting',mask_seed=3000,seed=1226,sigma=.05)
  op=physical.make_operator(c,'cpu',size=8)
  net=Network();model=SimpleNamespace(model=net,device='cpu')
  y=torch.full((1,3,8,8),.1);captured=[];select=policy.select
  def spy(dot,energy,response,count,sigma):
   captured.append((energy,count));return select(dot,energy,response,count,sigma)
  with input_vjp_parameters(net),patch.object(policy,'select',side_effect=spy),counters(net) as count:
   out,detail=runtime.run(model,op,y,c,'SHRINK',CONFIG,ROOT,count,size=8)
  active=int(op._mask_cache.sum())*3
  self.assertTrue(all(n==active for _,n in captured))
  self.assertLess(active,y.numel())
  self.assertTrue(torch.isfinite(out).all())

if __name__=='__main__':unittest.main()
