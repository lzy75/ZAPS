"""Required real Torch/original DPS-loop gate; missing dependency is a failure."""
from pathlib import Path
import types
import unittest
import torch
from utils import dps_multiplier_transfer_runtime as runtime
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.paper_clock_runtime import counters


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.05),requires_grad=False)
    def forward(self,x,t):return torch.cat((self.weight*x,torch.zeros_like(x)),dim=1)


class Operator:
    def H(self,x):return .7*x


class RuntimeTests(unittest.TestCase):
    def test_original_conditioning_vs_gamma1_and_equal_rng_nfe_vjp(self):
        root=Path(__file__).resolve().parents[2]/'DPS'
        config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
            model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
            rescale_timesteps=False,timestep_respacing='8')
        net=Network();model=types.SimpleNamespace(model=net,device='cpu');y=torch.full((1,3,8,8),.1)
        outputs={};details={};version=net.weight._version
        with input_vjp_parameters(net):
            for arm in ('DPS','NULL','PROJECT','SHRINK'):
                with counters(net) as count:
                    outputs[arm],details[arm]=runtime.run(model,Operator(),y,dict(seed=1226,sigma=.05),
                        arm,config,root,count,size=8)
                self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(8,8,0))
                self.assertEqual(count['calls'],details[arm]['grid'])
                self.assertTrue(torch.isfinite(outputs[arm]).all())
                self.assertIsNone(net.weight.grad)
            torch.testing.assert_close(outputs['DPS'],outputs['NULL'],rtol=1e-6,atol=1e-6)
            for arm in details:
                torch.testing.assert_close(details['DPS']['initial'],details[arm]['initial'],rtol=0,atol=0)
                self.assertTrue(torch.equal(details['DPS']['cpu_rng'],details[arm]['cpu_rng']))
            self.assertEqual(net.weight._version,version)
        self.assertFalse(net.weight.requires_grad)


if __name__=='__main__':unittest.main()
