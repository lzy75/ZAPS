"""Mandatory Torch + official Euler/MC/operator gates, no skip without dependencies."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from utils import daps_multiplier_runtime as runtime
from utils.diag_learning_rate_ablation import set_seed


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.03),requires_grad=False)
    def forward(self,x,t):return torch.cat((self.weight*x,torch.zeros_like(x)),dim=1)


def toy():
    ref=runtime.reference()
    class Model(ref.models.DiffusionModel):
        def __init__(self):
            super().__init__();self.model=ref.pre.VPPrecond(model=Network(),learn_sigma=True);self.device='cpu'
        def tweedie(self,x,sigma):return self.model(x,torch.as_tensor(sigma).to(x.device))
        def get_in_shape(self):return (3,8,8)
    return Model()


class TorchTests(unittest.TestCase):
    def test_original_loop_gamma_one_noise_grid_and_measurement_gradients(self):
        ref=runtime.reference();model=toy();raw=model.model.model
        config=dict(annealing_scheduler_config=dict(name='edm',num_steps=3,sigma_max=1.,sigma_min=.2,timestep='poly-7'),
                    diffusion_scheduler_config=dict(name='edm',num_steps=2,sigma_min=.01,timestep='poly-7'),
                    mcmc_sampler_config=dict(num_steps=2,lr=1e-4,tau=.01,lr_min_ratio=.01,prior_solver='gaussian',mc_algo='langevin'))
        op=ref.operators.DownSampling(resolution=8,scale_factor=4,device='cpu',sigma=.05)
        y=torch.full((1,3,2,2),.1);case=dict(seed=123)
        set_seed(case['seed']);sampler=ref.DAPS(**copy.deepcopy(config));initial=sampler.get_start(1,model)
        with runtime.counters(raw) as count:oracle=sampler.sample(model,initial,op,y)
        oracle_rng=torch.get_rng_state().clone();self.assertEqual(count['forwards'],6)
        versions=[p._version for p in raw.parameters()];outputs={};details={}
        for arm in ('DAPS','ONES','SHRINK'):
            with runtime.counters(raw) as count:
                outputs[arm],details[arm]=runtime.run(model,op,y,case,arm,count,sampler_config=config)
            self.assertEqual((count['forwards'],count['operator_gradient_attempts'],count['operator_gradients_completed']),(6,6,6))
            self.assertEqual((count['input_vjp_attempts'],count['optimizer_backward_attempts']),(0,0))
            self.assertEqual(count['module_H_actions'],0 if arm=='DAPS' else 6)
            self.assertEqual((count['annealing_steps'],count['mcmc_updates']),(3,6))
            self.assertTrue(torch.equal(details[arm]['final_cpu_rng'],oracle_rng))
            torch.testing.assert_close(details[arm]['initial'],initial,rtol=0,atol=0)
            self.assertEqual(len(details[arm]['trace']),3)
            self.assertTrue(torch.isfinite(outputs[arm]).all())
        torch.testing.assert_close(outputs['DAPS'],oracle,rtol=0,atol=0)
        torch.testing.assert_close(outputs['ONES'],oracle,rtol=0,atol=0)
        self.assertTrue(all(p._version==version and p.grad is None and not p.requires_grad for p,version in zip(raw.parameters(),versions)))

    def test_original_motion_state_reproducible_and_input_gradient_finite(self):
        # Real 61x61 original constructor and reflection operator, using 64px to keep the gate small.
        ref=runtime.reference();set_seed(30226);op=ref.operators.MotionBlur(kernel_size=61,intensity=.5,device='cpu',sigma=.05)
        first=runtime.operator_state(op);set_seed(30226);second=ref.operators.MotionBlur(kernel_size=61,intensity=.5,device='cpu',sigma=.05)
        self.assertEqual(set(first),set(runtime.operator_state(second)))
        self.assertTrue(all(torch.equal(v,runtime.operator_state(second)[k]) for k,v in first.items()))
        x=torch.randn(1,3,64,64);y=op(x).detach();g,loss=op.gradient(x,y,return_loss=True)
        self.assertTrue(torch.isfinite(g).all());self.assertEqual(float(loss),0.)

    def test_strict_weight_reload_rejects_upstream_random_fallback(self):
        model=toy()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.pt';torch.save({},path)
            with patch.object(runtime.reference().models,'DDPM',return_value=model):
                with self.assertRaises(RuntimeError):runtime.load_model(path,'cpu')


if __name__=='__main__':unittest.main()
