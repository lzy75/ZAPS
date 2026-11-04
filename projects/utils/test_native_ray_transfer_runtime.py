"""REQUIRED server Torch/CUDA gates; absence is a failure, never a skip."""
import os
import types
import unittest
import numpy as np
import torch
from utils import native_ray_transfer_runtime as runtime
from utils import native_ray_transfer as rules


class Identity:
    def H(self,x):return x.reshape(x.shape[0],-1)
    def Ht(self,x):return x
    def H_pinv(self,x):return x
    def singulars(self):return torch.ones(3*8*8,device='cuda:0')
    def U(self,x):return x
    def Ut(self,x):return x
    def forward(self,x,**kwargs):return self.H(x)


class Toy(torch.nn.Module):
    def forward(self,x,t):return torch.cat((.05*torch.tanh(x),torch.zeros_like(x)),dim=1)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('A82-A84 runtime gates REQUIRE allocated CUDA; no skip')
        cls.experiment=os.environ.get('RAY_EXPERIMENT','A82')
        if cls.experiment not in rules.HOSTS:raise ValueError('Unknown runtime test host')

    def test_native_loop_and_forced_one_are_identical(self):
        experiment=self.experiment
        model=types.SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        case=dict(task='super_resolution',sigma=.05,seed=4321)
        y=torch.linspace(-.7,.7,3*8*8,device='cuda:0').reshape(1,-1)
        observed=[]
        for arm,native in (('BASE',True),('BASE',False),('ONES',False),('SHRINK',False)):
            with runtime.counters(model.model) as count:
                out,detail=runtime.run(experiment,model,runtime.Operator(Identity(),count),y,case,arm,count,
                    size=8,test_steps=4,native=native)
            observed.append((out,detail,count))
        native,base,ones,shrink=observed
        self.assertTrue(torch.equal(native[0],base[0]))
        self.assertTrue(torch.equal(native[0],ones[0]))
        for candidate in (base,ones,shrink):
            self.assertTrue(torch.equal(candidate[1]['initial'],native[1]['initial']))
            self.assertTrue(torch.equal(candidate[1]['final_rng']['cpu'],native[1]['final_rng']['cpu']))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(candidate[1]['final_rng']['cuda'],native[1]['final_rng']['cuda'])))
            self.assertEqual(candidate[2]['forwards'],4)
        self.assertEqual(base[2]['module_H_actions'],0)
        self.assertEqual(ones[2]['module_H_actions'],8)
        self.assertEqual(shrink[2]['module_H_actions'],8)
        self.assertEqual(shrink[2]['input_vjp_attempts'],4 if experiment=='A84' else 0)
        self.assertEqual(shrink[2]['operator_gradient_attempts'],4 if experiment=='A82' else 0)
        self.assertEqual(len(shrink[1]['trace']),4)

    def test_real_native_operators_reproduce_state_and_gradient(self):
        # Small real operators, no UNet weights. Gaussian reflection kernel requires size>30.
        for task in rules.TASKS:
            runtime.seed(123)
            first=runtime.make_operator(self.experiment,task,'cuda:0',size=64)
            runtime.seed(123)
            second=runtime.make_operator(self.experiment,task,'cuda:0',size=64)
            a,b=runtime.state(first),runtime.state(second)
            self.assertEqual(set(a),set(b))
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            x=torch.linspace(-1.,1.,3*64*64,device='cuda:0').reshape(1,3,64,64).requires_grad_()
            op=runtime.Operator(first);result=op.H(x)
            self.assertTrue(torch.isfinite(result).all())
            gradient=torch.autograd.grad(result.square().sum(),x)[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_dmps_cuda_numpy_index_only_changes_transport(self):
        data=np.linspace(.01,.99,1000,dtype=np.float64)
        bridged=runtime.numpy_cuda_index_bridge(data)
        indices=torch.tensor([0,35,999],device='cuda:0')
        self.assertTrue(np.array_equal(bridged[indices],data[[0,35,999]]))
        self.assertEqual(bridged.dtype,np.float64)


if __name__=='__main__':unittest.main()
