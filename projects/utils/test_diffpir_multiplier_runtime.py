"""Mandatory real Torch/native upstream loop gates, never skipped without Torch."""
import types
import unittest
from unittest.mock import patch
import torch
from utils import diffpir_multiplier_runtime as runtime
from utils import diffpir_multiplier_transfer as rules
from utils.paper_clock_runtime import counters
from utils.a67_model_freeze import frozen_network


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.05),requires_grad=False)
    def forward(self,x,t):return torch.cat((self.weight*x,torch.zeros_like(x)),dim=1)


class TorchTests(unittest.TestCase):
    def test_native_fft_solution_and_adjoint_match_dense_normal_equation(self):
        torch.manual_seed(13);k=torch.tensor([[[[0.,.1,0.],[.2,.4,.2],[0.,.1,0.]]]],dtype=torch.float64)
        shape=(4,4);op=runtime.CircularBlur(k,shape);x=torch.randn(1,1,4,4,dtype=torch.float64);y=torch.randn_like(x)
        torch.testing.assert_close((op.H(x)*y).sum(),(x*op.transpose(y)).sum(),rtol=1e-12,atol=1e-12)
        _,sr,_,_=runtime.reference();rho=torch.tensor(.2,dtype=torch.float64).reshape(1,1,1,1)
        solved=sr.data_solution(x,*sr.pre_calculate(y,k,1),rho,1)
        basis=torch.eye(16,dtype=torch.float64).reshape(16,1,4,4);matrix=op.H(basis).reshape(16,16).T
        expected=torch.linalg.solve(matrix.T@matrix+.2*torch.eye(16,dtype=torch.float64),matrix.T@y.reshape(-1)+.2*x.reshape(-1))
        torch.testing.assert_close(solved.reshape(-1),expected,rtol=1e-10,atol=1e-10)

    def test_unmodified_native_loop_vs_gamma1_rng_count_and_no_derivatives(self):
        model=types.SimpleNamespace(model=Network(),device='cpu');y=torch.full((1,3,8,8),.5)
        k=torch.tensor([[[[0.,.1,0.],[.2,.4,.2],[0.,.1,0.]]]])
        case=rules.cases(dict(ffhq='/ffhq',imagenet='/imagenet'))[0]
        outputs={};details={};version=model.model.weight._version
        with frozen_network(model.model),patch.object(torch.autograd,'grad',side_effect=AssertionError('No input derivative allowed')):
            for arm in ('DIFFPIR','ONES','SHRINK'):
                with counters(model.model) as count:
                    outputs[arm],details[arm]=runtime.run(model,y,k,case,arm,count,steps=8)
                self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(8,0,0))
                self.assertEqual(count['calls'],details[arm]['grid']);self.assertEqual(len(details[arm]['trace']),7)
                self.assertEqual((count['native_data_solve_attempts'],count['native_data_solves_completed']),(7,7))
                self.assertTrue(torch.isfinite(outputs[arm]).all());self.assertIsNone(model.model.weight.grad)
            torch.testing.assert_close(outputs['DIFFPIR'],outputs['ONES'],rtol=0,atol=0)
            for detail in details.values():
                torch.testing.assert_close(detail['initial'],details['DIFFPIR']['initial'],rtol=0,atol=0)
                self.assertTrue(torch.equal(detail['final_cpu_rng'],details['DIFFPIR']['final_cpu_rng']))
            self.assertEqual(model.model.weight._version,version)
            # The shrink policy itself cannot consume any sampling randomness.
            self.assertEqual(sum(v['gamma']!=1 for v in details['SHRINK']['trace'])>0,True)


if __name__=='__main__':unittest.main()
