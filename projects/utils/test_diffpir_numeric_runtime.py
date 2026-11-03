"""Mandatory Torch gates for the separately named equivalent solver arms."""
import types
import unittest
from unittest.mock import patch
import torch
from utils import diffpir_numeric_runtime as runtime
from utils import diffpir_numeric_transfer as rules
from utils.test_diffpir_multiplier_runtime import Network
from utils.paper_clock_runtime import counters
from utils.a67_model_freeze import frozen_network


class TorchTests(unittest.TestCase):
    def test_small_rho_stable32_against_official64(self):
        torch.manual_seed(31)
        k=torch.tensor([[[[.05,.1,.05],[.1,.4,.1],[.05,.1,.05]]]])
        y=torch.randn(1,3,16,16);m=torch.randn_like(y)
        _,sr,_,_=runtime.reference()
        for scalar in (7e-7,.0015,20.):
            rho=torch.tensor(scalar).reshape(1,1,1,1)
            actual=runtime.stable_solution(m,*sr.pre_calculate(y,k,1),rho,1)
            expected=sr.data_solution(m.double(),*sr.pre_calculate(y.double(),k.double(),1),rho.double(),1)
            self.assertLess(float((actual.double()-expected).norm()/expected.norm()),1e-3)
            op=runtime.CircularBlur(k.double(),y.shape[-2:])
            rhs=op.transpose(y.double())+rho.double()*m.double()
            error=op.transpose(op.H(actual.double()))+rho.double()*actual.double()-rhs
            self.assertLess(float(error.norm()/rhs.norm()),1e-5)
        with self.assertRaises(ValueError):runtime.stable_solution(m,None,None,None,None,rho,2)

    def test_stable_gamma_one_and_native_binding_rng_cost(self):
        model=types.SimpleNamespace(model=Network(),device='cpu')
        y=torch.full((1,3,8,8),.5)
        k=torch.tensor([[[[0.,.1,0.],[.2,.4,.2],[0.,.1,0.]]]])
        case=rules.cases(dict(ffhq='/ffhq',imagenet='/imagenet'))[0]
        _,sr,_,_=runtime.reference();original=sr.data_solution
        outputs={};details={}
        with frozen_network(model.model),patch.object(torch.autograd,'grad',side_effect=AssertionError('No derivatives')):
            for arm in ('DIFFPIR','STABLE_DIFFPIR','STABLE_ONES','STABLE_SHRINK'):
                with counters(model.model) as count:
                    outputs[arm],details[arm]=runtime.run(model,y,k,case,arm,count,steps=8)
                self.assertIs(sr.data_solution,original)
                self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(8,0,0))
                self.assertEqual(count['native_data_solve_attempts'],7)
                self.assertTrue(torch.isfinite(outputs[arm]).all())
                self.assertTrue(torch.equal(details[arm]['final_cpu_rng'],details['DIFFPIR']['final_cpu_rng']))
                torch.testing.assert_close(details[arm]['initial'],details['DIFFPIR']['initial'],rtol=0,atol=0)
        torch.testing.assert_close(outputs['STABLE_DIFFPIR'],outputs['STABLE_ONES'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
