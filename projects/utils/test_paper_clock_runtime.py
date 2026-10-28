"""A64 REQUIRED Torch numeric checks. Missing torch is an error, never a skip."""
import unittest
from unittest.mock import patch
import torch
from utils import paper_clock_alignment as rules
from utils import paper_clock_runtime as runtime
from utils.current_score_runtime_r2 import input_vjp_parameters,field_from_mean,raw_mean
from modules.zaps_algorithm import ZAPS


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.05),requires_grad=False)
    def forward(self,x,t):return self.weight*x


class TinyDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__();self.device='cpu';self.num_steps=1000;self.model=TinyNetwork()
        self.register_buffer('alphas_cumprod',torch.linspace(.9999,.2,1000))
    def _predict_eps(self,x,t):return self.model(x,t)


class Linear(torch.nn.Module):
    def H(self,x):return x
    def transpose(self,x):return x


class Unit(torch.nn.Module):
    def __init__(self):super().__init__();self.base=Linear()
    def H(self,x):return self.base.H((x+1)*.5)
    def transpose(self,x):return .5*self.base.transpose(x)


def fallback(z,y,x,t,s,positions,sigma,**kwargs):
    _,raw=raw_mean(z.dm,x,t,graph=True)
    return field_from_mean(z,y,raw,t,positions),1.,dict(resolved=False,reason='TEST_FORCE_FALLBACK')


class NumericTests(unittest.TestCase):
    def execute(self,arm,sensed=None):
        model=TinyDiffusion();op=Unit();y=torch.full((1,3,16,16),.55)
        with input_vjp_parameters(model.model),runtime.counters(model.model) as counts:
            result,detail=runtime.execute(model,op,y,'super_resolution',arm,counts,
                                          size=16,level=1,sensed=sensed)
        runtime.validate_counts(counts,arm)
        self.assertIsNone(model.model.weight.grad)
        self.assertFalse(model.model.weight.requires_grad)
        self.assertEqual(float(model.model.weight),float(torch.tensor(.05)))
        return result,detail,counts

    def test_independent270_prefix_and_noise_match_without_learning_cache(self):
        baseline,one,c1=self.execute('PAPER_ZAPS')
        candidate,two,c2=self.execute('RESPONSE_300',fallback)
        self.assertTrue(runtime.snapshot_equal(one['snapshot'],two['snapshot'])['passed'])
        torch.testing.assert_close(candidate,baseline,rtol=1e-6,atol=1e-6)
        self.assertEqual((c1['forwards'],c2['forwards']),(300,300))
        self.assertEqual((c1['optimizer_backward_completed'],c2['optimizer_backward_completed']),(10,9))

    def test_independent30_forbids_optimize_and_loading_learned_tensor(self):
        with (patch.object(ZAPS,'optimize',side_effect=AssertionError('No learning allowed')),
              patch.object(torch,'load',side_effect=AssertionError('No learned cache allowed'))):
            result,detail,counts=self.execute('RESPONSE_30',fallback)
        from utils.no_learning_replay import noise_from_seed
        initial,noises,before=noise_from_seed(1000,'cpu',shape=(1,3,16,16))
        self.assertTrue(torch.equal(initial,detail['snapshot']['initial']))
        self.assertTrue(torch.equal(before['cpu'],detail['snapshot']['cpu_rng']))
        self.assertEqual(counts['input_vjp_attempts'],0)
        self.assertEqual(counts['forwards'],30)

    def test_real_response_tracks_input_vjp_separately_from_forward_and_weights(self):
        result,detail,counts=self.execute('RESPONSE_30')
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(counts['forwards'],30)
        self.assertEqual(counts['input_vjp_attempts'],30)
        self.assertEqual(counts['optimizer_backward_attempts'],0)
        self.assertEqual(len(detail['trace']),30)


if __name__=='__main__':unittest.main()
