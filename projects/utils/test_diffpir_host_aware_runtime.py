"""Mandatory real Torch gates; absence of Torch is an error, not a skipped test."""
import types
import unittest
from unittest.mock import patch
import torch
from utils import diffpir_host_aware_runtime as runtime
from utils import diffpir_multiplier_runtime as historical
from utils import diffpir_multiplier_transfer as rules
from utils.test_diffpir_multiplier_runtime import Network
from utils.paper_clock_runtime import counters
from utils.a67_model_freeze import frozen_network


class TorchTests(unittest.TestCase):
    def test_native_rng_count_gamma_one_and_binding_restoration(self):
        model=types.SimpleNamespace(model=Network(),device='cpu')
        y=torch.full((1,3,8,8),.5)
        k=torch.tensor([[[[0.,.1,0.],[.2,.4,.2],[0.,.1,0.]]]])
        c=rules.cases(dict(ffhq='/ffhq',imagenet='/imagenet'))[0]
        _,sr,_,_=runtime.reference();saved_solve=sr.data_solution;saved_policy=historical.policy
        outputs={};details={}
        with frozen_network(model.model),patch.object(torch.autograd,'grad',side_effect=AssertionError('No derivatives')):
            for arm in ('DIFFPIR_REFERENCE','HOST_ONES','HOST_AWARE'):
                with counters(model.model) as count:
                    outputs[arm],details[arm]=runtime.run(model,y,k,c,arm,count,steps=8)
                self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(8,0,0))
                self.assertEqual(count['native_data_solve_attempts'],7)
                self.assertEqual(count.get('module_q_norms',0),0 if arm=='DIFFPIR_REFERENCE' else 7)
                self.assertIs(sr.data_solution,saved_solve);self.assertIs(historical.policy,saved_policy)
                self.assertTrue(torch.equal(details[arm]['final_cpu_rng'],details['DIFFPIR_REFERENCE']['final_cpu_rng']))
                torch.testing.assert_close(details[arm]['initial'],details['DIFFPIR_REFERENCE']['initial'],rtol=0,atol=0)
                self.assertTrue(torch.isfinite(outputs[arm]).all())
        torch.testing.assert_close(outputs['DIFFPIR_REFERENCE'],outputs['HOST_ONES'],rtol=0,atol=0)
        for row in details['HOST_AWARE']['trace']:
            if row['resolved']:
                self.assertLess(row['stationarity_relative_error'],2e-4)
                self.assertLess(abs(row['gamma_minus_noise_weight']),5e-4)
        def broken(*args,**kwargs):raise RuntimeError('model failure fixture')
        with patch.object(model.model,'forward',side_effect=broken),counters(model.model) as count:
            with self.assertRaisesRegex(RuntimeError,'model failure'):runtime.run(model,y,k,c,'HOST_AWARE',count,steps=8)
        self.assertIs(sr.data_solution,saved_solve);self.assertIs(historical.policy,saved_policy)


if __name__=='__main__':unittest.main()
