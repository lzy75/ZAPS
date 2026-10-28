"""Server-required Torch checks; CPU toy is not an actual FFHQ experiment."""
import unittest
from unittest.mock import patch
import torch
from utils import next_response_bayes_runtime as runtime
from utils import next_response_bayes as rule
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters,snapshot_equal
from utils.test_paper_clock_runtime import TinyDiffusion,Linear


class RuntimeTests(unittest.TestCase):
    def test_default_flags_and_null_loop_match_core(self):
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True);y=torch.full((1,3,16,16),.15)
        outputs=[];entries=[]
        with frozen_network(dm.model):
            for arm in ('BASE','NULL'):
                with counters(dm.model) as count:
                    out,data=runtime.execute(dm,Linear(),y,1000,arm,.05,size=16,level=1)
                self.assertEqual(count['forwards'],300);self.assertEqual(count['optimizer_backward_completed'],10)
                outputs.append(out);entries.append(data['entry'])
        self.assertTrue(dm.model.weight.requires_grad)
        self.assertTrue(snapshot_equal(*entries)['passed'])
        torch.testing.assert_close(*outputs,rtol=1e-6,atol=1e-6)

    def test_real_predictions_reused_and_paid_queries_counted(self):
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True);y=torch.full((1,3,16,16),.15)
        with frozen_network(dm.model):
            for arm in ('REAL_ISO','REAL_STATE'):
                with patch.object(torch.autograd,'grad',side_effect=AssertionError('No input VJP')):
                    with counters(dm.model) as count:
                        out,data=runtime.execute(dm,Linear(),y,1000,arm,.05,size=16,level=1)
                self.assertEqual(count['optimizer_backward_completed'],10)
                self.assertEqual(count['forwards'],270+data['native_last_calls']+data['probe_calls'])
                self.assertLessEqual(count['forwards'],rule.MAX_FULL[arm])
                self.assertTrue(any(r['cached_prediction_reused'] for r in data['trace'][1:]))
                self.assertTrue(any(r['resolved'] for r in data['trace']))
                self.assertTrue(all(r['selected_objective']<=min(r['objective0'],r['objective1'])+1e-8
                                    for r in data['trace'] if r['resolved']))
                if arm=='REAL_STATE':
                    measured=[r for r in data['trace'] if r['uncertainty_source']=='DIRECTIONAL_STATE' and r['next_t']>=0]
                    self.assertTrue(measured)
                    for row in measured:
                        alpha=float(dm.alphas_cumprod[row['next_t']])
                        expected=(1-(1-alpha)**.5*float(dm.model.weight))/alpha**.5
                        self.assertAlmostEqual(row['kappa'],expected,delta=.01*(1+abs(expected)))
                self.assertTrue(torch.isfinite(out).all());self.assertIsNone(dm.model.weight.grad)


if __name__=='__main__':unittest.main()
