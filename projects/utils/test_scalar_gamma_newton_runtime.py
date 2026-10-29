"""Required Torch scalar AD, native checkpoint and cost checks; no skip."""
import unittest
import torch
from utils import scalar_gamma_newton_runtime as runtime
from utils import scalar_gamma_newton as rule
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters,snapshot_equal
from utils.test_paper_clock_runtime import TinyDiffusion,Linear


class RuntimeTests(unittest.TestCase):
    def test_frozen_default_flags_null_and_autograd_gradient(self):
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True)
        y=torch.full((1,3,16,16),.15);entries=[];outputs=[]
        with frozen_network(dm.model):
            for arm in ('BASE','NULL','MODEL_NEWTON','AUTO_NEWTON'):
                with counters(dm.model) as count:
                    out,data=runtime.execute(dm,Linear(),y,1000,arm,.05,count,size=16,level=1)
                entries.append(data['entry']);outputs.append(out)
                self.assertTrue(torch.isfinite(out).all());self.assertEqual(count['optimizer_backward_completed'],10)
                if arm in rule.MAX_FULL:self.assertLessEqual(count['forwards'],rule.MAX_FULL[arm])
                if arm=='AUTO_NEWTON':
                    self.assertGreater(count['input_vjp_attempts'],0)
                    self.assertLessEqual(count['input_vjp_attempts'],58)
                    self.assertTrue(all(r['value_points']<=3 and r['gradient_points']<=2 for r in data['trace']))
                else:self.assertEqual(count['input_vjp_attempts'],0)
                self.assertIsNone(dm.model.weight.grad);self.assertFalse(dm.model.weight.requires_grad)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-6,atol=1e-6)
        self.assertTrue(all(snapshot_equal(entries[0],e)['passed'] for e in entries[1:]))
        self.assertTrue(dm.model.weight.requires_grad)

    def test_scalar_ad_decreases_actual_linear_objective(self):
        from modules.zaps_algorithm import ZAPS
        from utils.historical_ffhq_gain import config
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True)
        cfg=config();cfg.update(img_size=16,level=1)
        z=ZAPS(dm,Linear(),**cfg);z.sigma=.05;z.cost={'input_vjp_attempts':0}
        uncond=torch.full((1,3,16,16),.1);q=torch.full_like(uncond,.01);y=torch.full_like(q,.2)
        with frozen_network(dm.model),torch.no_grad():
            gamma,_,row=runtime.decide(z,y,uncond,q,999,916,29,'AUTO_NEWTON')
        self.assertTrue(torch.isfinite(torch.tensor(gamma)))
        self.assertLessEqual(row['objective'],row['reference_objective']+1e-6)
        self.assertGreater(z.cost['input_vjp_attempts'],0)


if __name__=='__main__':unittest.main()
