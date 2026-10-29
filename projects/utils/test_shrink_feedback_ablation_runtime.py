"""Required real Torch gate; no dependency skip, no pretrained GPU claim."""
import unittest
import torch
from utils import shrink_feedback_ablation as rules
from utils import shrink_feedback_ablation_runtime as runtime
from utils import shrink_multitask_runtime as parent
from utils import noise_signal_shrink as original
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters
from utils.test_paper_clock_runtime import TinyDiffusion, Linear


class RuntimeTests(unittest.TestCase):
    def test_cached_replay_preserves_prefix_and_uses_exactly30_forwards(self):
        dm = TinyDiffusion(); op = Linear(); y = torch.full((1, 3, 16, 16), .15)
        c = dict(task='super_resolution', seed=1226, sigma=.05, mask_seed=2001226)
        with frozen_network(dm.model):
            _, detail = parent.execute(dm, op, y, c, 'BASE', size=16, level=1)
            entry = detail['entry']
            before = {k: v.clone() for k, v in entry.items() if isinstance(v, torch.Tensor)}
            expected, trace = parent.replay(dm, op, y, c, entry, 'SHRINK', size=16, level=1)
            cfg = dict(fixed_scalar=1., fixed_profile=[v['gamma'] for v in trace], mismatch=[1.]*30)
            select_before = original.select
            for arm in rules.EXECUTED:
                with counters(dm.model) as count:
                    out, rows = runtime.replay(dm, op, y, c, entry, arm, cfg, size=16, level=1)
                self.assertEqual((count['forwards'], count['optimizer_backward_attempts'], count['input_vjp_attempts']), (30, 0, 0))
                self.assertEqual(len(rows), 30)
                self.assertTrue(torch.isfinite(out).all())
                if arm in (rules.CHECK, 'FIXED_PROFILE'):
                    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
                self.assertIs(original.select, select_before)
            for k, v in before.items():
                torch.testing.assert_close(entry[k], v, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
