"""Mandatory Torch verification of reduced grids and last-unroll-only control."""
import unittest
import torch
from modules.zaps_algorithm import build_irregular_timesteps
from utils import step_count_shrink as rules
from utils import step_count_shrink_runtime as runtime
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters,snapshot_equal
from utils.test_paper_clock_runtime import TinyDiffusion,Linear


class RuntimeTests(unittest.TestCase):
    def test_every_grid_and_gamma1_equivalence_with_same_nine_unroll_prefix(self):
        dm=TinyDiffusion();y=torch.full((1,3,16,16),.15)
        with frozen_network(dm.model):
            for steps in rules.COUNTS:
                torch.testing.assert_close(torch.tensor(rules.grid(steps)),build_irregular_timesteps(schedule=rules.COUNTS[steps]))
                c=dict(task='super_resolution',seed=1226,sigma=.05,steps=steps,mask_seed=2001226)
                out={};details={}
                for arm in ('BASE','NULL','SHRINK'):
                    with counters(dm.model) as count:
                        out[arm],details[arm]=runtime.execute(dm,Linear(),y,c,arm,size=16,level=1)
                    self.assertEqual((count['forwards'],count['optimizer_backward_completed'],count['input_vjp_attempts']),(10*steps,10,0))
                    self.assertEqual(count['calls'],rules.grid(steps)[::-1]*10)
                    self.assertTrue(torch.isfinite(out[arm]).all())
                torch.testing.assert_close(out['BASE'],out['NULL'],rtol=1e-6,atol=1e-6)
                for k in ('zeta','D','losses'):
                    torch.testing.assert_close(torch.as_tensor(details['BASE'][k]),torch.as_tensor(details['NULL'][k]),rtol=1e-6,atol=1e-6)
                self.assertTrue(snapshot_equal(details['BASE']['entry'],details['SHRINK']['entry'])['passed'])
                self.assertEqual(len(details['SHRINK']['trace']),steps)
                self.assertTrue(all(p.grad is None for p in dm.model.parameters()))


if __name__=='__main__':unittest.main()
