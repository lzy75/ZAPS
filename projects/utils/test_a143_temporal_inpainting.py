"""Verify temporal target memory, matched controls, and native hooks."""
from types import SimpleNamespace
import unittest

import torch

from utils import diag_a143_temporal_inpainting as trial
from utils import dps_missing_tasks_runtime, pigdm_noisy_r1_runtime
from utils.pigdm_noisy_runtime import Operator


class TemporalInpaintingTests(unittest.TestCase):
    def test_target_uses_per_pixel_causal_median(self):
        clean = torch.tensor([1., 3., 2.])
        earlier = [torch.tensor([0., 5., 2.]),torch.tensor([.5, 0., 3.])]
        torch.testing.assert_close(trial.temporal_target(clean,earlier,'TEMPORAL'),torch.tensor([-.5, 0., 0.]))
        torch.testing.assert_close(trial.temporal_target(clean,earlier,'ZERO_MISSING'),torch.zeros_like(clean))

    def test_temporal_target_changes_missing_direction_under_same_penalty(self):
        op = Operator('inpainting','cpu',size=8,seed=139)
        wave = SimpleNamespace(analysis=lambda x:x,synthesis=lambda x:x)
        q = torch.ones(1,3,8,8)
        q[...,0,0] = 0
        earlier = torch.zeros_like(q)
        clean = torch.full_like(q,-.1)
        residual = op.H(earlier)
        results = {}
        for arm in trial.ARMS:
            count,trace = dict(graph_builds=0,graph_actions=0),[]
            trial.native.initialize_counts(count)
            remember,refine = trial.controller(op,wave,arm,.05,count,trace)
            remember(earlier)
            remember(earlier)
            out = refine(q,residual,1.,dict(gamma=0.,weight=1.),0,clean)
            self.assertTrue(torch.equal(out[q==0],q[q==0]))
            results[arm] = (out,trace[0])
        zero,zero_trace = results['ZERO_MISSING']
        temporal,temporal_trace = results['TEMPORAL']
        torch.testing.assert_close(zero,torch.zeros_like(q))
        self.assertEqual(zero_trace['missing_target_energy'],0.)
        self.assertGreater(float(temporal[...,~op.mask].mean()),0.)
        self.assertLess(temporal_trace['objective_delta'],0.)

    def test_native_hooks_and_warm_positions(self):
        for host,runtime,warm in (('dps',dps_missing_tasks_runtime,(101,100)),
                                  ('pigdm',pigdm_noisy_r1_runtime,(110,100))):
            grid = trial.parent.GRIDS[host]
            active = tuple(t for t in grid if t<=99)
            self.assertEqual(trial.buffer_positions(grid,active),warm)
            fn = trial.instrument(runtime,host,active,warm,lambda *_:None,lambda *_:None)
            self.assertEqual(fn.__globals__['warm'],set(warm))
            self.assertEqual(fn.__globals__['positions'],set(active))
            self.assertEqual(fn.__kwdefaults__,runtime.run.__kwdefaults__)


if __name__=='__main__':
    unittest.main()
