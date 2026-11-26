"""Check exact SR projections and the role of the temporal nullspace target."""
from types import SimpleNamespace
import unittest

import torch

from utils import diag_a144_temporal_sr as trial
from utils.pigdm_noisy_runtime import Operator


class TemporalSRTests(unittest.TestCase):
    def test_sr_null_projection_is_orthogonal(self):
        op = Operator('super_resolution','cpu',size=8)
        count = {}
        trial.native.initialize_counts(count)
        view = trial.native.LinearView(op,count)
        x = torch.randn((1,3,8,8),generator=torch.Generator().manual_seed(144),dtype=torch.float64)
        xx = op.H(x)
        null = trial.null_projection(view,x)
        torch.testing.assert_close(16*op.H(op.Ht(xx)),xx,atol=1e-12,rtol=0)
        torch.testing.assert_close(op.H(null),torch.zeros_like(xx),atol=1e-12,rtol=0)
        torch.testing.assert_close(trial.null_projection(view,null),null,atol=1e-12,rtol=0)
        torch.testing.assert_close(((x-null)*null).sum(),x.new_tensor(0.),atol=1e-12,rtol=0)

    def test_temporal_signal_adds_only_nullspace_target(self):
        op = Operator('super_resolution','cpu',size=8)
        wave = SimpleNamespace(analysis=lambda x:x,synthesis=lambda x:x)
        q = torch.ones(1,3,8,8,dtype=torch.float64)
        earlier = torch.zeros_like(q)
        clean = (torch.arange(8,dtype=q.dtype)%2-.5).expand_as(q)*.1
        residual = torch.zeros_like(op.H(q))
        # Disable the Laplacian to isolate the projector and the target.
        from unittest.mock import patch
        with patch.object(trial.temporal,'STRENGTH',0.):
            for arm in trial.ARMS:
                counts,trace = dict(graph_builds=0,graph_actions=0),[]
                trial.native.initialize_counts(counts)
                remember,refine = trial.controller(op,wave,arm,.05,counts,trace)
                remember(earlier)
                remember(earlier)
                out = refine(q,residual,1.,dict(gamma=0.,weight=1.),0,clean)
                expected = -clean/(1+trial.graph.RIDGE) if arm=='TEMPORAL_NULL' else earlier
                torch.testing.assert_close(out,expected,atol=1e-12,rtol=0)
                torch.testing.assert_close(op.H(out),residual,atol=1e-12,rtol=0)
                self.assertLessEqual(trace[0]['objective_delta'],0.)


if __name__=='__main__':
    unittest.main()
