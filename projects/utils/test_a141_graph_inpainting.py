"""Verify graph signs, diagonal feasibility, and native-sampler insertion."""
from types import SimpleNamespace
import unittest

import torch

from utils import diag_a141_graph_inpainting as trial
from utils import dps_missing_tasks_runtime, pigdm_noisy_r1_runtime
from utils.pigdm_noisy_runtime import Operator


class GraphTests(unittest.TestCase):
    def test_laplacian_is_symmetric_and_energy_is_weighted_edge_difference(self):
        gen = torch.Generator().manual_seed(141)
        x = torch.randn((1,3,5,7),generator=gen,dtype=torch.float64)
        y = torch.randn(x.shape,generator=gen,dtype=x.dtype)
        east, south = weights = trial.edge_weights(x,.05)
        explicit = (east*x.diff(dim=-1).square()).sum()+(south*x.diff(dim=-2).square()).sum()
        torch.testing.assert_close((x*trial.laplacian(x,weights)).sum(),explicit)
        torch.testing.assert_close((y*trial.laplacian(x,weights)).sum(),(x*trial.laplacian(y,weights)).sum())
        t = torch.ones_like(x)
        torch.testing.assert_close(trial.laplacian(t,weights),torch.zeros_like(t))

    def test_cg_matches_dense_solution(self):
        matrix = torch.tensor([[2.5,-.3],[-.3,1.0]],dtype=torch.float64)
        rhs = torch.tensor([.1,.2],dtype=matrix.dtype)
        result,objectives,residual,stop = trial.conjugate_gradient(
            lambda x:matrix@x,rhs,trial._dot,12,1e-10)
        torch.testing.assert_close(result,torch.linalg.solve(matrix,rhs),atol=1e-12,rtol=0)
        self.assertLess(objectives[-1],0)
        self.assertLessEqual(residual,1e-10)
        self.assertEqual(stop,'projected_gradient_tolerance')

    def test_graph_correction_reaches_missing_pixels_within_native_support(self):
        pigdm = Operator('inpainting','cpu',size=8,seed=139)
        mask = pigdm.mask[None,None]
        dps = SimpleNamespace(_mask_cache=mask,H=lambda x:x*mask,transpose=lambda x:x*mask)
        wave = SimpleNamespace(analysis=lambda x:x,synthesis=lambda x:x)
        for operator in (dps,pigdm):
            physical = trial.parent.ObservedResponse(operator)
            q = torch.ones(1,3,8,8)
            q[...,0,0] = 0
            clean = torch.zeros_like(q)
            residual = physical.H(q)*.4
            count, trace = dict(graph_builds=0,graph_actions=0), []
            trial.native.initialize_counts(count)
            fn = trial.controller(operator,wave,.5,.05,count,trace)
            out = fn(q,residual,1.,dict(gamma=.1,weight=1.),0,clean)
            self.assertTrue(torch.equal(out[q==0],q[q==0]))
            self.assertGreater(trace[0]['missing_energy'],0)
            self.assertLess(trace[0]['objective_delta'],0)
            self.assertLessEqual(trace[0]['updates'],12)
            self.assertEqual(count['graph_builds'],1)

    def test_native_edits_compile(self):
        for host,runtime in (('dps',dps_missing_tasks_runtime),('pigdm',pigdm_noisy_r1_runtime)):
            sampler = trial.instrument(runtime,host,(0,),lambda *_:None)
            self.assertEqual(sampler.__globals__['positions'],{0})
            self.assertEqual(sampler.__defaults__,runtime.run.__defaults__)
            self.assertEqual(sampler.__kwdefaults__,runtime.run.__kwdefaults__)


if __name__=='__main__':
    unittest.main()
