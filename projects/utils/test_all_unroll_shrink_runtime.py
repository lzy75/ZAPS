"""Mandatory Torch test of original training and all10 controlled unrolls."""
import unittest
import torch
from utils import all_unroll_shrink_runtime as runtime
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters,snapshot_equal
from utils.test_paper_clock_runtime import TinyDiffusion,Linear


class RuntimeTests(unittest.TestCase):
    def test_all_unroll_gamma1_matches_original_output_parameters_and_losses(self):
        dm=TinyDiffusion();case=dict(task='super_resolution',seed=1226,sigma=.05,mask_seed=2001226)
        y=torch.full((1,3,16,16),.15);outputs={};details={}
        with frozen_network(dm.model):
            for arm in ('BASE','NULL_ALL','ALL_UNROLLS'):
                with counters(dm.model) as count:
                    out,detail=runtime.execute(dm,Linear(),y,case,arm,size=16,level=1)
                self.assertEqual((count['forwards'],count['optimizer_backward_completed'],count['input_vjp_attempts']),(300,10,0))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(all(p.grad is None for p in dm.model.parameters()))
                outputs[arm]=out;details[arm]=detail
            torch.testing.assert_close(outputs['BASE'],outputs['NULL_ALL'],rtol=1e-6,atol=1e-6)
            for name in ('zeta','D','losses'):
                torch.testing.assert_close(torch.as_tensor(details['BASE'][name]),torch.as_tensor(details['NULL_ALL'][name]),rtol=1e-6,atol=1e-6)
            self.assertTrue(snapshot_equal(details['BASE']['first_entry'],details['ALL_UNROLLS']['first_entry'])['passed'])
            trace=details['ALL_UNROLLS']['trace']
            self.assertEqual(len(trace),300)
            for u in range(1,11):
                rows=[r for r in trace if r['unroll']==u]
                self.assertEqual(len(rows),30)
                self.assertTrue(any(r['gamma']!=1. for r in rows))


if __name__=='__main__':unittest.main()
