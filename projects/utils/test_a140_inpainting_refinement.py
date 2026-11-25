"""Check the observed-window goal and the step applied by each A140 arm."""
from types import SimpleNamespace
import unittest

import torch

from utils import diag_a140_inpainting_refinement as trial
from utils.pigdm_noisy_runtime import Operator


class InpaintingRefinementTests(unittest.TestCase):
    def operators(self):
        pigdm = Operator('inpainting','cpu',size=8,seed=139)
        mask = pigdm.mask[None,None]
        yield pigdm
        yield SimpleNamespace(_mask_cache=mask, H=lambda x:x*mask, transpose=lambda x:x*mask)

    def test_local_target_matches_direct_observed_window(self):
        image = torch.arange(3*8*8,dtype=torch.float64).reshape(1,3,8,8)
        for op in self.operators():
            physical = trial.parent.ObservedResponse(op)
            y = physical.H(image)
            mask = physical.Ht(torch.ones_like(y)).bool()
            count = {}
            trial.native.initialize_counts(count)
            view = trial.native.LinearView(physical,count)
            support = trial.box_sum(view.transpose(torch.ones_like(y)))
            estimate = trial.local_target(view,y,support)
            expected = torch.zeros_like(image)
            for channel in range(3):
                for row,col in mask[0,channel].nonzero().tolist():
                    height = slice(max(0,row-2),row+3)
                    width = slice(max(0,col-2),col+3)
                    observed = mask[0,channel,height,width]
                    if int(observed.sum())>=trial.MIN_SAMPLES:
                        expected[0,channel,row,col] = image[0,channel,height,width][observed].mean()
            torch.testing.assert_close(estimate,physical.H(expected),rtol=0,atol=1e-12)

    def test_controller_moves_from_the_same_scalar_anchor(self):
        q = torch.ones(1,3,8,8)
        residual_image = torch.arange(3*8*8,dtype=torch.float32).reshape_as(q)/200
        scalar = dict(weight=.5,gamma=.2)
        wave = SimpleNamespace(analysis=lambda x:x,synthesis=lambda x:x)
        for op in self.operators():
            physical = trial.parent.ObservedResponse(op)
            r = physical.H(residual_image)
            anchor = scalar['gamma']*q
            error = scalar['weight']*r-physical.H(anchor)
            for arm in trial.CANDIDATES:
                count,trace = dict(local_box_sums=0),[]
                trial.native.initialize_counts(count)
                refine = trial.controller(op,arm,wave,count,trace)
                out = refine(op,q,r,1.,scalar,0)
                torch.testing.assert_close((out-anchor)[~physical.Ht(torch.ones_like(r)).bool()],
                                           torch.zeros_like((out-anchor)[~physical.Ht(torch.ones_like(r)).bool()]))
                if arm==trial.LOCAL:
                    view = trial.native.LinearView(physical,count)
                    support = trial.box_sum(view.transpose(torch.ones_like(error)))
                    expected = trial.local_target(view,error,support)
                else:
                    expected = trial.DAMPING[arm]*error
                torch.testing.assert_close(physical.H(out-anchor),expected,atol=1e-6,rtol=1e-5)
                self.assertLessEqual(trace[0]['proxy']['deep'],trace[0]['proxy']['scalar'])


if __name__=='__main__':
    unittest.main()
