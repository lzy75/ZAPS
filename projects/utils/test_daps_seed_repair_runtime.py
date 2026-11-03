"""Retain all original gates, correcting NumPy seeding in the real Motion fixture."""
import torch
from utils import daps_seed_repair as runtime
from utils.test_daps_multiplier_runtime import TorchTests as OriginalTests


class TorchTests(OriginalTests):
    def test_original_motion_state_reproducible_and_input_gradient_finite(self):
        case=dict(task='motion_deblur',kernel_seed=30226)
        first=runtime.make_operator(case,'cpu');state=runtime.operator_state(first)
        second=runtime.make_operator(case,'cpu');other=runtime.operator_state(second)
        self.assertEqual(set(state),set(other))
        self.assertTrue(all(torch.equal(value,other[key]) for key,value in state.items()))
        x=torch.randn(1,3,64,64)
        # Match Operator.gradient's forward context when constructing an exact zero-residual oracle.
        y=first(x.clone().detach().requires_grad_(True)).detach()
        gradient,loss=first.gradient(x,y,return_loss=True)
        self.assertTrue(torch.isfinite(gradient).all());self.assertEqual(float(loss),0.)


# Do not let unittest discover the imported uncorrected base class as another test suite.
del OriginalTests
