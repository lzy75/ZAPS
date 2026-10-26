"""Server-required actual SR adjoint regression, not skipped without Torch."""
import unittest
import torch
from modules.degradations import SuperResolutionOperator
from utils.linear_adjoint_autograd_guard import operator_only_autograd


class ExactAdjointTests(unittest.TestCase):
    def test_real_sr_adjoint_allowed_and_unchanged(self):
        op = SuperResolutionOperator(scale_factor=2, noise_sigma=.05, in_shape=(1, 3, 16, 16))
        x = torch.linspace(-.5, .5, 3*16*16).reshape(1, 3, 16, 16)
        y = torch.linspace(-.2, .3, 3*8*8).reshape(1, 3, 8, 8)
        expected = op.transpose(y)
        counts = {}
        with operator_only_autograd(counts):
            actual = op.transpose(y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        left, right = (op.H(x).double()*y.double()).sum(), (x.double()*actual.double()).sum()
        self.assertLessEqual(float(abs(left-right)), 1e-6*max(1., float(abs(left))))
        self.assertEqual(counts, dict(operator_grad_attempts=1, operator_grad_completed=1, forbidden_grad_attempts=0))

    def test_non_operator_input_gradient_is_still_forbidden(self):
        counts = {}; x = torch.tensor([.3, .7], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'forbids denoiser/parameter'):
            with operator_only_autograd(counts):
                torch.autograd.grad(x.square().sum(), x)
        self.assertEqual(counts['forbidden_grad_attempts'], 1)


if __name__ == '__main__':
    unittest.main()
