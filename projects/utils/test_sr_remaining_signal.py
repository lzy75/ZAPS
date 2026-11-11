import unittest

from utils.diag_sr_remaining_signal import corrected_update, remaining_signal
from utils.noise_signal_shrink import select


class RuleTests(unittest.TestCase):
    def test_original_ray_is_kept_when_remaining_energy_is_below_noise(self):
        scalar = select(2., 5., 1., 4, 1.)
        self.assertGreater(scalar['weight'], 0.)
        result = remaining_signal(scalar, 4, 1.)
        self.assertEqual(result['residual_energy'], 1.)
        self.assertEqual(result['noise_energy'], 3.)
        self.assertEqual(result['beta'], 0.)

    def test_surplus_is_rescaled_relative_to_original_target(self):
        scalar = select(2., 10., 1., 4, 1.)
        result = remaining_signal(scalar, 4, 1.)
        self.assertAlmostEqual(result['beta'] * scalar['weight'], .5)
        self.assertEqual(remaining_signal(select(2., 20., 1., 4, 1.), 4, 1.)['beta'], 1.)


class TorchTests(unittest.TestCase):
    def test_zero_remaining_signal_returns_exact_scalar_with_graph(self):
        import torch

        scalar = select(2., 5., 1., 4, 1.)

        def original(zaps, operator, clean, q, residual, alpha, sigma, counts):
            return q + 100, {'scalar': scalar}, counts

        q = torch.ones(4, requires_grad=True)
        chosen, detail, _ = corrected_update(original)(None, None, q, q, q, None, 1., {})
        torch.testing.assert_close(chosen, scalar['gamma']*q, rtol=0, atol=0)
        chosen.sum().backward()
        torch.testing.assert_close(q.grad, torch.full_like(q, scalar['gamma']))
        self.assertEqual(detail['remaining_signal']['applied_increment_norm'], 0.)
