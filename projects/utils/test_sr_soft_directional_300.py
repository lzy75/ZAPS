import unittest

from utils.diag_sr_soft_directional_300 import reliability, soft_update


class RuleTests(unittest.TestCase):
    def test_noise_is_estimated_after_ray_projection(self):
        scalar = dict(resolved=True, weight=.7, residual_energy=10.,
                      response_energy=2., residual_response_dot=4.)
        beta, remaining, noise = reliability(scalar, 9, .5)
        self.assertEqual((beta, remaining, noise), (.5, 2., 2.))
        self.assertLess(reliability(scalar, 9, 1.)[0], beta)
        self.assertEqual(reliability(dict(scalar, weight=0.), 9, .5)[0], 0.)

    def test_nonpositive_projection_does_not_subtract_a_degree_of_freedom(self):
        scalar = dict(resolved=True, weight=.7, residual_energy=10.,
                      response_energy=2., residual_response_dot=-4.)
        self.assertEqual(reliability(scalar, 9, .5)[1:], (10., 2.25))


class TorchTests(unittest.TestCase):
    def test_fractional_update_improves_proxy_and_keeps_zaps_gradient_bridge(self):
        import torch

        class Linear:
            def H(self, x):
                return x

            def transpose(self, x):
                return x

        class Wavelet:
            analysis = Linear.H
            synthesis = Linear.H

        class Zaps:
            dwt = Wavelet()

        q = torch.tensor([1., 1.], requires_grad=True)
        r = torch.tensor([2., 0.])
        counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
        candidate, detail, _ = soft_update(Zaps(), Linear(), torch.zeros_like(q), q, r,
                                            torch.tensor(1.), .5, counts)
        self.assertLess(detail['soft_proxy_final'], detail['soft_proxy_initial'])
        self.assertAlmostEqual(detail['beta'], 2./2.25)
        torch.testing.assert_close(candidate, torch.tensor([1.6527778, .0972222]))
        candidate.sum().backward()
        torch.testing.assert_close(q.grad, torch.ones_like(q))
