import unittest
from types import SimpleNamespace

from utils.sr_bandwise_feedback import band_weights, controller


class NoiseTests(unittest.TestCase):
    def test_one_band_can_survive_a_global_noise_deficit(self):
        weights, noise = band_weights([20., 50.], [10, 90], [.25, .75], 1.)
        self.assertEqual(noise.sum(), 99.)
        self.assertGreater(weights[0], .5)
        self.assertEqual(weights[1], 0.)


class TorchTests(unittest.TestCase):
    def test_ray_and_diagonal_use_one_target_and_diagonal_descends(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D

        wavelet = OrthogonalDWT2D('db4', 3)
        operator = SimpleNamespace(H=lambda x: x, base=SimpleNamespace(transpose=lambda x: x))
        zaps = SimpleNamespace(dwt=wavelet)
        coeff = torch.zeros(1, 1, 64, 64)
        coeff[..., 0, 0] = 1
        coeff[..., 0, 32] = 1
        q = wavelet.synthesis(coeff).requires_grad_()
        coeff[..., 0, 0] = 16
        coeff[..., 0, 32] = 0
        residual = wavelet.synthesis(coeff)
        clean = torch.zeros_like(q)
        def run(diagonal, r):
            counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
            return controller(diagonal)(zaps, operator, clean, q, r, torch.ones(()), .1, counts)

        _, ray, _ = run(False, residual)
        output, diag, _ = run(True, residual)
        self.assertEqual(ray['weights'], diag['weights'])
        self.assertAlmostEqual(ray['proxy_at_ray'], diag['proxy_at_ray'], places=5)
        self.assertLess(diag['proxy_at_selected'], ray['proxy_at_selected'])
        self.assertTrue(output.requires_grad)
        zero, detail, _ = run(True, torch.zeros_like(residual))
        self.assertEqual(detail['reason'], 'ORIGINAL_OMEGA_ZERO')
        self.assertTrue(zero.requires_grad)
        self.assertEqual(float(zero.abs().sum()), 0.)
