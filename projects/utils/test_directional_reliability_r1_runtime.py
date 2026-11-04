"""A91R1: preserve A91 gates; use scale-aware float32 FFT homogeneity checks."""
import unittest
import torch
from modules.wavelet import OrthogonalDWT2D
from utils import directional_reliability as rt
from utils import test_directional_reliability_runtime as parent


class Tests(parent.Tests):
    def test_identity_zero_and_zeta_homogeneity_without_projection(self):
        torch.manual_seed(8)
        q = torch.randn(1, 3, 32, 32)
        dwt = OrthogonalDWT2D('db4', 3)
        bank = {k: torch.ones_like(q) for k in ('unit_variance', 'first_half', 'second_half')}
        factor = torch.full((3, 32, 32), .3)
        one, _ = rt.shrink(q, factor, dwt, bank, 0.)
        torch.testing.assert_close(one, q, rtol=0, atol=0)
        zero, _ = rt.shrink(q, factor, dwt, bank, 100.)
        self.assertEqual(float(zero.abs().max()), 0.)
        a, ra = rt.shrink(q, factor, dwt, bank, .1)
        b, rb = rt.shrink(3*q, 3*factor, dwt, bank, .1)
        expected = 3*a
        error = (b.double()-expected.double()).norm()/expected.double().norm()
        eps = torch.finfo(q.dtype).eps
        # FFT analysis/synthesis roundoff is absolute near zero. Keep a separate
        # strict aggregate check so a broad error cannot pass via the local floor.
        self.assertLessEqual(float(error), 16*eps)
        torch.testing.assert_close(b, expected, rtol=1e-5, atol=32*eps)
        torch.testing.assert_close(torch.tensor(ra['weights']), torch.tensor(rb['weights']),
                                   rtol=1e-5, atol=1e-7)
        print('A91R1 HOMOGENEITY:', dict(relative_L2=float(error),
              max_abs=float((b-expected).abs().max()), element_atol=32*eps), flush=True)


if __name__ == '__main__':
    unittest.main()
