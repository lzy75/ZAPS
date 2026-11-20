"""Gates for the new A117 controls. Torch is required by the server gate."""
import unittest
from unittest.mock import patch
from utils import diag_thesis_feedback_confirmation as driver


class Controls(unittest.TestCase):
    def test_fixed_and_no_shrink_only_replace_multiplier(self):
        def suffix(model, operator, y, case, arm, prefix):
            self.assertEqual(arm, driver.SCALAR)
            return [driver.baseline.runtime.scalar_rule.select(4., 9., 4., 6, .5)
                    for _ in range(30)]

        case = dict(fixed_gamma=[.2 + k / 100 for k in range(30)])
        with patch.object(driver, 'BASE_SUFFIX', suffix):
            fixed = driver.adapted_suffix(None, None, None, case, driver.FIXED, None)
            raw = driver.adapted_suffix(None, None, None, case, driver.NOSHRINK, None)
        self.assertEqual([r['gamma'] for r in fixed], case['fixed_gamma'])
        self.assertTrue(all(r['weight'] == 1 and r['gamma'] == 1 for r in raw))
        self.assertIs(driver.baseline.runtime.scalar_rule.select, driver.SCALAR_SELECT)

    def test_pixel_line_search_matches_full_support_wavelet(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D

        class Operator:
            def __init__(self, weight):
                self.weight = weight

            def H(self, value):
                return self.weight * value

            def transpose(self, value):
                return self.weight * value

        generator = torch.Generator().manual_seed(117)
        clean = torch.randn((1, 3, 32, 32), generator=generator).double()
        q = torch.randn((1, 3, 32, 32), generator=generator).double().requires_grad_()
        residual = q.detach() + torch.randn(q.shape, generator=generator).double()
        op = Operator(torch.linspace(.5, .9, 32, dtype=torch.float64).reshape(1, 1, 1, 32))
        zaps = type('Fixture', (), {'dwt': OrthogonalDWT2D(level=3).double()})()
        counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
        # The production filter coefficients are stored at float32 precision.
        clean, q, residual = clean.float(), q.detach().float().requires_grad_(), residual.float()
        zaps.dwt = zaps.dwt.float()
        output, row, _ = driver.pixel_step(zaps, op, clean, q, residual, torch.tensor(.8), .05, counts)
        self.assertEqual(row['nonzero_support_fraction'], 1.)
        self.assertLess(row['same_state_correction_relative_error'], 2e-5)
        self.assertLess(row['proxy_final'], row['proxy_initial'])
        output.sum().backward()
        torch.testing.assert_close(q.grad, torch.ones_like(q))


if __name__ == '__main__':
    unittest.main()
