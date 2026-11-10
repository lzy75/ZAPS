"""Required real-Torch algebra tests for A101; the server must not skip these."""
import unittest


class TorchTests(unittest.TestCase):
    def test_observed_tensor_is_adjoint_of_informative(self):
        import torch
        from utils import equal_nfe_feedback_runtime as runtime
        class Operator: pass
        op = Operator(); op._mask_cache = torch.tensor([[[[1., 0.], [0., 1.]]]])
        x = torch.arange(12., dtype=torch.float32).reshape(1, 3, 2, 2)
        y = torch.linspace(-1, 1, 6)
        lhs = (runtime._informative(x, op).double() * y.double()).sum()
        rhs = (x.double() * runtime._observed_tensor(y, x, op).double()).sum()
        torch.testing.assert_close(lhs, rhs)

    def test_linear_diagonal_step_contains_scalar_and_descends(self):
        import torch
        from utils import equal_nfe_feedback_runtime as runtime
        class IdentityDWT:
            def analysis(self, x): return x
            def synthesis(self, x): return x
        class Operator:
            _mask_cache = None
            def H(self, x): return x
            def transpose(self, x): return x
        class Z: dwt = IdentityDWT()
        clean = torch.zeros(1, 1, 2, 2)
        q = torch.tensor([[[[1., 2.], [3., 4.]]]])
        residual = torch.tensor([[[[2., -1.], [.5, 3.]]]])
        counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0,
                      module_adjoint_actions=0)
        delta, detail, counts = runtime.diagonal_update(
            Z(), Operator(), clean, q, residual, torch.tensor(.8), .1, counts)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertLessEqual(detail['proxy_final'], detail['proxy_initial'] + 1e-7)
        self.assertEqual(counts['module_adjoint_actions'], 1)
        self.assertEqual(counts['module_dwt_analysis'], 2)
        self.assertEqual(counts['module_dwt_synthesis'], 3)


if __name__ == '__main__':
    unittest.main()
