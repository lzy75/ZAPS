"""A58 real Torch checks; missing Torch is a failure, never skipped."""
import unittest


class DesignTests(unittest.TestCase):
    def test_independent_executor_restores_prior_bindings(self):
        from utils import diag_blur_pcg_admission as d
        original = d.engine.check, d.engine.item, d.engine.sample
        with d.executor_bindings():
            self.assertIs(d.engine.sample, d.sample)
        self.assertEqual(original, (d.engine.check, d.engine.item, d.engine.sample))


class TorchTests(unittest.TestCase):
    def test_true_zero_boundary_target_matches_dense_solution(self):
        import torch
        import torch.nn.functional as F
        from utils.blur_pcg import preconditioned_cg, periodic_inverse
        torch.manual_seed(580027)
        shape = (1, 1, 9, 11)
        k = torch.tensor([[[[.02, .05, .03], [.1, .4, .12], [.04, .16, .08]]]], dtype=torch.float64)
        H = lambda x: F.conv2d(x, k, padding=1)
        HT = lambda x: F.conv2d(x, k.flip((-2, -1)), padding=1)
        eta = .003
        action = lambda x: .25*H(HT(x))+eta*x
        basis = torch.eye(99, dtype=torch.float64).reshape(99, 1, 9, 11)
        dense = action(basis).reshape(99, 99).T
        rhs = torch.randn(shape, dtype=torch.float64)
        inverse = periodic_inverse(k, shape, eta)
        x, gate = preconditioned_cg(action, rhs, inverse, rtol=1e-8, maxiter=128)
        expected = torch.linalg.solve(dense, rhs.flatten()).reshape(shape)
        self.assertTrue(gate['passed'])
        self.assertLessEqual(float((action(x)-rhs).norm()/rhs.norm()), 1e-8)
        self.assertLess(float((x-expected).norm()/expected.norm()), 1e-7)
        self.assertGreater(gate['preconditioner_calls'], 0)
        self.assertLessEqual(gate['iterations'], 128)
        self.assertGreater(float((rhs*inverse(rhs)).sum()), 0.)

    def test_budget_failure_stays_failure_and_zero_rhs(self):
        import torch
        from utils.blur_pcg import preconditioned_cg
        b = torch.tensor([1., 1.], dtype=torch.float64)
        action = lambda x: x*torch.tensor([1., 100.], dtype=torch.float64)
        _, gate = preconditioned_cg(action, b, lambda x: x, maxiter=1, rtol=1e-5)
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['iterations'], 1)
        self.assertEqual(gate['operator_calls'], 2)
        x, gate = preconditioned_cg(action, b*0, lambda x: x)
        self.assertTrue(gate['passed']); self.assertEqual(gate['operator_calls'], 0)


if __name__ == '__main__':
    unittest.main()
