"""Small exact convolution and residual checks, not server/model results."""
import unittest
import numpy as np
from utils.zero_boundary_fft import ZeroBlur, solve_budget, cg


def direct(x, kernel):
    ph, pw = [n//2 for n in kernel.shape]
    z = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    return sum(kernel[i, j]*z[:, i:i+x.shape[-2], j:j+x.shape[-1]]
               for i in range(kernel.shape[0]) for j in range(kernel.shape[1]))


class FFTTests(unittest.TestCase):
    def test_asymmetric_original_zero_boundary_and_adjoint(self):
        rng = np.random.default_rng(610027)
        k = rng.normal(size=(3, 5)); x = rng.normal(size=(2, 8, 7)); y = rng.normal(size=x.shape)
        op = ZeroBlur(k, x.shape)
        np.testing.assert_allclose(op.H(x), direct(x, k), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(op.transpose(y), direct(y, k[::-1, ::-1]), rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(op.H(x)*y)), float(np.sum(x*op.transpose(y))), places=10)

    def test_condition_budget_and_actual_residual(self):
        k = np.array([[0., .1, 0.], [.1, .6, .1], [0., .1, 0.]])
        op = ZeroBlur(k, (7, 6)); eta = .02; b = np.random.default_rng(4).normal(size=(1, 7, 6))
        budget = solve_budget(k, eta)
        action = lambda x: .25*op.H(op.transpose(x))+eta*x
        x, result = cg(action, b, budget['maximum_iterations'])
        self.assertTrue(result['passed'])
        self.assertLessEqual(float(np.linalg.norm(b-action(x))/np.linalg.norm(b)), 1e-5)
        self.assertIsNone(solve_budget(k, 1e-12)['maximum_iterations'])

    def test_failure_remains_failure(self):
        b = np.array([1., 1.]); matrix = np.diag([1., 100.])
        _, result = cg(lambda x: matrix@x, b, 1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'], 'ITERATION_BUDGET_EXHAUSTED')


if __name__ == '__main__': unittest.main()
