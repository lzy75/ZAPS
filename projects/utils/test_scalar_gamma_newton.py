import unittest
from utils import scalar_gamma_newton as rule


def quadratic(center,curvature):
    def evaluate(g,want_gradient,predicate):
        value=.5*curvature*(g-center)**2
        gradient=curvature*(g-center) if want_gradient and (predicate is None or predicate(value)) else None
        return dict(value=value,gradient=gradient)
    return evaluate


class SolverTests(unittest.TestCase):
    def test_fractional_and_above_two_minima(self):
        for center in (.37,1.23,3.4):
            r=rule.solve(quadratic(center,2.),2.)
            self.assertAlmostEqual(r['gamma'],center);self.assertTrue(r['stationarity_met'])

    def test_armijo_backtracks_without_integer_fallback(self):
        r=rule.solve(quadratic(.37,2.),1.)
        self.assertAlmostEqual(r['gamma'],.37)
        self.assertEqual(r['etas'],[.5]);self.assertEqual(r['value_points'],3)

    def test_positive_boundary_gradient(self):
        r=rule.solve(quadratic(-1.,2.),2.)
        self.assertEqual(r['gamma'],0.);self.assertTrue(r['stationarity_met'])

    def test_budget_is_not_convergence(self):
        r=rule.solve(quadratic(.37,2.),.1)
        self.assertFalse(r['stationarity_met']);self.assertIn('BUDGET',r['stop_reason'])

    def test_wrong_curvature_rejected(self):
        with self.assertRaises(ValueError):rule.solve(quadratic(.3,1),0.)


if __name__=='__main__':unittest.main()
