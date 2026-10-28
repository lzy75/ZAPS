import unittest
from utils import next_response_bayes as rule


class AlgebraTests(unittest.TestCase):
    def test_no_hidden_multiplier_two_cap(self):
        self.assertGreater(rule.proposal(10.,1.,1.,.05,1.),2.)

    def test_more_observation_noise_reduces_correction(self):
        self.assertGreater(rule.proposal(1,1,1,.05,1),rule.proposal(1,1,1,.2,1))

    def test_more_uncertainty_allows_larger_correction(self):
        self.assertGreater(rule.proposal(1,1,1,.1,2),rule.proposal(1,1,1,.1,.1))
        self.assertGreater(rule.diffusion_variance(.2),rule.diffusion_variance(.99))

    def test_linear_quadratic_minimum(self):
        g=rule.proposal(3.,2.,4.,.2,.3)
        def loss(x):return (8-6*x+2*x*x)/(.08)+x*x*4/(.6)
        self.assertLessEqual(loss(g),loss(0));self.assertLessEqual(loss(g),loss(1))
        self.assertAlmostEqual((2*g-3)/(.04)+4*g/(.3),0.)

    def test_nonnegative_and_fixed_old_cohort(self):
        self.assertEqual(rule.proposal(-1,1,1,.1,1),0)
        self.assertEqual(len(rule.cases('/images')),8)

    def test_invalid_variance_rejected(self):
        with self.assertRaises(ValueError):rule.proposal(1,1,1,.05,-1)


if __name__=='__main__':unittest.main()
