import math
import unittest
from utils.noise_feasible_multiplier import select, cases
from utils.continuous_response_feedback import select as original_a33


class ScalarTests(unittest.TestCase):
    def test_a33_matches_original_grid(self):
        for r2 in (.001,1.,100.):
            for d2 in (.0001,.1,10.):
                for cosine in (-.8,0.,.3,1.):
                    dot=cosine*math.sqrt(r2*d2)
                    self.assertAlmostEqual(select(dot,r2,d2,10,.05,'A33')['gamma'],
                                           original_a33(dot,r2,d2,2.)['gamma'],places=10)

    def test_nearest_is_not_old_first_crossing(self):
        # residual=(2-gamma)^2, noise-energy=2.25: interval[.5,2].
        self.assertEqual(select(2.,4.,1.,1,1.5,'FIRST_NOISE')['gamma'],.5)
        self.assertEqual(select(2.,4.,1.,1,1.5,'NEAREST_NOISE')['gamma'],1.)

    def test_increasing_noise_changes_action_without_changing_response(self):
        self.assertAlmostEqual(select(2.,4.,1.,1,.2,'NEAREST_NOISE')['gamma'],1.8)
        self.assertEqual(select(2.,4.,1.,1,1.5,'NEAREST_NOISE')['gamma'],1.)

    def test_infeasible_and_zero_response(self):
        r=select(1.,4.,1.,1,.1,'NEAREST_NOISE')
        self.assertFalse(r['noise_target_feasible']);self.assertEqual(r['gamma'],1.)
        self.assertEqual(select(0.,4.,0.,1,.1,'NEAREST_NOISE')['gamma'],1.)

    def test_projection_is_feasible_and_nearest_on_dense_grid(self):
        for dot,r2,d2,sigma in ((2,4,1,1.5),(.2,1,1,1.),(1,2,4,1.2)):
            row=select(dot,r2,d2,1,sigma,'NEAREST_NOISE')
            if row.get('noise_target_feasible'):
                g=row['gamma'];self.assertLessEqual(r2-2*g*dot+g*g*d2,sigma*sigma+1e-12)
                for j in range(2001):
                    x=j/1000
                    if r2-2*x*dot+x*x*d2<=sigma*sigma:
                        self.assertLessEqual(abs(g-1),abs(x-1)+1e-10)

    def test_cohort_fixed_old_images_no_silent_substitution(self):
        c=cases('/x');self.assertEqual(len(c),12)
        self.assertEqual(len({v['key'] for v in c}),12)


if __name__=='__main__':unittest.main()
