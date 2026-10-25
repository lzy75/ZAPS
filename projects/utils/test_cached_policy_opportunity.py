import unittest
import numpy as np
from utils.cached_policy_opportunity import analyze


class OpportunityTests(unittest.TestCase):
    def test_mean_guard_oracle_can_have_space_when_no_fixed_improvement_is_feasible(self):
        values=np.zeros((2,2,4))
        values[:,1,0]=1.;values[0,1,2]=-.01;values[1,1,2]=.03
        r=analyze(values,('C','P'))
        self.assertEqual(r['best_fixed_feasible']['actions'],['C','C'])
        self.assertEqual(r['GT_cohort_mean_guard_oracle']['actions'],['P','C'])
        self.assertAlmostEqual(r['oracle_headroom_over_best_fixed'],.5)
        self.assertTrue(r['GT_cohort_mean_guard_oracle']['quality_guard'])
        self.assertFalse(r['leave_one_image_out_noise_only']['quality_guard'])

    def test_uniform_winner_has_no_individual_selection_space_and_aliases_add_none(self):
        values=np.zeros((3,3,4));values[:,1,0]=1.;values[:,2]=values[:,1]
        r=analyze(values,('C','P','ALIAS'))
        self.assertEqual(r['oracle_headroom_over_best_fixed'],0.)
        self.assertEqual(r['GT_cohort_mean_guard_oracle']['actions'],['P']*3)
        invalid=values.copy();invalid[0,0,0]=1.
        with self.assertRaises(ValueError):analyze(invalid,('C','P','ALIAS'))


if __name__=='__main__':unittest.main()
