import unittest
import numpy as np
from utils.diag_uncertainty_cache_audit import curve_report,closure_report


class CacheAuditTests(unittest.TestCase):
    def test_vertex_outside_bank_is_diagnostic_only(self):
        curve={str(float(g)):(g-4.)**2+1 for g in (0,.5,1,1.5,2)}
        result=curve_report(curve)
        self.assertAlmostEqual(result['quadratic_vertex'],4.)
        self.assertEqual(result['status'],'UPPER_BANK_BOUNDARY')
        self.assertTrue(result['vertex_is_unmeasured'])
        self.assertIsNone(curve_report({str(float(g)):1. for g in (0,.5,1,1.5,2)})['quadratic_vertex'])

    def test_discarded_columns_detected_without_changing_original_covariance(self):
        V=np.eye(3)[:,:2];C=np.array([[2.,0.,.4],[0.,1.,0.],[.4,0.,1.]])
        Y=C@V;B=V.T@Y;alpha=.5;factor=(1-alpha)/np.sqrt(alpha)
        result=closure_report(V,[B,B],[Y/factor,Y/factor],alpha)
        self.assertAlmostEqual(result['response_column_energy_outside_V'],.16/5.16)
        self.assertEqual(result['full_column_relative_scale_spread'],0.)
        with self.assertRaises(ValueError):closure_report(V,[B*2,B],[Y/factor,Y/factor],alpha)


if __name__=='__main__':unittest.main()
