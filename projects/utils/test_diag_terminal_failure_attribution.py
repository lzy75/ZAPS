import unittest
import numpy as np
from utils import diag_terminal_failure_attribution as d


class DecompositionTests(unittest.TestCase):
    def test_fitting_observed_noise_can_reduce_loss_and_increase_true_error(self):
        p=d.decomposition([0.],[.5],[0.],[1.],sigma=1.)
        self.assertAlmostEqual(p['observed_loss_delta'],-.375)
        self.assertAlmostEqual(p['true_measurement_error_delta'],.125)
        self.assertAlmostEqual(p['noise_alignment_credit'],.5)
        self.assertLess(p['identity_absolute_error'],1e-12)

    def test_no_noise_and_both_terms_can_also_improve(self):
        p=d.decomposition([0.],[.5],[1.],[1.],sigma=1.)
        self.assertAlmostEqual(p['true_measurement_error_delta'],-.375)
        self.assertAlmostEqual(p['noise_alignment_credit'],0.)
        p=d.decomposition([0.],[.5],[1.],[1.2],sigma=1.)
        self.assertLess(p['true_measurement_error_delta'],0.)
        self.assertGreater(p['noise_alignment_credit'],0.)
        self.assertLess(p['observed_loss_delta'],p['true_measurement_error_delta'])

    def test_vector_identity_and_bad_input(self):
        rng=np.random.default_rng(42)
        for _ in range(10):
            p=d.decomposition(*(rng.normal(size=(2,3,4)) for i in range(4)))
            self.assertLess(p['identity_absolute_error'],1e-9)
        with self.assertRaises(ValueError):d.decomposition([0.],[1.,2.],[0.],[0.])
        with self.assertRaises(ValueError):d.decomposition([0.],[np.nan],[0.],[0.])


class ScopeTests(unittest.TestCase):
    def test_cohorts_remain_separate_and_no_new_NFE_or_control_claim(self):
        records=[]
        for cohort,count in (('A16',6),('A17',6),('A18',12)):
            for i in range(count):
                c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
                f0=dict(loss=1.01,psnr=20.,ssim=.7,lpips=.203)
                f1=dict(loss=.9,psnr=20.1,ssim=.701,lpips=.204)
                f2=dict(loss=.8,psnr=20.2,ssim=.702,lpips=.208 if cohort=='A18' else .204)
                records.append(dict(status='complete',case=dict(key=f'{cohort}_{i}',image_id=f'{cohort}_{i//2}'),cohort=cohort,
                    branches=dict(C=c,F0=f0,F1=f1,F2=f2),decomposition={a:dict(observed_loss_delta=0.,true_measurement_error_delta=0.,noise_alignment_credit=0.) for a in ('F0','F1','F2')},
                    geometry={a:dict(out_of_range_fraction=0.) for a in d.ARMS}))
        s=d.summarize(records)
        self.assertEqual((s['completed_cases'],s['image_clusters']),(24,12))
        self.assertEqual((s['method_nfe'],s['diagnostic_nfe']),(0,0))
        self.assertEqual(s['classification'],'OFFLINE_TERMINAL_ATTRIBUTION_COMPLETE')
        self.assertEqual(len(s['groups'][2]['F2_lpips_over_tolerance']),12)
        self.assertEqual(len(s['groups'][0]['F2_lpips_over_tolerance']),0)
        self.assertEqual(d.summarize(records[:1])['classification'],'INCOMPLETE_OFFLINE_ATTRIBUTION')


if __name__=='__main__':unittest.main()
