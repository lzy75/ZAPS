import unittest
from utils import diag_cached_prior_ablation as d


def records():
    rows=[]
    for cohort,count in (('A16',6),('A17',6),('A18',12)):
        for i in range(count):
            c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
            f=dict(loss=.9,psnr=20.16,ssim=.702,lpips=.204)
            b=dict(loss=.89,psnr=20.15,ssim=.702,lpips=.201)
            rows.append(dict(status='complete',cohort=cohort,case=dict(key=f'{cohort}_{i}',image_id=f'{cohort}_{i//2}'),
                             branches=dict(C=c,F1=f,B=b),floors=dict.fromkeys(d.METRICS,0.)))
    return rows


class DecisionTests(unittest.TestCase):
    def test_fixed_ablation_retention_recovery_and_cost_are_distinct(self):
        s=d.summarize(records())
        for group in s['groups']:
            self.assertEqual(group['classification'],'PRIOR_DISPLACEMENT_ABLATION_EXPLORATORY_SUPPORT')
            self.assertAlmostEqual(group['retained_F1_psnr_gain_fraction'],.9375)
            self.assertAlmostEqual(group['recovered_F1_lpips_degradation_fraction'],.75)
        self.assertEqual((s['method_nfe'],s['diagnostic_nfe']),(0,0))
        self.assertEqual(s['deployment_method_nfe']['B'],301)

    def test_old_cohort_cannot_rescue_new_cohort_retention_failure(self):
        r=records()
        for row in r:
            if row['cohort']=='A18':row['branches']['B']['psnr']=20.14
        s=d.summarize(r)
        self.assertEqual(s['groups'][0]['classification'],'PRIOR_DISPLACEMENT_ABLATION_EXPLORATORY_SUPPORT')
        self.assertTrue(s['groups'][2]['B_acceptable_vs_C'])
        self.assertEqual(s['groups'][2]['classification'],'PRIOR_DISPLACEMENT_ABLATION_NOT_SUPPORTED')

    def test_recovery_threshold_and_undefined_denominator(self):
        r=records()
        for row in r:row['branches']['B']['lpips']=.203
        self.assertEqual(d.summarize(r)['groups'][2]['classification'],'PRIOR_DISPLACEMENT_ABLATION_NOT_SUPPORTED')
        for row in r:row['branches']['F1']['lpips']=.199
        self.assertIsNone(d.summarize(r)['groups'][2]['recovered_F1_lpips_degradation_fraction'])

    def test_four_of_six_image_clusters_and_incomplete_records(self):
        r=records()
        new_ids=sorted({row['case']['image_id'] for row in r if row['cohort']=='A18'})
        for row in r:
            if row['cohort']=='A18':row['branches']['B']['psnr']=21. if row['case']['image_id'] in new_ids[:3] else 19.99
        group=d.summarize(r)['groups'][2]
        self.assertGreater(group['comparisons']['B_C']['mean_delta']['psnr'],0.)
        self.assertFalse(group['B_acceptable_vs_C'])
        self.assertEqual(d.summarize(r[:1])['groups'][2]['classification'],'INCOMPLETE_PILOT')


if __name__=='__main__':unittest.main()
