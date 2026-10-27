"""Pure aggregation gates for the early report; no tensor/model loading."""
from copy import deepcopy
import unittest
from utils import a56_ffhq_readout as r


class ReportTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for task in r.TASKS:
            for i in range(6):
                metrics={a:dict(psnr=20.,ssim=.7,lpips=.3) for a in r.ARMS}
                metrics['U_CURRENT']=dict(psnr=20.125,ssim=.71,lpips=.29)
                rows.append(dict(case=dict(dataset='ffhq',task=task,image_id=str(i),key=task+str(i)),metrics=metrics,
                    resolved={a:30 for a in ('U_CURRENT','V_PHYSICAL','Q_ORIGINAL','TIME_GAIN')}))
        return rows

    def test_complete_paired_cohort_and_effects(self):
        gs=r.aggregate(self.rows())
        self.assertEqual(len(gs),2)
        self.assertTrue(all(g['pullback_direction_transfer_candidate'] for g in gs))
        self.assertTrue(all(g['image_clusters']==6 for g in gs))
        self.assertEqual(gs[0]['U_minus']['C_IEEE']['psnr'],.125)
        self.assertEqual(gs[0]['U_positive_images']['V_PHYSICAL'],6)

    def test_gain_vs_C_does_not_imply_direction_increment(self):
        rows=self.rows()
        for x in rows:
            x['metrics']['V_PHYSICAL']['psnr']=21.
            x['metrics']['Q_ORIGINAL']['psnr']=21.
        gs=r.aggregate(rows)
        self.assertTrue(all(g['quality_transfer_candidate'] for g in gs))
        self.assertTrue(all(not g['pullback_direction_transfer_candidate'] for g in gs))

    def test_guards_and_resolution_remain_mandatory(self):
        for field,value in (('ssim',.697),('lpips',.306)):
            rows=self.rows()
            for x in rows:x['metrics']['U_CURRENT'][field]=value
            self.assertTrue(all(not g['quality_transfer_candidate'] for g in r.aggregate(rows)))
        rows=self.rows();rows[0]['resolved']['U_CURRENT']=29
        self.assertFalse(r.aggregate(rows)[0]['pullback_direction_transfer_candidate'])

    def test_no_missing_or_duplicate_case_subset(self):
        rows=self.rows()
        for variant in (rows[:-1],rows[:-1]+[deepcopy(rows[0])]):
            with self.assertRaises(ValueError):r.aggregate(variant)

    def test_no_imagenet_or_nonfinite_or_missing_arm(self):
        for change in ('dataset','nonfinite','missing'):
            rows=self.rows()
            if change=='dataset':rows[0]['case']['dataset']='imagenet'
            elif change=='nonfinite':rows[0]['metrics']['U_CURRENT']['psnr']=float('nan')
            else:del rows[0]['metrics']['C_REPLAY']
            with self.assertRaises(ValueError):r.aggregate(rows)

    def test_all_six_images_required_four_positive_at_least(self):
        rows=self.rows()
        for x in rows:
            if int(x['case']['image_id'])<3:x['metrics']['U_CURRENT']['psnr']=19.99
            else:x['metrics']['U_CURRENT']['psnr']=21.
        gs=r.aggregate(rows)
        self.assertTrue(all(g['U_minus']['C_IEEE']['psnr']>.05 for g in gs))
        self.assertTrue(all(not g['quality_transfer_candidate'] for g in gs))


if __name__=='__main__':unittest.main()
