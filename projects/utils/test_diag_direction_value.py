import unittest
from utils import diag_direction_value as d


class DirectionValueTests(unittest.TestCase):
    def rows(self,cohort):
        ids=('00020','00021','00022') if cohort=='development' else ('00023','00024','00025')
        rows=[]
        for k,image in enumerate(ids):
            for j,seed in enumerate((1000,2000)):
                turn=.2+.2*(2*k+j)+(.1 if cohort=='transfer' else 0)
                row=dict(cohort=cohort,case=f'{image}_s{seed}',image_id=image,sampling_seed=seed,source_t=14,
                         usable=True,log_R=0.,log_a=0.,turn=turn,log1p_omega=0.)
                row.update({target:scale*turn for target,scale in zip(d.TARGETS,(1.,.1,.01,.2))});rows.append(row)
        return rows

    def test_fixed_transfer_fit_never_reads_transfer_targets(self):
        train=self.rows('development');test=self.rows('transfer')
        predicted,fit=d.ridge_predict(train,test,d.DIRECTION,'gain_psnr')
        for r in test:r['gain_psnr']=1000000.
        new,changed=d.ridge_predict(train,test,d.DIRECTION,'gain_psnr')
        self.assertEqual(predicted,new);self.assertEqual(fit,changed)
        self.assertIn('log_R',fit['constant_columns'])

    def test_joint_transfer_threshold_and_image_grouping(self):
        rows=self.rows('development')+self.rows('transfer')
        layer=d.analyze_layer(rows,14)
        self.assertEqual(layer['classification'],'INCREMENTAL_DIRECTION_TRANSFER_CANDIDATE_ONLY')
        self.assertEqual(layer['quality_targets_meeting_threshold'],3)
        target=layer['targets']['gain_psnr']
        self.assertEqual(len(target['direction']['per_image_mae']),3)
        self.assertEqual(len(target['development_cv']['direction']['folds']),3)
        # With no directional variation both models reduce to the same phase mean: not evidence of increment.
        for r in rows:r['turn']=0.
        self.assertEqual(d.analyze_layer(rows,14)['classification'],'DIRECTION_INCREMENT_NOT_SUPPORTED_IN_THIS_TRANSFER_TEST')

    def test_missing_readout_retained_no_fit_and_wrong_split_refused(self):
        rows=self.rows('development')+self.rows('transfer');rows[0]['usable']=False
        self.assertEqual(d.analyze_layer(rows,14)['classification'],'UNRESOLVED_INPUTS_NO_FIT')
        with self.assertRaises(ValueError):d.analyze_layer(rows[:-1],14)
        missing=d.features(dict(residual_noise_ratio=1.,guidance_rms=.1,guidance_cosine=None,guidance_variation3_per_u=2.))
        self.assertFalse(missing['usable']);self.assertIn('turn',missing)

    def test_bridge_average_gains_and_direction_chord(self):
        case=dict(key='00020_s1000',image_id='00020',sampling_seed=1000)
        row=dict(t=14,noise_u=.8,residual_noise_ratio=1.,guidance_rms=.1,guidance_cosine=1.,guidance_variation3_per_u=0.)
        base=dict(psnr=20.,ssim=.7,lpips=.3,loss=1.)
        branches=dict(F0=dict(psnr=22.,ssim=.72,lpips=.2,loss=.9),F1=dict(psnr=20.,ssim=.7,lpips=.3,loss=1.))
        point=d.observation('development',case,row,base,branches)
        self.assertEqual(point['gain_psnr'],1.);self.assertEqual(point['turn'],0.)
        self.assertAlmostEqual(point['gain_lpips'],.05);self.assertEqual(point['bridge_gap_psnr'],2.)

    def test_progress_is_explicit_milestone_rubric_and_no_early_credit(self):
        p=d.progress();self.assertEqual(sum(v['weight'] for v in p['rubric']),100)
        self.assertEqual(p['percent'],45)
        self.assertTrue(all(v['earned']==0 for v in p['rubric'][3:]))


if __name__=='__main__':unittest.main()
