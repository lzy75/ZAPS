import unittest
from utils import diag_ieee_response_value as d
from utils import test_diag_clean_field_response as fixture


class ValueTests(unittest.TestCase):
    def rows(self,t=14):
        rows=[]
        for cohort,ids in (('development',('00020','00021','00022')),('transfer',('00023','00024','00025'))):
            for k,image in enumerate(ids):
                for j,seed in enumerate((1000,2000)):
                    signal=.2+.2*(2*k+j)+(.1 if cohort=='transfer' else 0.)
                    row=dict(cohort=cohort,case=f'{image}_s{seed}',image_id=image,sampling_seed=seed,source_t=t,usable=True,
                        log_R=0.,log_a=0.,log_orthogonal_gain=0.,log_gain_ratio=signal)
                    row.update({key:scale*signal for key,scale in zip(d.old.TARGETS,(1.,.1,.01,.2))});rows.append(row)
        return rows

    def test_direction_increment_and_generic_response_control(self):
        rows=self.rows();result=d.analyze_layer(rows,14)
        self.assertEqual(result['classification'],'IEEE_DIRECTION_INCREMENT_EXPLORATORY_CANDIDATE_ONLY')
        self.assertEqual(result['quality_targets_meeting_threshold'],3)
        for row in rows:row['log_orthogonal_gain']=row['log_gain_ratio']
        self.assertEqual(d.analyze_layer(rows,14)['classification'],'IEEE_DIRECTION_INCREMENT_NOT_SUPPORTED')

    def test_no_fit_instead_of_dropping_unusable_case(self):
        rows=self.rows();rows[0]['usable']=False
        self.assertEqual(d.analyze_layer(rows,14)['classification'],'UNRESOLVED_METROLOGY_NO_FIT')
        with self.assertRaises(ValueError):d.analyze_layer(rows[1:],14)

    def test_transfer_labels_do_not_change_fits_or_predictions(self):
        rows=self.rows(54);before=d.analyze_layer(rows,54)
        for row in rows:
            if row['cohort']=='transfer':
                for target in d.old.TARGETS:row[target]=999.
        after=d.analyze_layer(rows,54)
        for target in d.old.TARGETS:
            for name in d.MODELS:
                self.assertEqual(before['targets'][target]['models'][name]['predictions'],after['targets'][target]['models'][name]['predictions'])
                self.assertEqual(before['targets'][target]['models'][name]['fit'],after['targets'][target]['models'][name]['fit'])

    def test_feature_is_ratio_of_two_scale_means_and_missing_is_retained(self):
        response=d.field.assess(fixture.ResponseTests().payload())
        result=d.feature_row(dict(case='test'),response)
        self.assertTrue(result['usable']);self.assertAlmostEqual(result['q_gain'],2.)
        self.assertAlmostEqual(result['orthogonal_gain'],.5)
        import math
        self.assertAlmostEqual(result['log_gain_ratio'],math.log(4.))
        response['usable']=False;response['directions']['q']['pairs'][0]['gain']=None
        result=d.feature_row(dict(case='test'),response)
        self.assertFalse(result['usable']);self.assertIsNone(result['log_gain_ratio'])


if __name__=='__main__':unittest.main()
