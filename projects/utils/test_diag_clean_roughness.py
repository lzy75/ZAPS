import unittest
from unittest.mock import patch
import numpy as np
from utils import diag_clean_roughness as d
try:
    import torch
except ImportError:
    torch=None


class RoughnessTests(unittest.TestCase):
    def trajectory(self):
        rows=[dict(t=999-i,noise_u=i/30) for i in range(30)]
        for j,t,u in ((26,124,.6),(27,54,.7),(28,14,.9)):
            rows[j]=dict(t=t,noise_u=u)
        return np.zeros((30,2)),rows

    def test_nonuniform_linear_has_speed_without_acceleration(self):
        z,rows=self.trajectory()
        for j in (26,27,28):z[j]=3*rows[j]['noise_u']+2
        r=d.geometry(z,rows)
        self.assertAlmostEqual(r['clean_speed'],3.)
        self.assertAlmostEqual(r['clean_roughness'],0.,places=11)

    def test_nonuniform_quadratic_recovers_second_derivative(self):
        z,rows=self.trajectory()
        for j in (26,27,28):z[j]=2*rows[j]['noise_u']**2
        self.assertAlmostEqual(d.geometry(z,rows)['clean_roughness'],4.)

    def test_equal_speed_turn_and_future_independence(self):
        z,rows=self.trajectory();z[26]=[0,0];z[27]=[.1,0];z[28]=[.1,.2]
        r=d.geometry(z,rows)
        self.assertGreater(r['clean_roughness'],0.)
        z[29]=np.nan;rows[29]['noise_u']=None
        self.assertEqual(r,d.geometry(z,rows))
        rows[27]['noise_u']=rows[26]['noise_u']
        with self.assertRaises(ValueError):d.geometry(z,rows)

    def observations(self):
        result=[]
        for cohort,ids in (('development',('00020','00021','00022')),('transfer',('00023','00024','00025'))):
            for k,image in enumerate(ids):
                for j,seed in enumerate((1000,2000)):
                    signal=.2+.2*(2*k+j)+(.1 if cohort=='transfer' else 0.)
                    r=dict(case=f'{image}_s{seed}',image_id=image,sampling_seed=seed,cohort=cohort,source_t=14,
                           usable=True,log_R=0.,log_a=0.,log1p_speed=0.,log1p_roughness=signal)
                    r.update({t:s*signal for t,s in zip(d.old.TARGETS,(1.,.1,.01,.2))});result.append(r)
        return result

    def test_frozen_transfer_threshold_speed_control_and_grouping(self):
        rows=self.observations();r=d.analyze(rows)
        self.assertEqual(r['classification'],'CLEAN_ROUGHNESS_EXPLORATORY_CANDIDATE_ONLY')
        self.assertEqual(r['quality_targets_meeting_threshold'],3)
        self.assertEqual(len(r['targets']['gain_psnr']['models']['roughness']['per_image_mae']),3)
        for row in rows:row['log1p_speed']=row['log1p_roughness']
        self.assertEqual(d.analyze(rows)['classification'],'CLEAN_ROUGHNESS_INCREMENT_NOT_SUPPORTED')

    def test_prediction_does_not_fit_transfer_outcomes(self):
        rows=self.observations();before=d.analyze(rows)
        for row in rows:
            if row['cohort']=='transfer':
                for t in d.old.TARGETS:row[t]=999.
        after=d.analyze(rows)
        for t in d.old.TARGETS:
            for m in d.MODELS:
                self.assertEqual(before['targets'][t]['models'][m]['predictions'],after['targets'][t]['models'][m]['predictions'])
                self.assertEqual(before['targets'][t]['models'][m]['fit'],after['targets'][t]['models'][m]['fit'])

    def test_missing_or_wrong_split_is_not_replaced(self):
        rows=self.observations();rows[0]['usable']=False
        self.assertEqual(d.analyze(rows)['classification'],'UNRESOLVED_INPUTS_NO_FIT')
        with self.assertRaises(ValueError):d.analyze(rows[:-1])
        rows[0]['source_t']=54
        with self.assertRaises(ValueError):d.analyze(rows)


@unittest.skipIf(torch is None,'CPU Torch loader contract runs on server; no model required')
class StorageTests(unittest.TestCase):
    def test_cpu_float32_cache_contract_and_no_future_read(self):
        _,rows=RoughnessTests().trajectory();clean=torch.zeros((30,3,256,256),dtype=torch.float32)
        clean[29]=float('nan')
        with patch('torch.load',return_value={'clean':clean}) as loader:
            self.assertEqual(d.load_geometry('fixed.pt',rows)['clean_roughness'],0.)
            loader.assert_called_once_with('fixed.pt',map_location='cpu',weights_only=True)
        with patch('torch.load',return_value={'clean':clean[:29]}):
            with self.assertRaises(ValueError):d.load_geometry('fixed.pt',rows)


if __name__=='__main__':unittest.main()
