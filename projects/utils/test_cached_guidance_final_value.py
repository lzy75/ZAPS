"""A31 label isolation and known stage-only opportunity checks, no Torch."""
import unittest
import numpy as np
from utils.diag_cached_guidance_final_value import predict, analyze, TIMES, GAMMAS, METRICS, MODELS


class Tests(unittest.TestCase):
    def test_prediction_accepts_unlabelled_test_state(self):
        train=[dict(features={'x':float(i)},delta={'psnr':float(i)}) for i in range(5)]
        test=[dict(features={'x':2.5})]
        actual=predict(train,test,('x',),'psnr')
        self.assertTrue(np.isfinite(actual).all())
        test[0]['delta']={'psnr':1e20}
        np.testing.assert_array_equal(actual,predict(train,test,('x',),'psnr'))

    def test_constant_training_feature_does_not_use_test_statistics(self):
        train=[dict(features={'x':1.},delta={'psnr':v}) for v in (1.,2.,3.)]
        actual=predict(train,[dict(features={'x':1e10})],('x',),'psnr')
        self.assertEqual(float(actual[0]),2.)

    def test_known_fixed_winner_has_zero_oracle_headroom(self):
        rows=[]
        for image in ('a','b','c'):
            for seed in (1,2):
                for t in TIMES:
                    for g in GAMMAS:
                        gain=.3 if (t,g)==(319,2.) else -.1
                        rows.append(dict(case=f'{image}_{seed}',image_id=image,seed=seed,t=t,gamma=g,
                            arm=f'T{t}_G{g:g}',features={f:1. for f in MODELS['DUAL']},
                            delta=dict(psnr=gain,ssim=.0,lpips=.0,loss=.0)))
        r=analyze(rows)
        self.assertEqual(r['oracle_headroom_psnr'],0.)
        self.assertEqual(set(r['loio_policies']['DUAL']['actions']),{'T319_G2'})


if __name__=='__main__': unittest.main()
