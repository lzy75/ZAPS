import unittest
from types import SimpleNamespace
from utils import one_sided_noise_feedback as rules
from utils import continuous_response_feedback as old
from utils import noise_regularized_feedback as reg


class SideTests(unittest.TestCase):
    def test_separates_both_sides_and_retains_exact_original_or_regularized_action(self):
        for dot, r2, a2, count, sigma in ((.1,1.,1.,1,.3), (1.6,4.,1.,1,.5),
                                         (0.,1.,0.,1,.05), (1.,1.,1.,1,.05)):
            original=old.select(dot,r2,a2,2.)
            constrained=reg.select(dot,r2,a2,count,sigma)
            for arm in rules.ARMS:
                result=rules.select(dot,r2,a2,count,sigma,arm)
                raw=original['raw_gamma']
                active=raw is not None and (raw>1 if arm=='UP_REG' else raw<1)
                self.assertEqual(result['gamma'],constrained['gamma'] if active else original['gamma'])
                self.assertEqual(result['selected_component'],'REG' if active else 'OLD')
        self.assertEqual(18*rules.CALLS,1080)
        with self.assertRaises(ValueError):rules.select(1.,1.,1.,1,.05,'UNKNOWN')

    def test_actual_torch_sensor_does_not_mutate_inputs(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; mandatory on server')
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.ones(1,dtype=torch.float64)),
                          A=SimpleNamespace(H=lambda x:2*x))
        q=torch.tensor([1.,2.],dtype=torch.float64);clean=torch.zeros_like(q);y=.31*2*q
        inputs=[v.clone() for v in (q,clean,y)]
        for arm in rules.ARMS:
            actual=rules.sensor(z,y,clean,q,0,.05,arm)
            expected=rules.select(6.2,1.922,20.,2,.05,arm)
            self.assertAlmostEqual(actual['gamma'],expected['gamma'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip((q,clean,y),inputs)))

    def test_cross_condition_candidate_must_be_the_same_arm(self):
        from utils.diag_one_sided_noise_feedback import interpretation
        records=[]
        for i in range(6):
            for sigma in (.025,.05,.1):
                baseline=dict(psnr=25.,ssim=.8,lpips=.2,loss=1.)
                gain=dict(psnr=25.2,ssim=.81,lpips=.199,loss=.9)
                trace=[dict(changed_from_OLD=True)]*30
                records.append(dict(case=dict(image_id=str(i),sigma=sigma),
                    metrics=dict(C=baseline,OLD=baseline,REG=baseline,TIME=baseline,TIME_NOISE=baseline,
                                 UP_REG=gain,DOWN_REG=baseline),
                    traces=dict(UP_REG=trace,DOWN_REG=trace),arm_seconds=dict(UP_REG=1.,DOWN_REG=1.)))
        self.assertEqual(interpretation(records)['across_condition_candidates'],['UP_REG'])
        for r in records:
            if r['case']['sigma']==.1:
                r['metrics']['UP_REG'],r['metrics']['DOWN_REG']=r['metrics']['DOWN_REG'],r['metrics']['UP_REG']
        self.assertEqual(interpretation(records)['across_condition_candidates'],[])


if __name__=='__main__':unittest.main()
