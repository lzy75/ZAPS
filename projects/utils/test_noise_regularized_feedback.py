import unittest
from types import SimpleNamespace
from utils import noise_regularized_feedback as rules
from utils.continuous_response_feedback import select as original


class RegularizedTests(unittest.TestCase):
    def test_zero_penalty_is_exact_old_policy_and_missing_response_keeps_original(self):
        for dot,r2,a2 in ((.31,.31**2,1.),(10.,100.,1.),(-1.,1.,1.),(0.,0.,1.),(0.,1.,0.)):
            expected=original(dot,r2,a2,2.)
            result=rules.select(dot,r2,a2,1,.05,strength=0.)
            self.assertEqual(expected,{k:result[k] for k in expected})
        self.assertEqual(rules.select(0.,1.,0.,1,.05)['gamma'],1.)

    def test_convex_quadratic_solution_and_noise_pull_toward_one(self):
        import numpy as np
        dot,r2,a2,count,sigma=.6,1.,2.,3,.5
        result=rules.select(dot,r2,a2,count,sigma)
        expected=(dot+count*sigma**2)/(a2+count*sigma**2)
        self.assertAlmostEqual(result['gamma'],expected)
        grid=np.linspace(0,2,10001)
        losses=.5*(r2-2*grid*dot+grid**2*a2+count*sigma**2*(grid-1)**2)
        objective=.5*(r2-2*result['gamma']*dot+result['gamma']**2*a2+count*sigma**2*(result['gamma']-1)**2)
        self.assertLessEqual(objective,float(losses.min())+1e-12)
        for signed_dot in (.2,1.8):
            values=[rules.select(signed_dot,4.,1.,1,s)['raw_gamma'] for s in (.025,.05,.1,1.,10.)]
            self.assertTrue(all(abs(b-1)<=abs(a-1) for a,b in zip(values,values[1:])))
        self.assertEqual(sum(rules.calls(s)*6 for s in (.025,.05,.1)),1440)

    def test_old_reference_table_ignores_GT_and_is_noise_dependent(self):
        grid=list(range(30))
        records=[dict(case=dict(image_id=f'{i:05d}',sampling_seed=s),
                      rows=[dict(t=t,R=2.,cosine=.9,relative_response=.1) for t in reversed(grid)],
                      metrics={'psnr':-100.}) for i in range(20,26) for s in (1000,2000)]
        first=rules.reference_profiles(records,grid,(.025,.05,.1))
        for r in records:r['metrics']['psnr']=10000.
        self.assertEqual(first,rules.reference_profiles(records,grid,(.025,.05,.1)))
        self.assertNotEqual(first['values']['0.025'][0]['gamma'],first['values']['0.1'][0]['gamma'])
        with self.assertRaises(ValueError):rules.reference_profiles(records[:-1],grid,(.05,))

    def test_quality_gain_does_not_imply_state_information(self):
        from utils.diag_noise_regularized_feedback import interpretation
        records=[]
        for i in range(6):
            for s in (.025,.05,.1):
                base=dict(psnr=25.,ssim=.8,lpips=.2,loss=1.)
                better=dict(psnr=25.3,ssim=.81,lpips=.199,loss=.9)
                trace=[dict(gamma=1.2,trust_weight=.5,R=1.1)]*30
                records.append(dict(case=dict(image_id=str(i),sigma=s),
                    metrics=dict(C=base,OLD=base,TIME=base,REG=better,BLIND=better,TIME_NOISE=better),
                    traces=dict(REG=trace,TIME_NOISE=trace)))
        result=interpretation(records)
        self.assertTrue(result['quality_candidate']);self.assertEqual(result['state_increment_candidate_sigmas'],[])
        for r in records:
            if r['case']['sigma']==.1:r['metrics']['REG']=dict(r['metrics']['REG'],psnr=25.33)
        self.assertEqual(interpretation(records)['state_increment_candidate_sigmas'],[.1])

    def test_torch_sensor_matches_formula_and_native_alias(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; mandatory on server')
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.ones(1,dtype=torch.float64)),
                          A=SimpleNamespace(H=lambda x:2*x))
        q=torch.tensor([1.,2.],dtype=torch.float64);clean=torch.zeros_like(q);y=.31*2*q
        actual=rules.sensor(z,y,clean,q,0,.05)
        self.assertAlmostEqual(actual['raw_gamma'],(6.2+2*.05**2)/(20.+2*.05**2))
        self.assertEqual(actual,rules.sensor(z,y,clean,q,0,.05,.05))
        self.assertAlmostEqual(actual['raw_gamma'],1.+actual['trust_weight']*(actual['gamma_LS']-1.))


if __name__=='__main__':unittest.main()
