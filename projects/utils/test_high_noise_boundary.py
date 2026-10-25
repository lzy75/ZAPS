import json
from pathlib import Path
import tempfile
import unittest
from utils import high_noise_boundary as rules
from utils.diag_a46_cached_audit import opportunity,agreement


def cases_choices():
    cases=[dict(key=f'{cohort}_{task}_{image}',cohort=cohort,task=task,image_id=image)
        for cohort,images in (('calibration',rules.CALIBRATION),('validation',rules.VALIDATION))
        for image in images for task in rules.old.WEIGHTS]
    choices={}
    for i,c in enumerate(cases):
        p=float(i+2);s=p+1.;selected=p if rules.old.WEIGHTS[c['task']]==0. else s if rules.old.WEIGHTS[c['task']]==1. else p+.5
        choices[c['key']]=dict(PHYSICAL=p,ISO=p-.1,STATE=s,SELECTED=selected)
    return cases,choices


class BoundaryTests(unittest.TestCase):
    def test_noise_reference_uses_only_calibration_and_shuffle_only_validation(self):
        cases,choices=cases_choices();fixed=rules.old.MATCHED
        assignment,matched=rules.allocation(cases,choices,fixed)
        self.assertEqual(len(assignment),12)
        valkeys={c['key'] for c in cases if c['cohort']=='validation'};self.assertEqual(set(assignment),valkeys)
        for key,row in assignment.items():
            self.assertIn(row['donor'],valkeys);self.assertNotEqual(key,row['donor'])
        modified={k:dict(v) for k,v in choices.items()}
        for c in cases:
            if c['cohort']=='validation':modified[c['key']]={k:g+10 for k,g in modified[c['key']].items()}
        self.assertEqual(rules.allocation(cases,modified,fixed)[1],matched)
        self.assertEqual(sum(rules.old.cap('BASELINE',c['task'])+rules.old.cap('choose',c['task'])+
            (rules.old.cap('finish',c['task']) if c['cohort']=='validation' else 0) for c in cases),8865)
        with self.assertRaises(TypeError):rules.allocation(cases,choices,fixed,ground_truth={})

    def test_noise_quality_and_matching_are_separate(self):
        cases,_=cases_choices();records=[]
        for c in cases:
            if c['cohort']!='validation':continue
            ev={n:dict(delta=dict(psnr=.2,ssim=.01,lpips=-.01,loss=-.02)) for n in rules.old.POLICIES}
            ev['C']['delta']=dict(psnr=0.,ssim=0.,lpips=0.,loss=0.)
            records.append(dict(case=c,gamma={},evaluation=ev))
        groups=rules.summarize(records)
        self.assertTrue(all(g['quality_candidate'] for g in groups));self.assertFalse(any(g['state_matching_candidate'] for g in groups))
        json.dumps(groups,allow_nan=False)

    def test_finite_library_oracle_keeps_bad_quality_actions_and_excludes_them_from_guarded_choice(self):
        arms=dict(C=dict(psnr=0.,ssim=0.,lpips=0.),MATCHED_FIXED=dict(psnr=1.,ssim=.01,lpips=-.01),
            GOOD=dict(psnr=1.1,ssim=.012,lpips=-.012),BAD=dict(psnr=2.,ssim=-.2,lpips=.5))
        result=opportunity([dict(task='toy',per_image={'a':arms})])[0]
        self.assertAlmostEqual(result['mean_raw_headroom'],1.)
        self.assertAlmostEqual(result['mean_guarded_headroom'],.1)
        self.assertEqual(result['rows'][0]['guarded_oracle_arm'],'GOOD')

    def test_local_model_agreement_is_not_final_quality_claim(self):
        record=dict(choices={'SELECTED':dict(gamma=3.,fit=dict(usable=True,fit_relative_error=0.,coefficients=[1.,-6.,9.]))},
            query_readouts={'1.0':dict(loss=4.,risks={}), '3.0':dict(loss=2.,risks={})})
        out=agreement(record,0.)
        self.assertAlmostEqual(out['actual_over_predicted'],.5)
        self.assertNotIn('final_psnr',out)

    def test_real_torch_paired_measurement_scales_sigma_without_redrawing_noise(self):
        try:import torch
        except ImportError:self.skipTest('Torch absent on Mac; mandatory without skips on server')
        from utils.diag_high_noise_boundary import a40
        import modules.main_single  # Complete lazy dependency imports before measuring RNG invariance.
        case=dict(dataset='ffhq',task='gaussian_deblur',image_id='toy',image_sha256='abc',measurement_seed=123,sigma=.05)
        mean=torch.full((1,3,256,256),.2);standard=torch.ones_like(mean)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'measurements/ffhq/gaussian_deblur/toy.pt';path.parent.mkdir(parents=True)
            torch.save(dict(mean=mean,standard_noise=standard,image_sha256='abc',seed=123),path)
            before=torch.get_rng_state().clone()
            low,meta0=a40.measurement(root,case,None,'cpu')
            high,meta1=a40.measurement(root,dict(case,sigma=.10),None,'cpu')
            self.assertTrue(torch.allclose(high-low,.05*standard,atol=1e-7,rtol=1e-6))
            self.assertEqual(meta0,meta1);self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
