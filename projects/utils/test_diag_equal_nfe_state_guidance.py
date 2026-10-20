import copy
import unittest
from utils import diag_equal_nfe_state_guidance as d


def records(headroom=True):
    rows=[]
    for cohort,first in (('development',20),('transfer',23)):
        for image in range(first,first+3):
            for seed in (1000,2000):
                c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
                zero=dict(loss=1.1,psnr=19.9,ssim=.699,lpips=.2)
                two=dict(loss=.9,psnr=20.1,ssim=.701,lpips=.199)
                if headroom and image==first+2:two=dict(two,psnr=20.05,lpips=.25)
                rows.append(dict(case=dict(key=f'{image:05d}_s{seed}',image_id=f'{image:05d}',sampling_seed=seed),cohort=cohort,
                    branches={'0':zero,'1':c,'2':two},features=dict(log_R=float(image-first),log_a=seed/1000,turn=(image-first)*.1,log1p_omega=.2),floors=dict.fromkeys(d.METRICS,0.)))
    return rows


class PolicyTests(unittest.TestCase):
    def test_nominal_tie_and_predicted_guards(self):
        p={str(g):dict.fromkeys(d.METRICS,0.) for g in d.GAMMAS}
        self.assertEqual(d.choose(p),1)
        p['2']['psnr']=.2;self.assertEqual(d.choose(p),2)
        p['2']['lpips']=.006;self.assertEqual(d.choose(p),1)
        p['2']['lpips']=0.;p['2']['loss']=.001;self.assertEqual(d.choose(p),1)

    def test_transfer_labels_forbidden_and_only_train_normalizes(self):
        rows=records();train=[r for r in rows if r['cohort']=='development'];test=[r['features'] for r in rows if r['cohort']=='transfer']
        policy=d.policies(train,test)
        self.assertEqual(len(policy['state']['actions']),6)
        self.assertEqual(policy['state']['models']['2']['psnr']['x_mean'][0],1.)
        with self.assertRaises(ValueError):d.policies(train,[dict(test[0],branches={})])
        changed=copy.deepcopy(test);changed[0]['log_R']=1000.
        self.assertEqual(d.policies(train,changed)['state']['models'],policy['state']['models'])

    def test_equal_cost_state_requires_actual_increment_not_extra_call(self):
        rows=records();policy=dict(calibration_fixed_gamma=1,physical=dict(actions=[1]*6),state=dict(actions=[2,2,2,2,1,1]))
        out=d.analyze(rows,policy)
        self.assertEqual(out['classification'],'EXPLORATORY_STATE_INCREMENT_AT_EQUAL_NFE')
        self.assertTrue(all(v==300 for v in out['deployment_nfe_per_case'].values()))
        self.assertEqual((out['method_nfe'],out['diagnostic_nfe']),(0,0))
        self.assertEqual(len(out['image_permutation_results']),6)
        self.assertEqual(out['strongest_fixed_gamma_on_transfer_for_evaluation_only'],1)

    def test_strong_fixed_oracle_closes_false_state_claim(self):
        rows=records(headroom=False);policy=dict(calibration_fixed_gamma=2,physical=dict(actions=[2]*6),state=dict(actions=[2]*6))
        out=d.analyze(rows,policy)
        self.assertEqual(out['classification'],'NO_USEFUL_STATE_SELECTION_HEADROOM_IN_THIS_ACTION_SET')
        self.assertFalse(out['state_actions_vary'])
        self.assertAlmostEqual(out['oracle_psnr_headroom_over_strongest_fixed'],0.)

    def test_omitting_direction_gain_or_bad_quality_prevents_support(self):
        rows=records();actions=[2,2,2,2,1,1]
        policy=dict(calibration_fixed_gamma=1,physical=dict(actions=actions),state=dict(actions=actions))
        self.assertEqual(d.analyze(rows,policy)['classification'],'STATE_INCREMENT_NOT_SUPPORTED_AT_EQUAL_NFE')
        policy['physical']['actions']=[1]*6;policy['state']['actions']=[2]*6
        self.assertFalse(d.analyze(rows,policy)['policy_evaluation']['state']['quality_guard'])


if __name__=='__main__':unittest.main()
