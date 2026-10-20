import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import diag_terminal_state_strength as d

try:
    import torch
except ImportError:
    torch = None


class StrengthTests(unittest.TestCase):
    def test_affine_operator_requires_offset_cancellation(self):
        # H(x)=2x+3: H(q) would produce the wrong step even for this scalar problem.
        h = lambda x: 2*x+3
        clean, q, y = np.array([1.]), np.array([.5]), np.array([6.5])
        a = h(clean+q)-h(clean)
        s = d.strength(np.array([1.]),q,y-h(clean),a)
        self.assertEqual(s['alpha_L'],1.5)
        self.assertEqual(s['alpha'],1.5)
        self.assertAlmostEqual(float((h(clean+s['alpha']*q)-y)[0]),0.)
        self.assertNotAlmostEqual(float((a@ (y-h(clean)))/(h(q)@h(q))),1.5)

    def test_conflict_cap_preserves_half_prior_progress(self):
        p = np.array([1.,0.]); q = np.array([-1.,1.])
        s = d.strength(p,q,np.array([2.]),np.array([1.]))
        self.assertTrue(s['guard_active'])
        self.assertEqual(s['alpha'],.5)
        self.assertAlmostEqual(float(p@(p+s['alpha']*q)),.5*float(p@p))
        self.assertAlmostEqual(s['prior_q_cosine'],-2**-.5)

    def test_zero_directions_and_overshoot_are_explicit(self):
        s = d.strength(np.zeros(2),np.zeros(2),np.ones(1),np.zeros(1))
        self.assertFalse(s['action_resolved']); self.assertIsNone(s['prior_q_cosine'])
        self.assertEqual(s['alpha'],0.)
        s = d.strength([0.],[1.],[20.],[1.])
        self.assertEqual(s['alpha'],2.)
        self.assertIsNone(s['q_over_prior_norm'])
        s = d.strength([1.],[1.],[-1.],[1.])
        self.assertEqual(s['alpha'],0.)

    def test_nonfinite_or_incompatible_shapes_fail(self):
        with self.assertRaises(ValueError):
            d.strength([1.],[float('nan')],[1.],[1.])
        with self.assertRaises(ValueError):
            d.strength([1.],[1.,2.],[1.],[1.])


def records():
    result = []
    for image in ('00023','00024','00025'):
        for seed in (1000,2000):
            c = dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
            fixed = dict(loss=.9,psnr=20.1,ssim=.699,lpips=.204)
            adaptive = dict(loss=.8,psnr=20.2,ssim=.699,lpips=.204)
            branches = dict(C=c, **{a:dict(fixed) for a in (*d.FIXED,'S_native','L')}, A=adaptive)
            result.append(dict(status='complete', case=dict(key=f'{image}_s{seed}',image_id=image),
                method_nfe=0,diagnostic_nfe=2,branches=branches,floors=dict.fromkeys(d.METRICS,0.),
                state=dict(guard_active=True),seconds=1.,gates=dict(test=dict(passed=True))))
    return result


class DecisionTests(unittest.TestCase):
    def test_user_tolerances_allow_small_degradation_not_prediction_MAE(self):
        r = records(); s = d.summarize(r)
        self.assertEqual(s['classification'],'STATE_STRENGTH_EXPLORATORY_CANDIDATE')
        self.assertTrue(s['direction_guard_increment_candidate'])
        self.assertEqual(s['diagnostic_nfe'],12)
        self.assertEqual(s['cumulative_queue_cost']['total'],5876)
        for x in r:
            x['branches']['A']['lpips'] = .2051
        self.assertFalse(d.summarize(r)['comparisons']['A_C']['acceptable_vs_C'])

    def test_fixed_strong_control_and_ablation_cannot_be_ignored(self):
        r = records()
        for x in r:
            x['branches']['F2']['psnr'] = 20.3
        self.assertEqual(d.summarize(r)['classification'],'FIXED_STRENGTH_CANDIDATE_ONLY')
        r = records()
        for x in r:
            x['branches']['L'] = copy.deepcopy(x['branches']['A'])
        s = d.summarize(r)
        self.assertEqual(s['classification'],'STATE_STRENGTH_EXPLORATORY_CANDIDATE')
        self.assertFalse(s['direction_guard_increment_candidate'])

    def test_psnr_numerical_floor_physics_and_image_clusters(self):
        r = records()
        for x in r:
            x['branches']['A']['loss'] = 1.1
        self.assertFalse(d.summarize(r)['comparisons']['A_C']['acceptable_vs_C'])
        r = records()
        for x in r:
            x['branches']['A']['psnr'] = 20.+1e-7
        self.assertFalse(d.summarize(r)['comparisons']['A_C']['psnr_positive_resolved'])
        r = records()
        for x in r:
            x['branches']['A']['psnr'] = 21. if x['case']['image_id'] == '00023' else 20.09
        self.assertFalse(d.summarize(r)['comparisons']['A_F1']['pair_advantage'])
        self.assertEqual(d.summarize(r[:5])['classification'],'INCOMPLETE_PILOT')
        failed = dict(status='failed',method_nfe=0,diagnostic_nfe=1)
        self.assertEqual(d.summarize(r[:5]+[failed])['diagnostic_nfe'],11)


@unittest.skipIf(torch is None,'Server Torch required; no UNet loaded')
class CoreTests(unittest.TestCase):
    def test_actual_terminal_core_and_repeats_share_original_input(self):
        z = SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.tensor([.9999])))
        c = torch.full((1,3,4,4),.4); seen = []
        def evaluate(z,y,x,t,index):
            self.assertEqual((t,index),(0,0)); seen.append(x.clone())
            return dict(clean=x*.5,q=torch.full_like(x,.02))
        fields,outputs = d.parent.polish_pair(z,None,c,evaluate)
        for x in seen:
            torch.testing.assert_close(x,c,atol=0,rtol=0)
        for out in outputs:
            torch.testing.assert_close(out,fields[0]['clean']+fields[0]['q'],atol=0,rtol=0)
        self.assertTrue(torch.equal(outputs[0],outputs[1]))


if __name__ == '__main__':
    unittest.main()
