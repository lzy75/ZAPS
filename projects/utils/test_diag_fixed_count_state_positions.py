"""A23 causal scalar decisions, unlabeled references and fixed-count parameter contracts."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import fixed_count_state_positions as d
from utils import diag_fixed_count_state_positions as run
try:
    import torch
except ImportError:
    torch=None

GRID=[999,989,978,967,956,944,931,917,903,888,872,854,835,815,793,769,742,712,679,641,597,546,484,409,319,218,124,54,14,0]


def data():
    records=[]
    for i in range(3):
        for seed in (1000,2000):
            records.append(dict(case=dict(image_id=str(i),sampling_seed=seed),rows=[dict(noise_u=k/29,residual_noise_ratio=2.+i,
                guidance_variation3_per_u=None if k<2 else 3.+i,psnr=999.) for k in range(30)]))
    return records


class PolicyTests(unittest.TestCase):
    def test_reference_uses_no_quality_labels_and_excludes_entire_image(self):
        rr=data();ref=d.reference(rr,exclude_image='0')
        self.assertEqual(ref['reference_images'],['1','2']);self.assertFalse(ref['GT_used'])
        changed=copy.deepcopy(rr)
        for r in changed:
            for row in r['rows']:
                row['psnr']=-1000
                if r['case']['image_id']=='0':row['residual_noise_ratio']=1e9
        self.assertEqual(d.reference(changed,exclude_image='0'),ref)
        self.assertAlmostEqual(ref['R']['log_values'][0],.5*(np.log(3)+np.log(4)))

    def test_signal_is_stage_relative_and_missing_direction_is_not_fabricated(self):
        ref=dict(R=dict(u=[0,1],log_values=[np.log(4),np.log(2)]),Omega=dict(u=[0,1],log_values=[np.log(8+d.EPS),np.log(4+d.EPS)]))
        s=d.scores(4,None,0,ref);self.assertAlmostEqual(s['e_R'],0);self.assertIsNone(s['e_C'])
        self.assertEqual(d.combined('D',s),0);self.assertEqual(d.combined('RC',dict(e_R=.4,e_C=None)),.4)
        s=d.scores(8,16,0,ref);self.assertAlmostEqual(s['e_R'],1);self.assertAlmostEqual(s['e_C'],1)
        self.assertAlmostEqual(d.combined('RC',dict(e_R=1,e_C=-1)),0)
        with self.assertRaises(ValueError):d.scores(1,float('nan'),0,ref)

    def test_fixed_count_endpoint_and_order_for_extreme_causal_responses(self):
        p=d.Positions(np.linspace(1,0,1000),GRID)
        for signals in ([1]*29,[-1]*29,[(-1)**k for k in range(29)]):
            times=[999]+[p.choose(k+1,v) for k,v in enumerate(signals)]
            self.assertEqual(len(times),30);self.assertEqual(times[-1],0)
            self.assertTrue(all(t>s for t,s in zip(times,times[1:])))
        self.assertEqual([999]+[p.choose(j,0) for j in range(1,30)],GRID)
        self.assertGreater(p.choose(1,1),GRID[1]);self.assertLess(p.choose(1,-1),GRID[1])
        self.assertEqual(p.choose(8,.9,rho=0),GRID[8])

    def test_anchor_mapping_and_interpolation_are_valid_off_original_grid(self):
        p=d.Positions(np.linspace(1,0,1000),GRID)
        for k,t in enumerate(GRID):self.assertEqual(p.weights(t),(29-k,29-k,0.))
        for t in range(1000):
            i,j,w=p.weights(t);self.assertTrue(0<=w<=1);self.assertTrue(29>=i>=j>=0)
            if i!=j:self.assertEqual(i-j,1)

    def test_incomplete_summary_retains_cost_without_method_claim(self):
        out=run.summary([dict(status='failed',diagnostic_nfe=13)])
        self.assertEqual(out['classification'],'INCOMPLETE_PILOT');self.assertEqual(out['diagnostic_nfe'],13)
        self.assertEqual(out['deployment_nfe_per_case'],300);self.assertNotIn('relative_to_C',out)

    def test_fixed_gain_without_direction_increment_is_not_state_support(self):
        records=[]
        for image in ('0','1','2'):
            for seed in (1000,2000):
                c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
                b={a:dict(c,loss=.9,psnr=20.2) for a in ('R','D','RC')};b['C']=c
                records.append(dict(status='complete',case=dict(image_id=image,key=image+str(seed)),diagnostic_nfe=120,branches=b,
                    floors=dict.fromkeys(run.parent.METRICS,0.),a22_fixed=b['R'],gates={'test':{'passed':True}},traces={a:[dict(t=999,nominal_t=999)] for a in d.ARMS}))
        self.assertEqual(run.summary(records)['classification'],'CURRENT_POSITION_FEEDBACK_NOT_SUPPORTED')
        for r in records:r['branches']['RC']=dict(r['branches']['RC'],psnr=20.25)
        self.assertEqual(run.summary(records)['classification'],'INITIAL_POSITION_FEEDBACK_CANDIDATE_ONLY')
        for r in records:r['branches']['RC']['lpips']=.21
        self.assertEqual(run.summary(records)['classification'],'CURRENT_POSITION_FEEDBACK_NOT_SUPPORTED')


@unittest.skipIf(torch is None,'Server Torch required; no UNet checkpoint loaded')
class TensorTests(unittest.TestCase):
    def test_noise_anchor_parameters_are_exact_and_interpolation_stays_between_neighbors(self):
        p=d.Positions(np.linspace(1,0,1000),GRID)
        z=SimpleNamespace(zeta=torch.arange(30,dtype=torch.float32),D=torch.arange(30,dtype=torch.float32).view(30,1,1,1).expand(30,3,4,4))
        for k,t in enumerate(GRID):
            a,b,weights=d.parameters(z,t,p)
            torch.testing.assert_close(a,z.zeta[29-k],atol=0,rtol=0);torch.testing.assert_close(b,z.D[29-k],atol=0,rtol=0)
            self.assertEqual(a.data_ptr(),z.zeta[29-k].data_ptr());self.assertEqual(weights['right_weight'],0.)
        t=p.choose(10,1);a,b,w=d.parameters(z,t,p);i,j,v=p.weights(t)
        torch.testing.assert_close(a,(1-v)*z.zeta[i]+v*z.zeta[j]);self.assertTrue(float(z.zeta[j])<=float(a)<=float(z.zeta[i]))


if __name__=='__main__':unittest.main()
