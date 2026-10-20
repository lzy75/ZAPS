"""A24 controls: remove amplitude confounding, preserve clocks and label conditional cost."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import diag_state_feedback_diagnosis as d
from utils import fixed_count_state_positions as p
try:
    import torch
except ImportError:
    torch=None

GRID=[999,989,978,967,956,944,931,917,903,888,872,854,835,815,793,769,742,712,679,641,597,546,484,409,319,218,124,54,14,0]


class DiagnosisTests(unittest.TestCase):
    def test_matched_control_keeps_missing_direction_fallback(self):
        self.assertEqual(d.matched(dict(e_R=.8,e_C=None)),.8)
        self.assertEqual(d.matched(dict(e_R=.8,e_C=0)),.4)
        self.assertEqual(d.matched(dict(e_R=.8,e_C=-1)),.4)
        self.assertEqual(p.combined('RC',dict(e_R=.8,e_C=0)),d.matched(dict(e_R=.8,e_C=0)))

    def test_shared_clock_uses_only_positions_and_has_30_ordered_points(self):
        positions=p.Positions(np.linspace(1,0,1000),GRID)
        records=[]
        for signal in (-1,-.6,-.2,.2,.6,1):
            times=[999]+[positions.choose(j,signal) for j in range(1,30)]
            records.append(dict(traces={'RC':[dict(t=t) for t in times]},branches=dict(psnr=123.)))
        clock=d.shared_grid(records,positions)
        self.assertEqual(len(clock),30);self.assertEqual((clock[0],clock[-1]),(999,0));self.assertTrue(all(t>s for t,s in zip(clock,clock[1:])))
        changed=copy.deepcopy(records)
        for r in changed:r['branches']['psnr']=-10000
        self.assertEqual(d.shared_grid(changed,positions),clock)
        with self.assertRaises(ValueError):d.shared_grid(records[:5],positions)

    def test_conditional_replay_and_pairwise_hint_never_become_state_success(self):
        rr=[]
        for image in ('0','1','2'):
            for seed in (1000,2000):
                c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
                branches=dict(C=c,RC=dict(c,loss=1.01,psnr=19.98),R_MATCHED=dict(c,loss=1.02,psnr=19.95),FIXED_GRID=c,SLOT_REPLAY=dict(c,loss=.99,psnr=20.1))
                rr.append(dict(status='complete',case=dict(image_id=image,key=image+str(seed)),branches=branches,diagnostic_nfe=120,floors=dict.fromkeys(d.METRICS,0.),gates={'x':{'passed':True}},
                    source_parameter_changes=[dict(zeta_relative_difference=.01,D_relative_difference=.02)]))
        result=d.summarize(rr)
        self.assertEqual(result['classification'],'MECHANISM_DIAGNOSIS_ONLY');self.assertFalse(result['source_RC_net_benefit_supported'])
        self.assertTrue(result['comparisons']['direction_after_residual_amplitude_match']['psnr_mechanism_hint'])
        self.assertEqual(result['diagnostic_nfe'],720)
        self.assertNotIn('SLOT_REPLAY',result['deployable_nfe_per_case']);self.assertIn('330',result['SLOT_REPLAY'])

    def test_failed_cost_is_retained_without_complete_mechanism_claim(self):
        out=d.summarize([dict(status='failed',diagnostic_nfe=9)])
        self.assertEqual(out['classification'],'INCOMPLETE_PILOT');self.assertEqual(out['diagnostic_nfe'],9)
        self.assertNotIn('comparisons',out)


@unittest.skipIf(torch is None,'Server Torch required; no UNet checkpoint loaded')
class TensorTests(unittest.TestCase):
    def test_parameter_control_keeps_same_input_time_prediction_and_clean(self):
        from modules.diffusion_model import _precompute_diffusion_params
        alpha=_precompute_diffusion_params()['alphas_cumprod'];positions=p.Positions(p.noise_u(alpha.numpy()),GRID)
        calls=[]
        def eps(x,t):calls.append((x.clone(),t.clone()));return torch.zeros_like(x)
        identity=SimpleNamespace(H=lambda x:x,transpose=lambda x:x,analysis=lambda x:x,synthesis=lambda x:x)
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=alpha,_predict_eps=eps),A=identity,dwt=identity,
            zeta=torch.linspace(.01,.5,30),D=torch.linspace(.1,.8,30).view(30,1,1,1).expand(30,3,4,4))
        x=torch.full((1,3,4,4),.01);y=torch.ones_like(x)
        for k,t in ((8,GRID[8]),(8,positions.choose(8,1))):
            native=d.source.field(z,y,x,t,positions);slot=d.source.field(z,y,x,t,SimpleNamespace(weights=lambda ignored:(29-k,29-k,0.)))
            torch.testing.assert_close(native['clean'],slot['clean'],atol=0,rtol=0)
            torch.testing.assert_close(calls[-1][0],calls[-2][0],atol=0,rtol=0);torch.testing.assert_close(calls[-1][1],calls[-2][1],atol=0,rtol=0)
            if t==GRID[k]:torch.testing.assert_close(native['q'],slot['q'],atol=0,rtol=0)
            else:self.assertFalse(torch.equal(native['q'],slot['q']))


if __name__=='__main__':unittest.main()
