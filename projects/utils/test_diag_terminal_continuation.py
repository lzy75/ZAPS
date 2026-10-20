import unittest
from types import SimpleNamespace
from unittest.mock import patch
from utils import diag_terminal_continuation as d
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def fixture(self):
        records=[]
        for image in ('00020','00021','00022'):
            for seed in (1000,2000):
                c=dict(loss=1.,psnr=25.,ssim=.8,lpips=.2);f=dict(loss=.99,psnr=25.11,ssim=.801,lpips=.198)
                floors={k:0. for k in c};effects={a+'_C':d.parent.history.effect(f,c,floors) for a in ('F0','F1')}
                records.append(dict(status='complete',case=dict(key=f'{image}_s{seed}',image_id=image),method_nfe=0,diagnostic_nfe=4,
                    branches=dict(C=c,F0=f,F1=f),effects=effects,bridge_candidates=dict(F0=True,F1=True),both_bridges_candidate=True,gates={'test':{'passed':True}}))
        return records

    def test_h6_requires_two_images_both_seeds_and_both_bridges(self):
        r=self.fixture();s=d.summarize(r)
        self.assertEqual(s['classification'],'FINAL_FIXED_REFINEMENT_CANDIDATE_ONLY');self.assertEqual(s['diagnostic_nfe'],24)
        for case in r:
            if case['case']['key'].endswith('s2000'):case['both_bridges_candidate']=False;case['bridge_candidates']['F1']=False
        self.assertEqual(d.summarize(r)['classification'],'FINAL_FIXED_REFINEMENT_NOT_SUPPORTED')
        self.assertEqual(d.summarize(r[:-1])['classification'],'INCOMPLETE_PILOT')

    def test_failed_and_running_attempt_costs_are_not_lost(self):
        records=self.fixture()[:1]+[dict(status='failed',method_nfe=0,diagnostic_nfe=3),dict(status='running',method_nfe=0,diagnostic_nfe=1)]
        s=d.summarize(records);self.assertEqual(s['diagnostic_nfe'],8);self.assertTrue(s['cost_is_lower_bound_if_abruptly_interrupted'])

    def test_only_saved_t14_interval_selected_no_quality_ranking(self):
        artifact={'points':[dict(source_step=k,spec=dict(t=14,m=3,s=0),F0=object(),F1=object()) for k in (8,19,24,27,28)]}
        self.assertIs(d.select_inputs(artifact)['F0'],artifact['points'][4]['F0'])
        artifact['points'][4]['spec']['t']=54
        with self.assertRaises(ValueError):d.select_inputs(artifact)


@unittest.skipIf(torch is None,'Requires server Torch; zero UNet calls')
class TerminalTests(unittest.TestCase):
    def test_actual_terminal_core_applies_q_and_never_draws_noise(self):
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.tensor([.9999])))
        x=torch.zeros(1,3,4,4);clean=torch.full_like(x,.25);q=torch.full_like(x,.02)
        with patch('torch.randn_like',side_effect=AssertionError('Terminal must not draw')):
            actual=d.terminal_output(z,x,dict(clean=clean,q=q))
        torch.testing.assert_close(actual,clean+q,atol=0,rtol=0)
        self.assertFalse(torch.equal(actual,clean))


if __name__=='__main__':unittest.main()
