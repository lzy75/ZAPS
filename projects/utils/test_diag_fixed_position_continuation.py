import unittest
from types import SimpleNamespace
from unittest.mock import patch
from utils import diag_fixed_position_continuation as d
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def test_position_requires_all_four_bridge_comparisons(self):
        base=dict(loss=1.,psnr=25.,ssim=.8,lpips=.2);floors={k:0. for k in base}
        reference=dict(F0=base,F1=base)
        good=dict(loss=.9,psnr=25.2,ssim=.81,lpips=.19)
        self.assertEqual(d.position_comparison(dict(F0=good,F1=good),reference,floors,floors)['readout'],'T54_DOMINATES_BOTH_T14_BRIDGES')
        # A single excellent bridge must not mask a worse other bridge.
        bad=dict(loss=1.1,psnr=24.9,ssim=.79,lpips=.21)
        mixed=d.position_comparison(dict(F0=good,F1=bad),reference,floors,floors)
        self.assertEqual(len(mixed['T54_minus_T14']),4)
        self.assertEqual(mixed['readout'],'TRADEOFF_OR_UNRESOLVED')
        self.assertEqual(d.position_comparison(dict(F0=bad,F1=bad),reference,floors,floors)['readout'],'T14_DOMINATES_BOTH_T54_BRIDGES')

    def test_combined_numerical_floors_preserved_and_equality_not_a_win(self):
        b=dict(loss=1.,psnr=25.,ssim=.8,lpips=.2);floors={k:0. for k in b};higher=dict(floors,psnr=.02)
        result=d.position_comparison(dict(F0=b,F1=b),dict(F0=b,F1=b),floors,higher)
        self.assertEqual(result['floors']['psnr'],.02)
        self.assertEqual(result['readout'],'TRADEOFF_OR_UNRESOLVED')

    def test_selects_t54_saved_point_without_ranking(self):
        points=[dict(source_step=k,spec=dict(t=0,m=0,s=0),F0=object(),F1=object()) for k in (8,19,24,27,28)]
        points[3]['spec']=dict(t=54,m=28,s=14);points[4]['spec']=dict(t=14,m=3,s=0)
        self.assertIs(d.select_inputs({'points':points})['F0'],points[3]['F0'])
        points[4]['spec']['s']=1
        with self.assertRaises(ValueError):d.select_inputs({'points':points})

    def test_summary_cluster_decision_and_failed_cost(self):
        records=[]
        base=dict(loss=1.,psnr=25.,ssim=.8,lpips=.2);good=dict(loss=.9,psnr=25.2,ssim=.81,lpips=.19)
        floors={k:0. for k in base}
        for image in ('00020','00021','00022'):
            for seed in (1000,2000):
                branches=dict(C=base,F0=good,F1=good)
                records.append(dict(status='complete',case=dict(key=f'{image}_s{seed}',image_id=image),method_nfe=0,diagnostic_nfe=6,
                    branches=branches,effects={a+'_C':d.parent.history.effect(good,base,floors) for a in ('F0','F1')},
                    bridge_candidates=dict(F0=True,F1=True),both_bridges_candidate=True,
                    position_comparison=d.position_comparison(branches,dict(F0=base,F1=base),floors,floors),gates={'test':{'passed':True}}))
        s=d.summarize(records)
        self.assertEqual(s['diagnostic_nfe'],36);self.assertEqual(s['classification'],'FINAL_T54_REFINEMENT_CANDIDATE_ONLY')
        self.assertEqual(s['position_images_both_seeds']['T54_DOMINATES_BOTH_T14_BRIDGES'],3)
        self.assertEqual(d.summarize(records[:-1]+[dict(status='failed',method_nfe=0,diagnostic_nfe=3)])['diagnostic_nfe'],33)
        for r in records:
            if r['case']['key'].endswith('s2000'):r['both_bridges_candidate']=False
        self.assertEqual(d.summarize(records)['classification'],'FINAL_T54_REFINEMENT_NOT_SUPPORTED')


@unittest.skipIf(torch is None,'Requires server Torch; no UNet loaded')
class SuffixTests(unittest.TestCase):
    def test_actual_suffix_uses_native_indices_original_noise_and_terminal_q(self):
        from modules.diffusion_model import _precompute_diffusion_params
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod']))
        x=torch.full((1,3,4,4),.4);noise=torch.full_like(x,.3);calls=[];contexts=[]
        def evaluate(z,y,state,t,index):
            calls.append((t,index))
            return dict(clean=state*.5,q=torch.full_like(state,.02 if t==14 else .03))
        before=torch.get_rng_state().clone()
        with patch('torch.randn_like',side_effect=AssertionError('Unexpected draw')):
            actual,first,last=d.continue_suffix(z,None,x,noise,evaluate,lambda t,i:contexts.append((t,i)))
        expected_zero=d.parent.posterior(x,x*.5,14,0,z.dm.alphas_cumprod,noise)+.02
        torch.testing.assert_close(actual,expected_zero*.5+.03,atol=1e-7,rtol=1e-7)
        self.assertEqual(calls,[(14,1),(0,0)]);self.assertEqual(contexts,calls)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
