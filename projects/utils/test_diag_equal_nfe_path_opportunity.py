"""A22 opportunity bounds and real-core suffix checks; no diffusion checkpoint loaded."""
import copy
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from utils import diag_equal_nfe_path_opportunity as d
try:
    import torch
except ImportError:
    torch=None


def rows():
    result=[]
    for i in range(3):
        c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
        a=dict(c,loss=.9,psnr=20.3 if i==0 else 20.01)
        b=dict(c,loss=.9,psnr=20.01 if i==0 else 20.3)
        result.append(dict(case=dict(image_id=str(i)),branches=dict(C=c,A=a,B=b),floors=dict.fromkeys(d.METRICS,0.)))
    return result


class OpportunityTests(unittest.TestCase):
    def test_budget_counts_original_prefix_and_remaining_suffix(self):
        self.assertEqual(d.CALLS,108)
        self.assertEqual(12*d.CALLS,1296)
        self.assertEqual(len(d.ARMS),13)
        for k,t in d.POINTS:
            self.assertEqual(270+(k+1)+(29-k),300)
        self.assertEqual(d.POINTS,((8,903),(19,641),(24,319),(27,54),(28,14)))

    def test_vectorized_oracle_matches_brute_force_under_mean_guards(self):
        rr=rows();rr[0]['branches']['A']['lpips']=.22
        arms=('C','A','B');out=d.opportunity(rr,arms)
        eligible=[d.endpoint(rr,a) for a in itertools.product(arms,repeat=3)]
        expected=max(v['mean_delta']['psnr'] for v in eligible if v['quality_guard'])
        self.assertAlmostEqual(out['oracle_for_diagnosis_only']['mean_delta']['psnr'],expected)
        self.assertTrue(out['oracle_for_diagnosis_only']['quality_guard'])
        self.assertEqual(out['assignments_enumerated'],27)
        self.assertEqual(out['best_fixed'],'B')

    def test_strong_fixed_closes_false_adaptivity_but_heterogeneity_opens_bound(self):
        rr=rows();out=d.opportunity(rr,('C','A','B'))
        self.assertEqual(out['classification'],'ACTION_SELECTION_OPPORTUNITY_ONLY')
        for r in rr:r['branches']['B']['psnr']=20.4
        out=d.opportunity(rr,('C','A','B'))
        self.assertEqual(out['classification'],'NO_USEFUL_ACTION_SELECTION_HEADROOM')
        self.assertAlmostEqual(out['oracle_psnr_headroom'],0.)
        self.assertEqual(out['best_fixed'],'B')

    def test_invalid_inputs_and_mean_not_individual_guard(self):
        rr=rows();rr[0]['branches']['A']['lpips']=.21
        out=d.endpoint(rr,['A']*3)
        self.assertTrue(out['quality_guard'])
        self.assertEqual(out['cases_worsened']['lpips'],1)
        with self.assertRaises(ValueError):d.endpoint(rr,['A'])
        bad=copy.deepcopy(rr);bad[0]['branches']['A']['loss']=float('nan')
        with self.assertRaises(ValueError):d.opportunity(bad,('C','A','B'))

    def test_incomplete_does_not_interpret_and_retains_failed_cost(self):
        r=dict(status='failed',diagnostic_nfe=7)
        out=d.summarize([r])
        self.assertEqual(out['classification'],'INCOMPLETE_PILOT')
        self.assertEqual(out['diagnostic_nfe'],7)
        self.assertTrue(all('global_opportunity' not in g for g in out['groups']))


@unittest.skipIf(torch is None,'Server Torch required; no UNet loaded')
class CoreTests(unittest.TestCase):
    def test_real_core_full_replay_and_pulsed_suffix_use_exact_supplied_noise(self):
        from modules.diffusion_model import _precompute_diffusion_params
        from modules.zaps_algorithm import ddpm_posterior_step
        alpha=_precompute_diffusion_params()['alphas_cumprod']
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=alpha))
        rr=[dict(t=t,t_prev=s,parameter_index=i) for t,s,i in ((54,14,2),(14,0,1),(0,-1,0))]
        gen=torch.Generator(device='cpu').manual_seed(220010)
        x=torch.randn((1,3,4,4),generator=gen);noises={k:torch.randn(x.shape,generator=gen) for k in (0,1)}
        def field(z,y,x,t,i):return dict(clean=.2*x, q=.01*(i+1)*torch.ones_like(x))
        expected=x.clone();inputs=[];qs=[]
        before=torch.get_rng_state().clone()
        for k,row in enumerate(rr):
            inputs.append(expected.clone());f=field(z,None,expected,row['t'],row['parameter_index']);qs.append(f['q'])
            with patch('torch.randn_like',return_value=noises.get(k)) as draw:
                expected=ddpm_posterior_step(expected,f['clean'],row['t'],row['t_prev'],alpha,eta=1.,mode='ddpm')+f['q']
            self.assertEqual(draw.call_count,int(k<2))
        calls=[]
        def evaluate(*args):calls.append(args[3]);return field(*args)
        with patch('torch.randn_like',side_effect=RuntimeError('Fresh noise forbidden')):
            actual=d.continue_suffix(z,None,x,rr,0,noises,evaluate)
            self.assertEqual(calls,[54,14,0]);torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            calls.clear();null=d.continue_suffix(z,None,inputs[1],rr,1,noises,evaluate)
            torch.testing.assert_close(null,expected,atol=0,rtol=0);self.assertEqual(calls,[14,0])
            for gamma in (.5,2.):
                calls.clear();pulsed=d.continue_suffix(z,None,inputs[1]+(gamma-1)*qs[0],rr,1,noises,evaluate)
                self.assertEqual(calls,[14,0]);self.assertFalse(torch.equal(pulsed,expected))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
