"""A26/A27 bounded controls, leakage boundary, clustered uncertainty and actual-core replay."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from utils import state_weight_scan as r
from utils import diag_state_weight_scan as d
from utils.test_diag_state_feedback_diagnosis import GRID
try:
    import torch
except ImportError:
    torch=None


class ScanTests(unittest.TestCase):
    def test_bank_is_unique_bounded_and_budget_counts_every_arm(self):
        bank=r.candidates();self.assertEqual(len(bank),27)
        states=[p for p in bank.values() if p['kind']=='STATE'];self.assertEqual(len(states),25)
        self.assertEqual(len({(p['rho'],p['a'],p['b']) for p in states}),25)
        self.assertEqual(12*27*30+24*(300+6*30),21240)
        for p in states:
            for er in (-1.,0.,1.):
                for ec in (None,-1.,0.,1.):
                    E,g=r.controls(p,dict(e_R=er,e_C=ec),319)
                    self.assertTrue(-1<=E<=1);self.assertTrue(.5<=g<=2)
                    self.assertEqual(r.controls(p,dict(e_R=er,e_C=ec),320)[1],1.)
        self.assertEqual(r.controls(bank['F'],dict(e_R=0,e_C=0),319),(0.,2.))
        self.assertEqual(r.controls(bank['F'],dict(e_R=0,e_C=0),318),(0.,1.))

    def test_direction_damps_guidance_and_scan_keeps_original_slot_contract(self):
        positions=r.prior.Positions(np.linspace(1,0,1000),GRID)
        for j in range(1,30):
            for E in (-1.,-.3,0.,.8,1.):
                self.assertEqual(r.choose(positions,j,E,.25),positions.choose(j,E))
                t=r.choose(positions,j,E,.5)
                self.assertTrue(t==0 if j==29 else t in positions.cells[j])
                self.assertEqual(r.choose(positions,j,E,0.),GRID[j])
        p=dict(kind='STATE',rho=.5,a=.5,b=1.)
        self.assertLess(r.controls(p,dict(e_R=.7,e_C=1),54)[1],r.controls(p,dict(e_R=.7,e_C=0),54)[1])
        matched=dict(p,kind='MATCHED')
        self.assertEqual(r.controls(matched,dict(e_R=.7,e_C=1),54),r.controls(p,dict(e_R=.7,e_C=0),54))
        self.assertEqual(r.controls(matched,dict(e_R=.7,e_C=None),54),r.controls(p,dict(e_R=.7,e_C=None),54))
        with self.assertRaises(ValueError):r.choose(positions,4,2,.5)

    def test_selection_uses_only_development_and_locks_phase_control(self):
        bank=r.candidates();positions=r.prior.Positions(np.linspace(1,0,1000),GRID);records=[]
        for image in range(6):
            for seed in (1000,2000):
                c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2);branches={};traces={}
                for name,p in bank.items():
                    gain=.1 if p['kind']=='STATE' else 0.
                    branches[name]=dict(c,psnr=20+gain)
                    traces[name]=[dict(t=t,gamma=1.1) for t in GRID]
                records.append(dict(case=dict(image_id=str(image),sampling_seed=seed),branches=branches,traces=traces,loss_floor=1e-5))
        chosen=r.select(records,bank,positions)
        self.assertFalse(chosen['validation_used']);self.assertEqual(chosen['stage']['times'],GRID)
        self.assertTrue(all(.5<=g<=2 for g in chosen['stage']['gamma']))
        self.assertTrue(0<chosen['parameters']['S']['a']<1)
        self.assertEqual(chosen['parameters']['P']['a'],1.);self.assertEqual(chosen['parameters']['D']['a'],0.)
        changed=copy.deepcopy(records)
        for x in changed:x['unseen_validation_scores']={'best':'R0.5_A0.5_B1'}
        self.assertEqual(r.select(changed,bank,positions),chosen)

    def test_confirmation_pairs_seeds_and_cannot_call_fixed_gain_state_success(self):
        records=[]
        for i in range(12):
            for seed,offset in ((1000,-1.),(2000,1.)):
                c=dict(loss=1.,psnr=20.+i,ssim=.7,lpips=.2)
                records.append(dict(case=dict(image_id=str(i),sampling_seed=seed),branches=dict(C=c,F=dict(c,psnr=c['psnr']+.2),P=dict(c,psnr=c['psnr']+.2),M=c,STAGE=dict(c,psnr=c['psnr']+.2),S=dict(c,psnr=c['psnr']+.2+offset)),loss_floor=1e-5))
        out=r.confirmation(records)
        self.assertEqual(out['image_clusters'],12);self.assertEqual(len(out['primary_bounds']),11)
        np.testing.assert_allclose(out['comparisons']['F']['psnr']['ci95'],[0.,0.],atol=1e-12,rtol=0)
        self.assertNotEqual(out['classification'],'CONFIRMED_STATE_GAIN_IN_THIS_COHORT')
        for x in records:x['branches']['S']['psnr']+=.2
        self.assertEqual(r.confirmation(records)['classification'],'CONFIRMED_STATE_GAIN_IN_THIS_COHORT')
        with self.assertRaises(ValueError):r.confirmation(records[:-1])


@unittest.skipIf(torch is None,'Server Torch required; no UNet checkpoint loaded')
class TensorTests(unittest.TestCase):
    def test_null_and_archived_pulse_arithmetic_exact_with_thirty_calls_and_rng(self):
        from modules.diffusion_model import _precompute_diffusion_params
        from modules.zaps_algorithm import ddpm_posterior_step
        alpha=_precompute_diffusion_params()['alphas_cumprod'];positions=r.prior.Positions(r.prior.noise_u(alpha.numpy()),GRID);calls=[]
        def eps(x,t):calls.append(int(t.item()));return .1*x
        ident=SimpleNamespace(H=lambda x:x,transpose=lambda x:x,analysis=lambda x:x,synthesis=lambda x:x)
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=alpha,_predict_eps=eps),A=ident,dwt=ident,zeta=torch.linspace(.0001,.0005,30),D=torch.full((30,3,4,4),.2))
        gen=torch.Generator().manual_seed(260027);x=.01*torch.randn((1,3,4,4),generator=gen);y=torch.zeros_like(x);state={'init_noise':x}
        noises={k:torch.randn(x.shape,generator=gen) for k in range(29)};rng=torch.get_rng_state().clone()
        clean=x.clone();inputs=[];qs=[]
        for k,t in enumerate(GRID):
            inputs.append(clean.clone());f=d.source.field(z,y,clean,t,positions);qs.append(f['q']);s=GRID[k+1] if k<29 else -1
            with patch('torch.randn_like',return_value=noises.get(k)):
                clean=ddpm_posterior_step(clean,f['clean'],t,s,alpha,eta=1.,mode='ddpm')+f['q']
        expectedF=inputs[25]+qs[24]
        for k in range(25,30):
            t=GRID[k];s=GRID[k+1] if k<29 else -1;f=d.source.field(z,y,expectedF,t,positions)
            with patch('torch.randn_like',return_value=noises.get(k)):
                expectedF=ddpm_posterior_step(expectedF,f['clean'],t,s,alpha,eta=1.,mode='ddpm')+f['q']
        ref=dict(R=dict(u=[0.,1.],log_values=[0.,0.]),Omega=dict(u=[0.,1.],log_values=[0.,0.]))
        for arm,expected in (('C',clean),('F',expectedF)):
            calls.clear()
            with patch('torch.randn_like',side_effect=RuntimeError('Fresh noise forbidden')):
                output,rows=d.roll(z,y,state,noises,positions,ref,dict(parameters=r.candidates()[arm]),lambda k,t:None)
            torch.testing.assert_close(output,expected,rtol=0,atol=0);self.assertEqual(calls,GRID)
            self.assertEqual([v['t'] for v in rows],GRID)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
