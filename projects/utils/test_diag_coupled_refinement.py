"""A3 protocol/core tests; Torch checks load no UNet."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from utils import diag_coupled_refinement as d
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def test_midpoints_unique_seeds_and_invalid_intervals(self):
        alpha=np.cumprod(1-np.linspace(1e-4,.02,1000)).astype(np.float32)
        specs=[d.interval_spec(alpha,t,s) for t,s in ((903,888),(641,597),(319,218),(54,14),(14,0))]
        self.assertEqual([s['m'] for s in specs],[896,619,267,28,3])
        seeds=[d.bridge_seed(c,k,b) for c in range(6) for k in (8,19,24,27,28) for b in (0,1)]
        self.assertEqual(len(set(seeds)),60)
        self.assertEqual(d.bridge_seed(0,8,0),322026)
        for args in ((6,8,0),(0,0,0),(0,8,2)):
            with self.assertRaises(ValueError):d.bridge_seed(*args)
        for t,s in ((14,-1),(1,0)):
            with self.assertRaises(ValueError):d.interval_spec(alpha,t,s)

    def test_joint_quality_and_practical_gain_both_required(self):
        c=dict(loss=10.,psnr=20.,ssim=.7,lpips=.3);floors={k:0. for k in c}
        def hit(**changes):
            return d.candidate(d.history.effect({k:c[k]+changes.get(k,0.) for k in c},c,floors))
        self.assertTrue(hit(psnr=.11,loss=-.1));self.assertTrue(hit(lpips=-.002))
        self.assertFalse(hit(psnr=.01));self.assertFalse(hit(psnr=1.,lpips=.01));self.assertFalse(hit(psnr=1.,loss=.01))

    def test_image_clusters_require_both_seeds_and_bridges(self):
        records=[]
        for image in ('00020','00021','00022'):
            for seed in (1000,2000):
                records.append(dict(status='complete',case=dict(image_id=image,sampling_seed=seed),method_nfe=0,diagnostic_nfe=30,
                    points=[dict(source_t=t,both_bridges_candidate=image!='00022',bridge_candidates=dict(F0=True,F1=image!='00022'),
                    effects={a:dict(delta=dict(loss=-1,psnr=.2,ssim=.001,lpips=-.002)) for a in ('F0_C','F1_C')}) for t in d.TIMES]))
        result=d.summarize(records)
        self.assertEqual(result['classification'],'LOCAL_COUPLED_REFINEMENT_CANDIDATE_ONLY')
        self.assertEqual(result['diagnostic_nfe'],180)
        self.assertEqual(result['layers'][0]['images_both_seeds_both_bridges'],2)
        for r in records[:1]:
            for p in r['points']:p['both_bridges_candidate']=False
        self.assertEqual(d.summarize(records)['classification'],'LOCAL_COUPLED_REFINEMENT_NOT_SUPPORTED')
        self.assertEqual(d.summarize(records[:-1])['classification'],'INCOMPLETE_PILOT')

    def test_interrupted_directory_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'case').mkdir()
            with self.assertRaisesRegex(ValueError,'Interrupted case'):d.completed(root,dict(source_cases={'case':'unused'}))

    def test_changed_outcome_not_reused_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'case';case.mkdir();(root/'plan.json').write_text('{}');(case/'branches.pt').write_bytes(b'test')
            c=dict(loss=10.,psnr=20.,ssim=.7,lpips=.3);f=dict(c,psnr=20.2);floors={k:0. for k in c}
            points=[dict(source_t=t,branches=dict(C=c,F0=f,F1=f),floors=floors,
                effects={a+'_C':d.history.effect(f,c,floors) for a in ('F0','F1')},
                bridge_candidates=dict(F0=True,F1=True),both_bridges_candidate=True) for t in d.TIMES]
            record=dict(status='complete',case={'key':'case'},method_nfe=0,diagnostic_nfe=30,calls=[{}]*30,points=points,
                gates={'x':{'passed':True}},plan_sha256=d.digest(root/'plan.json'),branch_archive={'sha256':d.digest(case/'branches.pt')})
            def save():
                (case/'run.json').write_text(json.dumps(record));(case/'run.sha256').write_text(d.digest(case/'run.json'))
            save();self.assertEqual(len(d.completed(root,dict(source_cases={'case':'unused'}))),1)
            record['points'][0]['both_bridges_candidate']=False;save()
            with self.assertRaisesRegex(ValueError,'H5 classification'):d.completed(root,dict(source_cases={'case':'unused'}))


@unittest.skipIf(torch is None,'Server Torch required; no UNet loaded')
class TorchTests(unittest.TestCase):
    def test_float32_actual_core_null_endpoints_and_rng(self):
        from modules.diffusion_model import _precompute_diffusion_params
        from utils.diag_zaps_trace_audit import relative_error
        alpha=_precompute_diffusion_params()['alphas_cumprod'];gen=torch.Generator(device='cpu').manual_seed(201020)
        before=torch.get_rng_state().clone()
        for t,s in ((903,888),(641,597),(319,218),(54,14),(14,0)):
            x,clean,q,z,aux=torch.randn((5,1,3,16,16),generator=gen)
            spec=d.interval_spec(alpha.numpy(),t,s);coarse=d.posterior(x,clean,t,s,alpha,z)+q
            mid,fine,n1,n2=d.frozen_bridge(x,clean,q,z,aux,spec,alpha)
            self.assertLessEqual(relative_error(coarse,fine),1e-6)
            self.assertLessEqual(float((coarse-fine).abs().max()),1e-5)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with self.assertRaises(ValueError):d.posterior(x,clean,t,s,alpha,z.double())

    def test_anchored_increment_matches_full_nonfrozen_two_substeps_in_double(self):
        from modules.diffusion_model import _precompute_diffusion_params
        alpha=_precompute_diffusion_params()['alphas_cumprod'].double()
        gen=torch.Generator(device='cpu').manual_seed(201021)
        for t,s in ((903,888),(641,597),(319,218),(54,14),(14,0)):
            x,clean,q,z,aux,clean_mid,q_mid=torch.randn((7,1,3,8,8),generator=gen,dtype=torch.float64)
            spec=d.interval_spec(alpha.numpy(),t,s)
            mid,_,n1,n2=d.frozen_bridge(x,clean,q,z,aux,spec,alpha)
            coarse=d.posterior(x,clean,t,s,alpha,z)+q
            direct=d.posterior(mid,clean_mid,spec['m'],s,alpha,n2)+spec['local_guidance_factor']*q_mid
            anchored=d.refined_endpoint(coarse,clean,q,clean_mid,q_mid,spec)
            torch.testing.assert_close(anchored,direct,atol=1e-11,rtol=1e-11)
            torch.testing.assert_close(d.refined_endpoint(coarse,clean,q,clean,q,spec),coarse,atol=0,rtol=0)

    def test_midpoint_guidance_uses_original_alpha_and_parameters(self):
        class Identity:
            H=staticmethod(lambda x:x);transpose=staticmethod(lambda x:x)
            analysis=staticmethod(lambda x:x);synthesis=staticmethod(lambda x:x)
        z=SimpleNamespace(A=Identity(),dwt=Identity(),dm=SimpleNamespace(alphas_cumprod=torch.tensor([.9,.4,.1])),
                          zeta=torch.tensor([.2,.5]),D=torch.tensor([3.,7.]))
        y=torch.ones((1,3,4,4));clean=torch.full_like(y,.25)
        actual=d.frozen_guidance(z,y,clean,2,1)
        torch.testing.assert_close(actual,.5*(y-clean)*(1.+.9*7.)/(.1**.5))
        self.assertFalse(torch.allclose(actual,d.frozen_guidance(z,y,clean,1,1)))
        self.assertFalse(torch.allclose(actual,d.frozen_guidance(z,y,clean,2,0)))

    def test_raw_noise_capture_replays_all_draws_without_changing_rng(self):
        from modules.diffusion_model import _precompute_diffusion_params
        class Identity:
            H=staticmethod(lambda x:x);transpose=staticmethod(lambda x:x)
            analysis=staticmethod(lambda x:x);synthesis=staticmethod(lambda x:x)
        alpha=_precompute_diffusion_params()['alphas_cumprod']
        z=SimpleNamespace(A=Identity(),dwt=Identity(),dm=SimpleNamespace(alphas_cumprod=alpha),zeta=torch.zeros(30),D=torch.ones(30))
        before=torch.get_rng_state().clone();gen=torch.Generator(device='cpu').manual_seed(1000)
        state=dict(cpu_rng=gen.get_state(),init_noise=torch.zeros(1,3,4,4))
        grid=np.linspace(0,999,30).round().astype(int)[::-1].tolist()
        rows=[dict(parameter_index=29-k,t=t,t_prev=grid[k+1] if k<29 else -1) for k,t in enumerate(grid)]
        x=torch.ones(1,3,4,4);inputs=[];cleans=[];draws=[]
        with torch.random.fork_rng():
            torch.set_rng_state(state['cpu_rng'])
            for row in rows:
                inputs.append(x.clone());clean=torch.full_like(x,.2);cleans.append(clean)
                if row['t_prev']>=0:
                    raw=torch.randn_like(x);draws.append(raw);x=d.posterior(x,clean,row['t'],row['t_prev'],alpha,raw)
                else:x=clean
        cache=dict(latent_inputs=torch.cat(inputs),clean=torch.cat(cleans),guidance=torch.zeros(30,3,4,4))
        def restore(state,device,index):torch.set_rng_state(state['cpu_rng'])
        with patch('utils.diag_zaps_guidance_clipping.saved_cuda_selection',return_value={'saved_sampling_index':0}), \
             patch('utils.diag_zaps_guidance_clipping.restore_sampling_rng',side_effect=restore):
            actual,gate=d.replay_noises(state,cache,{'rows':rows},z,torch.zeros_like(x),'cpu',x)
        self.assertEqual(gate['noise_draws'],29)
        for k in range(29):torch.testing.assert_close(actual[k],draws[k],atol=0,rtol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
