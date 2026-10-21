"""A25 clock assignment, paired inference boundary and real-core supplied-noise replay."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from utils import diag_state_clock_matching as d
from utils.test_diag_state_feedback_diagnosis import GRID
try:
    import torch
except ImportError:
    torch=None


def records():
    rr=[];positions=d.policy.Positions(np.linspace(1,0,1000),GRID)
    for i,image in enumerate(('00020','00021','00022')):
        for seed in (1000,2000):
            clock=[999]+[positions.choose(j,(i-1)*.8) for j in range(1,30)]
            c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
            branches=dict(C=c,SELF=dict(c,psnr=19.98),FIXED_GRID=dict(c,psnr=19.97),DONOR_PLUS=dict(c,psnr=19.96),DONOR_MINUS=dict(c,psnr=19.96),DONOR_MEAN=dict(c,psnr=19.96))
            rr.append(dict(status='complete',case=dict(image_id=image,sampling_seed=seed,key=image+'_s'+str(seed)),traces={'RC_REPLAY':[dict(t=t) for t in clock]},branches=branches,
                diagnostic_nfe=60,floors=dict.fromkeys(d.METRICS,0.),gates={'x':{'passed':True}},clock_changes={a:dict(different_nodes=12,mean_abs_delta_t=.2) for a in d.ARMS}))
    return rr


class MatchingTests(unittest.TestCase):
    def test_derangements_preserve_seed_clock_multiset_and_ignore_quality(self):
        rr=records();assignment=d.clock_map(rr);lookup={r['case']['key']:r for r in rr}
        for a in d.ARMS:
            self.assertEqual(sorted(v[a]['donor'] for v in assignment.values()),sorted(lookup))
            for key,arms in assignment.items():
                donor=lookup[arms[a]['donor']]
                self.assertNotEqual(donor['case']['image_id'],lookup[key]['case']['image_id'])
                self.assertEqual(donor['case']['sampling_seed'],lookup[key]['case']['sampling_seed'])
                self.assertEqual(arms[a]['times'],[x['t'] for x in donor['traces']['RC_REPLAY']])
        changed=copy.deepcopy(rr)
        for r in changed:r['branches']={'any_quality':1e99}
        self.assertEqual(d.clock_map(changed),assignment)
        with self.assertRaises(ValueError):d.clock_map(rr[:-1])

    def test_invalid_clocks_rejected_and_every_donor_in_original_cell(self):
        p=d.policy.Positions(np.linspace(1,0,1000),GRID)
        for v in d.clock_map(records()).values():
            for a in d.ARMS:d.validate_clock(v[a]['times'],p)
        for bad in (GRID[:-1],[998]+GRID[1:],GRID[:10]+[GRID[9]]+GRID[11:],[999,800]+GRID[2:]):
            with self.assertRaises(ValueError):d.validate_clock(bad,p)

    def test_matching_hint_cannot_erase_negative_net_result_or_pick_worst_donor(self):
        rr=records();out=d.summarize(rr)
        self.assertEqual(out['classification'],'CLOCK_MATCHING_HINT_WITHOUT_NET_GAIN')
        self.assertFalse(out['source_SELF_net_benefit_supported']);self.assertEqual(out['diagnostic_nfe'],360)
        for r in rr:
            r['branches']['DONOR_MINUS']['psnr']=20.01
            r['branches']['DONOR_PLUS']['psnr']=19.8
            r['branches']['DONOR_MEAN']['psnr']=19.905
        self.assertEqual(d.summarize(rr)['classification'],'CURRENT_STATE_CLOCK_MATCHING_NOT_SUPPORTED')
        out=d.summarize([dict(status='failed',diagnostic_nfe=7)])
        self.assertEqual(out['classification'],'INCOMPLETE_PILOT');self.assertEqual(out['diagnostic_nfe'],7)
        self.assertNotIn('SELF_minus_donor',out)


@unittest.skipIf(torch is None,'Server Torch required; no UNet checkpoint loaded')
class TensorTests(unittest.TestCase):
    def test_thirty_actual_times_supplied_draws_target_parameters_and_rng(self):
        from modules.diffusion_model import _precompute_diffusion_params
        from modules.zaps_algorithm import ddpm_posterior_step
        alpha=_precompute_diffusion_params()['alphas_cumprod'];p=d.policy.Positions(d.policy.noise_u(alpha.numpy()),GRID)
        clock=[999]+[p.choose(j,.7) for j in range(1,30)];calls=[]
        def eps(x,t):calls.append(int(t.item()));return .1*x
        identity=SimpleNamespace(H=lambda x:x,transpose=lambda x:x,analysis=lambda x:x,synthesis=lambda x:x)
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=alpha,_predict_eps=eps),A=identity,dwt=identity,
            zeta=torch.linspace(.0001,.0005,30),D=torch.linspace(.1,.8,30).view(30,1,1,1).expand(30,3,4,4))
        gen=torch.Generator().manual_seed(25001);x=.01*torch.randn((1,3,4,4),generator=gen);y=torch.zeros_like(x)
        noises=[torch.randn(x.shape,generator=gen) for _ in range(29)];before=torch.get_rng_state().clone()
        expected=x.clone()
        for k,t in enumerate(clock):
            f=d.source.field(z,y,expected,t,p);s=clock[k+1] if k<29 else -1
            with patch('torch.randn_like',return_value=noises[k] if k<29 else None) as draw:
                expected=ddpm_posterior_step(expected,f['clean'],t,s,alpha,eta=1.,mode='ddpm')+f['q']
            self.assertEqual(draw.call_count,int(k<29))
        calls.clear();observed=[]
        with patch('torch.randn_like',side_effect=RuntimeError('Fresh draw forbidden')):
            actual,rows=d.roll(z,y,x,clock,p,noises,lambda k,t:observed.append((k,t)))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(calls,clock);self.assertEqual(observed,list(enumerate(clock)));self.assertEqual([r['t'] for r in rows],clock)
        for row in rows:self.assertEqual(row['parameter_weights'],d.policy.parameters(z,row['t'],p)[2])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
