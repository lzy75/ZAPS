import unittest
from types import SimpleNamespace
from utils import transfer_response_feedback as rules


class TransferTests(unittest.TestCase):
    def test_time_profile_uses_old_traces_without_GT_metrics(self):
        grid = list(range(30))
        records = [dict(case=dict(image_id=f'{i:05d}',sampling_seed=s),
                        rows=[dict(t=t,gamma=.2+(i-20)*.1) for t in reversed(grid)],
                        metrics=dict(psnr=-1000.)) for i in range(20,26) for s in (1000,2000)]
        first = rules.frozen_profile(records,grid)
        self.assertAlmostEqual(first[0],.45)
        for r in records:r['metrics']['psnr']=999999.
        self.assertEqual(first,rules.frozen_profile(records,grid))
        with self.assertRaises(ValueError):rules.frozen_profile(records[:-1],grid)

    def test_cluster_summary_separates_baseline_transfer_from_state_value(self):
        records=[]
        for i in range(6):
            for sigma in rules.SIGMAS:
                base=dict(psnr=25.,ssim=.8,lpips=.2,loss=1.)
                better=dict(psnr=25.2,ssim=.81,lpips=.199,loss=.9)
                trace=[dict(gamma=1.,cap_hit=False,fallback=False,R=1.)]*30
                records.append(dict(case=dict(image_id=str(i),sigma=sigma,sampling_seed=1000),
                                    metrics=dict(C=base,TIME=better,STATE=better),traces=dict(TIME=trace,STATE=trace)))
        r=rules.summarize(records)
        self.assertTrue(r['state_transfer_candidate']);self.assertTrue(r['time_transfer_candidate'])
        self.assertEqual(r['state_increment_candidate_sigmas'],[])
        self.assertTrue(all(g['comparisons']['STATE_C']['image_clusters']==6 for g in r['groups']))
        for item in records:
            if item['case']['sigma']==.1:item['metrics']['STATE']=dict(item['metrics']['STATE'],lpips=.22)
        self.assertFalse(rules.summarize(records)['state_transfer_candidate'])
        with self.assertRaises(ValueError):rules.summarize(records+[records[0]])

    def test_actual_torch_sensor_preserves_gamma_and_uses_true_sigma_for_R(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; mandatory on server')
        from utils.continuous_response_feedback import sensor as original
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.ones(1)),A=SimpleNamespace(H=lambda x:2*x))
        q=torch.tensor([1.,2.]);clean=torch.zeros_like(q);y=.31*2*q
        expected=original(z,y,clean,q,0,2.)
        for sigma in rules.SIGMAS:
            r=rules.sensor(z,y,clean,q,0,sigma)
            self.assertEqual(r['gamma'],expected['gamma'])
            self.assertAlmostEqual(r['R']*sigma,expected['R']*.05)
            self.assertEqual(r['raw_gamma'],expected['raw_gamma'])

    def test_actual_core_replay_uses_the_same_twenty_nine_noise_draws(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; mandatory on server')
        from modules.zaps_algorithm import ZAPS
        from utils import diag_frozen_response_transfer as run
        class Operator(torch.nn.Module):
            def H(self,x):return .5*x
            def transpose(self,x):return .5*x
        dm=SimpleNamespace(device=torch.device('cpu'),num_steps=1000,
                           alphas_cumprod=torch.linspace(.9999,.01,1000),_predict_eps=lambda x,t:.01*x)
        z=ZAPS(dm,Operator(),img_size=16,level=1,use_learned_var=False)
        torch.manual_seed(123);initial=torch.randn(1,3,16,16);y=torch.zeros_like(initial)
        before=torch.get_rng_state()
        with torch.no_grad():original=z._reverse_diffusion(y,init_noise=initial)
        after=torch.get_rng_state();torch.set_rng_state(before)
        noises=[torch.randn_like(initial) for _ in range(29)]
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        positions=run.source.fields.policy.Positions(run.source.fields.policy.noise_u(dm.alphas_cumprod.numpy()),z.tau.tolist()[::-1])
        x=initial
        with torch.no_grad():
            for k,t in enumerate(positions.nominal):
                f=run.source.fields.field(z,y,x,t,positions)
                x=run.P.terminal.terminal_output(z,x,f) if k==29 else run.P.coupled.posterior(x,f['clean'],t,positions.nominal[k+1],dm.alphas_cumprod,noises[k])+f['q']
        self.assertLess(float((x-original).abs().max()),1e-6)


if __name__=='__main__':unittest.main()
