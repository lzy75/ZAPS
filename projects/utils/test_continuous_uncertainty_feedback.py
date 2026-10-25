import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from utils import continuous_uncertainty_feedback as rules


class ContinuousUncertaintyTests(unittest.TestCase):
    def test_nystrom_preserves_measured_offspan_columns_and_is_psd(self):
        V=np.eye(3)[:,:2];C=np.array([[2.,0.,.4],[0.,1.,.2],[.4,.2,1.]])
        Y=C@V;B=V.T@Y;F,gate=rules.nystrom_factor(V,Y,B)
        np.testing.assert_allclose(F@(F.T@V),Y,atol=1e-12)
        self.assertLess(gate['column_closure_relative_error'],1e-12)
        self.assertGreater(np.linalg.norm(V@B@V.T@V-Y),.4)
        self.assertGreaterEqual(np.linalg.eigvalsh(F@F.T).min(),-1e-12)
        # This preserves the sketch, not the complete covariance outside its observed span.
        self.assertGreater(np.linalg.norm(F@F.T-C),.5)
        with self.assertRaises(ValueError):rules.nystrom_factor(V,Y,-B)

    def test_factor_gaussian_target_equals_dense_conditioning(self):
        V=np.eye(3)[:,:2];C=np.array([[2.,0.,.4],[0.,1.,.2],[.4,.2,1.]])
        F,_=rules.nystrom_factor(V,C@V,V.T@C@V)
        L=np.array([[.5,.1,0.],[0.,.5,.2]]);mean=np.array([.1,-.2,.3]);res=np.array([.2,-.4]);sigma=.3
        target,_=rules.old.gaussian_target(mean,F,np.eye(2),L@F,res,sigma)
        N=F@F.T;direct=mean+N@L.T@np.linalg.solve(sigma*sigma*np.eye(2)+L@N@L.T,res)
        np.testing.assert_allclose(target,direct,atol=1e-12)

    def test_noise_cap_is_total_displacement_budget_and_rounding_stays_inside(self):
        result=rules.noise_budget(.3,.5,.02)
        self.assertAlmostEqual(result['posterior_noise_RMS'],np.sqrt(.5/.7*(1-.3/.5)))
        self.assertLessEqual(result['cap']*.02,result['posterior_noise_RMS']+1e-12)
        curve={g:(g-127.)**2 for g in rules.old.GAMMAS}
        fit=rules.proposal(curve,4.1237)
        self.assertTrue(fit['usable']);self.assertEqual(fit['gamma'],4.123)
        self.assertAlmostEqual(fit['vertex'],127.,places=6)
        self.assertEqual(rules.noise_budget(.3,.5,0.)['cap'],1.)
        self.assertEqual(rules.noise_budget(.3,.5,10.)['cap'],1.)
        with self.assertRaises(ValueError):rules.noise_budget(.5,.3,.02)

    def test_actual_checks_backtrack_instead_of_executing_unmeasured_vertex(self):
        curve={g:(g-6.)**2 for g in rules.old.GAMMAS};calls=[]
        def measured(g):
            calls.append(g)
            return dict(loss=10. if g>3 else 1.,objective=(g-2.)**2)
        result=rules.continuous_choice(curve,7.,measured)
        self.assertEqual(result['gamma'],2.25)
        self.assertEqual(calls,[1.,6.,3.5,2.25])
        self.assertEqual([v['accepted'] for v in result['attempts']],[False,False,True])
        with self.assertRaises(TypeError):rules.continuous_choice(curve,7.,measured,ground_truth=np.zeros(2))

    def test_unreliable_fit_state_or_nonfinite_candidate_falls_back(self):
        curve={g:(g-3.)**2 for g in rules.old.GAMMAS}
        def forbidden(_):raise AssertionError('No forward expected')
        self.assertEqual(rules.continuous_choice(curve,8.,forbidden,usable=False)['gamma'],1.)
        bad={g:v for g,v in zip(rules.old.GAMMAS,[0.,1.,0.,1.,0.])}
        self.assertEqual(rules.continuous_choice(bad,8.,forbidden)['gamma'],1.)
        def nonfinite(g):return dict(loss=1. if g==1 else None,objective=1. if g==1 else None)
        result=rules.continuous_choice(curve,8.,nonfinite)
        self.assertEqual(result['gamma'],1.);self.assertEqual(len(result['attempts']),3)
        json.dumps(rules.old.plain(result),allow_nan=False)

    def test_action_gain_does_not_automatically_count_as_state_increment(self):
        deltas={n:dict(psnr=.2,ssim=.001,lpips=-.001,loss=-.01) for n in rules.POLICIES}
        self.assertFalse(rules.state_increment(deltas))
        deltas['NYSTROM']['psnr']=.25;self.assertTrue(rules.state_increment(deltas))
        deltas['NYSTROM']['lpips']=.02;self.assertFalse(rules.state_increment(deltas))

    def test_trace_contract_includes_every_rejected_query_and_native_suffix(self):
        from utils.diag_continuous_uncertainty_feedback import expected_trace
        grid=list(range(30));r=dict(query_jobs=[dict(step=19,gamma=4.),dict(step=19,gamma=2.5)],
                                  suffix_jobs=[dict(step=24,gamma=2.5,seconds=1.)])
        result=expected_trace(r,grid)
        self.assertEqual(len(result),7);self.assertEqual(result[0]['t'],9)
        self.assertEqual([v['t'] for v in result[2:]],[4,3,2,1,0])

    def test_real_torch_lookahead_meter_and_cache_do_not_consume_rng(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; required without skips on server')
        from types import SimpleNamespace
        from utils.diag_continuous_uncertainty_feedback import counted,parent
        class LinearDenoiser(torch.nn.Module):
            def forward(self,x,t):return .2*x
        network=LinearDenoiser();model=SimpleNamespace(model=network)
        z=SimpleNamespace(dm=SimpleNamespace(_predict_eps=network,alphas_cumprod=torch.full((1000,),.5)))
        before=torch.get_rng_state().clone();context=dict(kind='lookahead',step=24,gamma=3.25);r={}
        with tempfile.TemporaryDirectory() as temp:
            with counted(model,Path(temp)/'run.json',r,context),torch.no_grad():
                raw,clean=parent.raw_field(z,torch.ones(1,1,4,4),218)
            self.assertEqual(r['nfe'],1);self.assertEqual(r['calls'],[dict(context,t=218)])
            self.assertTrue(torch.isfinite(raw).all());self.assertTrue(torch.equal(before,torch.get_rng_state()))
            saved=json.loads((Path(temp)/'run.json').read_text());self.assertEqual(saved['nfe'],1)


if __name__=='__main__':unittest.main()
