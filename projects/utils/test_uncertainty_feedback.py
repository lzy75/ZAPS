import json
import unittest
import numpy as np
from utils import uncertainty_feedback as rules

GRID=[0,14,54,124,218,319,409,484,546,597,641,679,712,742,769,793,815,835,854,872,888,903,917,931,944,956,967,978,989,999]


class UncertaintyTests(unittest.TestCase):
    def test_gaussian_target_matches_full_conditioning_and_basis_rotation(self):
        mean=np.array([.1,-.2]);B=np.array([[.4,.1],[.1,.2]]);W=np.array([[1.,.5],[-.2,1.],[.5,0.]])
        residual=np.array([.8,-.4,.2]);sigma=.3;V=np.eye(2)
        target,state=rules.gaussian_target(mean,V,B,W,residual,sigma)
        direct=mean+B@W.T@np.linalg.solve(sigma*sigma*np.eye(3)+W@B@W.T,residual)
        np.testing.assert_allclose(target,direct,rtol=1e-12,atol=1e-12)
        R=np.array([[.6,-.8],[.8,.6]])
        rotated,_=rules.gaussian_target(mean,V@R,R.T@B@R,W@R,residual,sigma)
        np.testing.assert_allclose(target,rotated,rtol=1e-12,atol=1e-12)
        noisy,_=rules.gaussian_target(mean,V,B,W,residual,10.)
        self.assertLess(np.linalg.norm(noisy-mean),np.linalg.norm(target-mean))
        self.assertGreater(state['whitened_innovation'],0.)

    def test_covariance_rejects_skew_large_negative_and_scale_instability(self):
        valid=np.array([[.2,.02],[.02,.1]])
        gate,B=rules.covariance_gate([valid,valid*1.01]);self.assertTrue(gate['usable'])
        np.testing.assert_allclose(B,valid,atol=1e-12)
        for bad in (np.array([[.2,.2],[0.,.1]]),np.diag([.2,-.1]),valid*2.):
            gate,_=rules.covariance_gate([valid,bad]);self.assertFalse(gate['usable'])
        self.assertFalse(rules.covariance_gate([valid,valid],input_span_error=.1)[0]['usable'])
        self.assertEqual(rules.basis([np.ones(5),np.ones(5)*2]).shape,(5,1))

    def test_selection_uses_local_quantities_and_preserves_baseline_feasibility(self):
        cleans={g:np.array([g,0.]) for g in rules.GAMMAS};means=dict(cleans)
        args=(cleans,means,np.array([3.,0.]),None,1.,np.zeros(2),np.array([[1.],[0.]]),
              np.ones((1,1)),np.array([[1.],[0.]]),True,np.zeros(2))
        result=rules.choose(*args)
        self.assertEqual(result['gamma']['PHYSICAL'],2.)
        self.assertEqual(result['gamma']['FULL'],1.5)
        g=result['gamma']['FULL']
        self.assertLessEqual(result['local_measurement_loss'][g],result['local_measurement_loss'][1.])
        self.assertLessEqual(result['predicted_risk']['FULL'][g],result['predicted_risk']['FULL'][1.])
        invalid=list(args);invalid[-2]=False;fallback=rules.choose(*invalid)
        self.assertEqual(fallback['gamma']['FULL'],1.)
        with self.assertRaises(TypeError):rules.choose(*args,ground_truth=np.ones(2))
        encoded=rules.plain(dict(decision=result,gate=np.bool_(True),count=np.int64(58)))
        self.assertEqual(json.loads(json.dumps(encoded,allow_nan=False))['gate'],True)

    def test_inpainting_mask_and_affine_linear_scaling(self):
        masked=rules.active(np.array([[[[1.,9999.]],[[2.,-9999.]]]]),np.array([[[[True,False]]]]))
        np.testing.assert_array_equal(masked,[1.,2.])
        # Unit-domain L=.5H must be used for covariance propagation, not H itself.
        mean=np.zeros(1);C=np.ones((1,1));V=np.ones((1,1));res=np.ones(1)
        target,_=rules.gaussian_target(mean,V,C,np.array([[.5]]),res,1.)
        self.assertAlmostEqual(float(target[0]),.4)
        self.assertEqual(len(rules.calls(GRID,[2,2])),58)
        self.assertEqual(len(rules.calls(GRID,[1,1])),50)
        with self.assertRaises(ValueError):rules.calls(GRID,[2])

    def test_real_torch_finite_covariance_matches_linear_denoiser_without_rng(self):
        try:import torch
        except ImportError:self.skipTest('Torch absent on Mac; mandatory on server')
        from types import SimpleNamespace
        from utils.diag_uncertainty_feedback import covariance_probes,raw_field
        alpha=.5;diag=torch.linspace(.2,.8,16).reshape(1,1,4,4);calls=[]
        def eps(x,t):
            calls.append(int(t[0]));return (x-alpha**.5*diag*x)/(1-alpha)**.5
        z=SimpleNamespace(dm=SimpleNamespace(_predict_eps=eps,alphas_cumprod=torch.full((1000,),alpha)))
        center=torch.linspace(-1,1,16).reshape_as(diag);V=np.eye(16)[:,:2];before=torch.get_rng_state()
        matrices,derivatives,span=covariance_probes(lambda x:raw_field(z,x,218)[0],center,V,alpha)
        truth=(1-alpha)/alpha**.5*np.diag(diag.flatten().numpy()[:2])
        np.testing.assert_allclose(matrices[0],truth,rtol=1e-3,atol=1e-5)
        self.assertTrue(rules.covariance_gate(matrices,span)[0]['usable'])
        self.assertEqual(calls,[218]*8);self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
