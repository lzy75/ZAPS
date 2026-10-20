import unittest
from unittest.mock import patch
import numpy as np
from utils import diag_ddpm_refinement_contract as d
try:
    import torch
except ImportError:
    torch=None


class ContractTests(unittest.TestCase):
    def test_frozen_field_endpoint_equality_across_random_intervals(self):
        rng=np.random.default_rng(20)
        for _ in range(80):
            at,am,ass=np.sort(rng.uniform(.0001,.9999,3))
            c=d.interval(at,am,ass)
            x,z0,q,z,aux=rng.normal(size=(5,3,8,8))
            coarse,fine=d.frozen_pair(x,z0,q,z,aux,c)
            np.testing.assert_allclose(coarse,fine,atol=2e-12,rtol=2e-12)
            self.assertGreaterEqual(c['local_guidance_factor'],.5)
            self.assertLessEqual(c['local_guidance_factor'],1.)

    def test_noise_rotation_preserves_independent_normal_marginals_and_coarse_draw(self):
        c=d.interval(.02,.17,.65)
        matrix=np.array([[c['w1'],c['w2']],[c['w2'],-c['w1']]])
        np.testing.assert_allclose(matrix@matrix.T,np.eye(2),atol=1e-14)
        z=np.array([1.,-2.,3.]);aux=np.array([.5,1.,-.2]);n1,n2=d.coupled_noises(z,aux,c)
        actual=c['second']['a']*c['first']['sigma']*n1+c['second']['sigma']*n2
        np.testing.assert_allclose(actual,c['coarse']['sigma']*z,atol=1e-14)

    def test_naive_half_doses_and_reused_noise_fail_the_contract(self):
        c=d.interval(.02,.17,.65)
        self.assertLess(.5*(1+c['second']['a']),1.)
        variance_wrong=(c['second']['a']*c['first']['sigma']+c['second']['sigma'])**2
        self.assertGreater(variance_wrong,c['coarse']['sigma']**2)

    def test_invalid_or_terminal_intervals_rejected_and_five_intervals_preserved(self):
        for values in ((.5,.5,.8),(.8,.5,.9),(.01,.5,1.)):
            with self.assertRaises(ValueError):d.interval(*values)
        result=d.preflight()
        self.assertEqual([r['t'] for r in result['rows']],[903,641,319,54,14])
        self.assertEqual(result['diagnostic_nfe'],0)


@unittest.skipIf(torch is None,'Requires server Torch to compare the actual core, no UNet needed')
class CoreTests(unittest.TestCase):
    def test_actual_ddpm_helper_with_explicit_coupled_draws_and_guidance(self):
        from modules.zaps_algorithm import ddpm_posterior_step
        from modules.diffusion_model import _precompute_diffusion_params
        alpha=_precompute_diffusion_params()['alphas_cumprod'].double()
        rng=np.random.default_rng(201)
        for row in d.preflight()['rows']:
            t,m,s=row['t'],row['m'],row['s']
            contract=d.interval(float(alpha[t]),float(alpha[m]),float(alpha[s]))
            x,clean,q,z,aux=[torch.tensor(v,dtype=torch.float64) for v in rng.normal(size=(5,1,3,8,8))]
            n1,n2=d.coupled_noises(z,aux,contract);dose=contract['local_guidance_factor']
            with patch('torch.randn_like',side_effect=[z,n1,n2]):
                coarse=ddpm_posterior_step(x,clean,t,s,alpha,eta=1.)+q
                mid=ddpm_posterior_step(x,clean,t,m,alpha,eta=1.)+dose*q
                fine=ddpm_posterior_step(mid,clean,m,s,alpha,eta=1.)+dose*q
            torch.testing.assert_close(coarse,fine,atol=1e-11,rtol=1e-11)


if __name__=='__main__':unittest.main()
