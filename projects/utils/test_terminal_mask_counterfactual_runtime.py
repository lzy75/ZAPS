"""Required Torch CPU gate; missing Torch is failure, never skip."""
import unittest
import torch
from modules.wavelet import OrthogonalDWT2D
from utils.terminal_mask_counterfactual import reconstruct,parts,region_errors

class Tests(unittest.TestCase):
    def test_reconstruction_includes_original_wavelet_coupling(self):
        torch.manual_seed(7);n=32;mask=(torch.rand(1,1,n,n)>.7).float()
        m=torch.rand(1,3,n,n);y=torch.rand_like(m);dwt=OrthogonalDWT2D('db4',3)
        alpha=torch.tensor(.9999);zeta=torch.tensor([.2]);v=mask*(y-mask*m)
        uniform=dict(tau=torch.tensor([0]),D=torch.ones(1,3,n,n),zeta=zeta)
        q=reconstruct(m,y,mask,uniform,alpha,dwt)
        expected=zeta[0]*(v+(1-alpha)*v)/alpha.sqrt()
        self.assertTrue(torch.allclose(q,expected,atol=1e-6,rtol=1e-5))
        varied=dict(uniform,D=torch.linspace(.1,2,3*n*n).reshape(1,3,n,n))
        q=reconstruct(m,y,mask,varied,alpha,dwt);qo,qu=parts(q,mask)
        self.assertGreater(float(qu.double().square().sum()),0.)
        self.assertTrue(torch.equal(mask*qu,torch.zeros_like(q)))
    def test_region_mse_is_exact_partition(self):
        torch.manual_seed(8);m=torch.randn(1,3,16,16);gt=torch.randn_like(m);mask=(torch.rand(1,1,16,16)>.7).float()
        r=region_errors(m,gt,mask)
        self.assertAlmostEqual(r['mse_full'],r['observed_fraction']*r['mse_observed']+(1-r['observed_fraction'])*r['mse_missing'],places=12)

if __name__=='__main__':unittest.main()
