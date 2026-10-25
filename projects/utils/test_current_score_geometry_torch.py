"""Required server preflight. Torch import failure is NOT a skipped pass."""
import unittest
import numpy as np
import torch
from utils import current_score_geometry as g


class TorchPreflight(unittest.TestCase):
    def test_cg_true_residual_spd_and_failed_budget(self):
        A=torch.tensor([[4.,1.],[1.,2.]],dtype=torch.float64)
        b=torch.tensor([1.,-2.],dtype=torch.float64)
        x,gate=g.conjugate_gradient(lambda z:A@z,b,rtol=1e-12,maxiter=4)
        self.assertTrue(gate["passed"])
        torch.testing.assert_close(x,torch.linalg.solve(A,b))
        _,gate=g.conjugate_gradient(lambda z:A@z,b,rtol=1e-12,maxiter=1)
        self.assertFalse(gate["passed"])
        with self.assertRaises(ValueError):
            g.conjugate_gradient(lambda z:-z,b)

    def test_affine_measurement_proximal_and_vjp_match_fixed_system_gradient(self):
        # Affine offset is observed; .5 unit-domain L is differentiated.
        dtype=torch.float64
        L=torch.tensor([[.5,.2,0.],[0.,.3,.4]],dtype=dtype)
        offset=torch.tensor([.1,-.3],dtype=dtype)
        J=torch.tensor([[.8,.1,0.],[.1,.5,.1],[0.,.1,.9]],dtype=dtype)
        weight=torch.nn.Parameter(J.clone())
        x=torch.tensor([.2,-.1,.4],dtype=dtype,requires_grad=True)
        mean=weight@x
        y=torch.tensor([.7,-.2],dtype=dtype)
        alpha,sigma=.64,.05
        residual=y-(L@mean.detach()+offset)
        v,gate=g.proximal_vector(lambda z:L@z,lambda z:L.T@z,residual,
                                 alpha,sigma,rtol=1e-12,maxiter=10)
        self.assertTrue(gate["passed"])
        a,b2=np.sqrt(alpha),1.-alpha
        S=L@L.T+(sigma*sigma*a/b2)*torch.eye(2,dtype=dtype)
        torch.testing.assert_close(v,(a/b2)*L.T@torch.linalg.solve(S,residual))
        u=g.input_vjp(mean,x,v)
        torch.testing.assert_close(u,J.T@v)
        self.assertIsNone(weight.grad)
        # Independent directional finite difference of the frozen-system cost.
        direction=torch.tensor([.3,-.2,.1],dtype=dtype)
        def objective(xx):
            rr=y-(L@(J@xx)+offset)
            return -.5*(a/b2)*(rr@torch.linalg.solve(S,rr))
        for h in (1e-3,5e-4):
            numeric=(objective(x.detach()+h*direction)-objective(x.detach()-h*direction))/(2*h)
            self.assertAlmostEqual(float(numeric),float(u@direction),places=9)

    def test_cotangent_is_not_differentiated_and_input_rng_unchanged(self):
        before=torch.random.get_rng_state().clone()
        x=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
        mean=x*x
        # If cotangent were differentiated, derivative of sum(mean*mean) differs.
        u=g.input_vjp(mean,x,mean)
        torch.testing.assert_close(u,2*x.detach()**3)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
