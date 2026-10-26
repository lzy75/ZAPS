"""Controls missing in A51. Torch numerical tests run on the server, no skips."""
import unittest
import torch
from utils.current_score_geometry import conjugate_gradient
from utils.diag_score_cg_metrology import reliable_cg


class UnrestartedTests(unittest.TestCase):
    def test_ill_conditioned_diagonal_and_analytic_solution(self):
        # Same spectral range as eta~1.59e-5 to eta+.25, not a blur/trajectory proof.
        diagonal=torch.logspace(-4.8,-.6,64,dtype=torch.float64)
        rhs=torch.ones(64,dtype=torch.float64)
        solution,gate=conjugate_gradient(lambda x:diagonal*x,rhs,rtol=1e-5,maxiter=512)
        self.assertTrue(gate['passed'])
        self.assertLessEqual(float((diagonal*solution-rhs).norm()/rhs.norm()),1e-5)
        expected=rhs/diagonal
        # Normwise solution error can exceed the residual for ill-conditioned SPD.
        self.assertLess(float((solution-expected).norm()/expected.norm()),.01)

    def test_periodic_restart_is_a_distinct_solver(self):
        diagonal=torch.logspace(-4.8,-.6,64,dtype=torch.float64)
        rhs=torch.ones(64,dtype=torch.float64)
        _,plain=conjugate_gradient(lambda x:diagonal*x,rhs,rtol=1e-5,maxiter=512)
        _,periodic=reliable_cg(lambda x:diagonal*x,rhs,rtol=1e-5,maxiter=512,refresh=32)
        self.assertTrue(plain['passed'])
        self.assertGreater(periodic['relative_residual'],plain['relative_residual'])
        self.assertGreater(periodic['restarts'],0)


if __name__=='__main__':unittest.main()
