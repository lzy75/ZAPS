"""Real Torch numerical admission; missing Torch is an error, never a skip."""
import unittest
import torch
from utils.diag_score_cg_metrology import reliable_cg


class NumericalTests(unittest.TestCase):
    def test_known_dense_spd_solution_and_true_residual(self):
        generator=torch.Generator().manual_seed(51)
        q,_=torch.linalg.qr(torch.randn(12,12,generator=generator,dtype=torch.float64))
        a=q@torch.diag(torch.logspace(-2,1,12,dtype=torch.float64))@q.T
        expected=torch.randn(12,generator=generator,dtype=torch.float64)
        b=a@expected
        x,g=reliable_cg(lambda v:a@v,b,maxiter=128,rtol=1e-10,refresh=32)
        self.assertTrue(g['passed'])
        self.assertLess(float((a@x-b).norm()/b.norm()),1e-10)
        self.assertLess(float((x-expected).norm()/expected.norm()),1e-8)

    def test_budget_failure_is_retained(self):
        a=torch.diag(torch.tensor([1.,10.,100.,1000.]))
        _,g=reliable_cg(lambda v:a@v,torch.ones(4),maxiter=1,rtol=1e-8)
        self.assertFalse(g['passed'])
        self.assertEqual(g['iterations'],1)
        self.assertEqual(g['operator_calls'],2)

    def test_operator_call_accounting_no_global_rng_or_input_mutation(self):
        calls=[]; b=torch.tensor([1.,2.,3.]); before=b.clone(); rng=torch.random.get_rng_state().clone()
        def action(v):calls.append(1);return 2*v
        x,g=reliable_cg(action,b)
        self.assertTrue(g['passed']);self.assertEqual(g['operator_calls'],len(calls))
        self.assertTrue(torch.equal(b,before));self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        torch.testing.assert_close(x,b/2)

    def test_zero_and_non_spd(self):
        _,g=reliable_cg(lambda x:x,torch.zeros(2))
        self.assertEqual(g['operator_calls'],0);self.assertTrue(g['passed'])
        with self.assertRaisesRegex(ValueError,'SPD'):
            reliable_cg(lambda x:-x,torch.ones(2))


if __name__=='__main__':unittest.main()
