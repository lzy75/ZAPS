"""A52 target algebra + actual Torch/legacy-checkpoint admission, no skipped tests."""
import unittest
import numpy as np
from utils import feedback_vector_constraint as rules
from utils import current_score_geometry as algebra


class NumpyTests(unittest.TestCase):
    def test_nonparallel_target_is_not_scalar_q(self):
        q=np.array([1.,0.]);v=np.array([1.,1.]);u=np.array([2.,1.])
        target,projection=rules.numpy_target(q,v,u,.5,'FULL_ALIGN')
        np.testing.assert_allclose(target,[.6,.3])
        self.assertAlmostEqual(projection['gamma'],.6)
        self.assertGreater(np.linalg.norm(target-projection['gamma']*q),0.)

    def test_parallel_positive_targets_agree(self):
        q=np.array([1.,2.]);v=3*q;u=2*q
        for mode in ('FULL_PROX','FULL_ALIGN'):
            target,decision=rules.numpy_target(q,v,u,.2,mode)
            np.testing.assert_allclose(target,decision['gamma']*q)

    def test_projection_norm_bound_and_old_multiplier(self):
        rng=np.random.default_rng(52)
        for _ in range(40):
            q,v,u=rng.normal(size=(3,12));coefficient=.37
            old=algebra.score_targets(q,v,u,coefficient)
            for mode,old_mode in (('FULL_PROX','PROX_Q'),('FULL_ALIGN','ALIGN_Q')):
                target,decision=rules.numpy_target(q,v,u,coefficient,mode)
                self.assertLessEqual(np.linalg.norm(target),coefficient*np.linalg.norm(v)+1e-12)
                self.assertAlmostEqual(decision['gamma'],old['decisions'][old_mode]['gamma'],places=12)

    def test_degenerate_u_is_not_accepted(self):
        with self.assertRaises(ValueError):rules.numpy_target([1.,0.],[0.,1.],[0.,0.],.2,'FULL_ALIGN')


class TorchTests(unittest.TestCase):
    def test_terminal_applies_full_vector_once(self):
        import torch
        from types import SimpleNamespace
        from utils.common_state_risk import advance
        clean=torch.tensor([[[[.2,-.1]]]],dtype=torch.float32)
        target=torch.tensor([[[[.03,.07]]]],dtype=torch.float32)
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.tensor([.9999])))
        out=advance(z,torch.zeros_like(clean),dict(clean=clean,q=target),
                    SimpleNamespace(nominal=[0]),0,[],1.)
        torch.testing.assert_close(out,clean+target,rtol=0.,atol=0.)

    def test_gpu_reduction_path_matches_independent_numpy(self):
        import torch
        from utils import current_score_runtime_r2 as runtime
        q=torch.tensor([[1.,0.]],dtype=torch.float64)
        v=torch.tensor([[1.,1.]],dtype=torch.float64)
        u=torch.tensor([[2.,1.]],dtype=torch.float64)
        reading=runtime.gram_readout(q,v,u,.5,1.)
        for mode in ('FULL_PROX','FULL_ALIGN'):
            target,extra=rules.target_from_readout(q,v,u,.5,mode,reading)
            expected,decision=rules.numpy_target(q.numpy(),v.numpy(),u.numpy(),.5,mode)
            np.testing.assert_allclose(target.numpy(),expected)
            self.assertAlmostEqual(extra['shadow_gamma'],decision['gamma'],places=12)
            self.assertLessEqual(extra['shadow_projection_error'],2e-6)

    def test_unresolved_fallback_and_no_input_mutation(self):
        import torch
        from utils import current_score_runtime_r2 as runtime
        q=torch.tensor([[1.,2.]],dtype=torch.float32);before=q.clone()
        v=q.clone();u=q.clone();reading=runtime.gram_readout(q,v,u,.2,1.)
        rules.target_from_readout(q,v,u,.2,'FULL_ALIGN',reading)
        self.assertTrue(torch.equal(q,before))
        with self.assertRaises(ValueError):
            rules.target_from_readout(q,v,u,.2,'FULL_ALIGN',dict(resolved=False))


if __name__=='__main__':unittest.main()
