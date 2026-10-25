"""Zero-UNet algebra preflight; real Torch/FFHQ gates are separate."""
import unittest
import numpy as np
from utils import current_score_geometry as g


class AlgebraTests(unittest.TestCase):
    def test_nonnegative_solution_satisfies_kkt_and_boundary(self):
        rng = np.random.default_rng(50)
        for _ in range(100):
            q, d = rng.normal(size=(2, 17))
            gamma = g.nonnegative_projection(q, d)["gamma"]
            derivative = float(q @ (gamma*q-d))
            self.assertGreaterEqual(derivative, -1e-12)
            self.assertAlmostEqual(gamma*derivative, 0., places=11)
            for alternative in [0., gamma+.03, max(0., gamma-.03), 3.]:
                self.assertLessEqual(np.linalg.norm(gamma*q-d),
                                     np.linalg.norm(alternative*q-d)+1e-12)
        self.assertEqual(g.nonnegative_projection([1.], [-2.])["gamma"], 0.)
        self.assertFalse(g.nonnegative_projection([0.], [1.])["resolved"])

    def test_q_scalar_cancellation_is_explicit_not_learned_weight_preservation(self):
        q=np.array([.2,.5,.1]);v=np.array([1.,2.,.2]);u=np.array([.7,1.4,.3])
        first,second=g.score_targets(q,v,u,.3),g.score_targets(7.*q,v,u,.3)
        for mode in first["decisions"]:
            a,b=first["decisions"][mode],second["decisions"][mode]
            np.testing.assert_allclose(a["gamma"]*q,b["gamma"]*7.*q,rtol=1e-14)
        self.assertAlmostEqual(first["alignment_scale"],float(v@u/(u@u)))

    def test_skip_units_match_ddim_expression_including_terminal(self):
        for a,b in [(.01,.2),(.4,.8),(.999,1.)]:
            sigma2=(1.-b)/(1.-a)*(1.-a/b)
            direct=np.sqrt(1.-a)*(np.sqrt(b/a)*np.sqrt(1.-a)-np.sqrt(max(0.,1.-b-sigma2)))
            self.assertAlmostEqual(g.skip_coefficient(a,b),direct,places=12)
        with self.assertRaises(ValueError):g.skip_coefficient(.5,.4)

    def test_unresolved_finite_and_no_gt_contract(self):
        result=g.score_targets([1.],[2.],[0.],.1)
        self.assertIsNone(result["cosine_v_u"])
        self.assertFalse(result["decisions"]["ALIGN_Q"]["resolved"])
        with self.assertRaises(ValueError):g.score_targets([1.],[np.nan],[1.],.1)
        with np.errstate(over="ignore",invalid="ignore"):
            with self.assertRaises(ValueError):g.nonnegative_projection([1e200],[1e200])
        with self.assertRaises(TypeError):g.score_targets([1.],[1.],[1.],.1,ground_truth=[1.])


if __name__ == "__main__":
    unittest.main()
