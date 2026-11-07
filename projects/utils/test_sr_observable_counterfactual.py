"""CPU NumPy exact-H counterfactual identities and decision boundaries."""
import subprocess,sys,unittest
from pathlib import Path
import numpy as np
from utils import sr_observable_counterfactual as r

class Tests(unittest.TestCase):
 def test_repeated_boundary_taps_are_accumulated(self):
  a=r.matrix_from_taps(np.array([[0,1],[0,2],[1,3]]),np.array([[.2,.3],[.3,.4],[.5,.3]]),4)
  np.testing.assert_allclose(a,[[.5,.5,0,0],[0,.3,.4,.3]])
 def test_true_separable_projector_matches_dense_pseudoinverse(self):
  rng=np.random.default_rng(19);ay=rng.normal(size=(2,8));ax=rng.normal(size=(3,8));p=r.Projector(ay,ax);x=rng.normal(size=(8,8))
  dense=np.kron(ay,ax);expected=(np.linalg.pinv(dense)@dense@x.reshape(-1)).reshape(x.shape)
  np.testing.assert_allclose(p.row(x),expected,rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(p.H(p.null(x)),0,atol=1e-12)
 def test_null_error_can_increase_without_observation_change(self):
  ay=np.array([[1.,1.,0.,0.],[0.,0.,1.,1.]])/2;p=r.Projector(ay,ay);s=np.zeros((4,4));v=np.array([[1,-1,1,-1],[-1,1,-1,1]]*2,dtype=float);d=s+v
  images,visible,invisible=r.counterfactuals(s,d,p)
  np.testing.assert_allclose(images['ROW_ONLY_DIFFERENCE'],s,atol=1e-12)
  np.testing.assert_allclose(p.H(images['ROW_ONLY_DIFFERENCE']),p.H(d),atol=1e-12)
  risk=r.risks(d,s,p);self.assertAlmostEqual(risk['null_mse'],risk['raw_mse'])
  np.testing.assert_allclose(visible+invisible,d-s)
 def test_decision_is_terminal_only_and_accepts_ratio_above_one(self):
  rows=[dict(raw_excess=1.,null_excess=1.2,row_only_raw_improvement_over_diag=1.2)]*3
  d=r.decision(rows);self.assertTrue(d['supports_terminal_null_risk_contribution']);self.assertAlmostEqual(d['null_share_of_net_excess'],1.2)
  self.assertFalse(r.decision(rows[:2])['supports_terminal_null_risk_contribution'])
 def test_cli_no_torch_dependency(self):
  f=Path(__file__).with_name('diag_sr_observable_counterfactual.py');p=subprocess.run([sys.executable,str(f),'--help'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
