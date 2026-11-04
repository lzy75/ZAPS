"""A90 true SciPy solves, ill-conditioned directions, multiscale layout."""
import unittest
import numpy as np
from utils import aligned_diagonal_feedback as rt

class Tests(unittest.TestCase):
 def test_original_db4_layout_has_ten_disjoint_subbands(self):
  bands=rt.subbands(256,256,3);cover=np.zeros((256,256),int)
  self.assertEqual(len(bands),10)
  self.assertEqual(bands[0][0],'LL3')
  counts=[]
  for label,(ys,xs) in bands:
   cover[ys,xs]+=1;counts.append(int(cover[ys,xs].size))
  np.testing.assert_array_equal(cover,np.ones((256,256),int))
  self.assertEqual(counts,[1024]*4+[4096]*3+[16384]*3)
 def test_mature_nnls_ball_solution_is_analytic_for_identity(self):
  out=rt.fit(np.eye(2),[3.,4.],[1.,0.])
  np.testing.assert_allclose(out['amplitudes'],[.6,.8],rtol=2e-7,atol=1e-8)
  self.assertGreater(out['dual_lambda'],0)
  self.assertLessEqual(out['norm_ratio'],1.)
  self.assertLess(out['projected_kkt_residual'],1e-6)
 def test_weak_response_cannot_amplify_correction_energy(self):
  a=np.diag([1.,1e-4]);b=np.ones(2);anchor=np.array([1.,0.])
  free=np.linalg.solve(a,b);self.assertGreater(np.linalg.norm(free),9999)
  out=rt.fit(a,b,anchor);z=np.array(out['amplitudes'])
  self.assertLessEqual(np.linalg.norm(z),1+1e-7)
  self.assertLessEqual(np.linalg.norm(a@z-b),np.linalg.norm(a@anchor-b)+1e-6)
 def test_positivity_and_zero_budget_are_preserved(self):
  out=rt.fit(np.eye(2),[-3.,4.],[1.,0.])
  np.testing.assert_allclose(out['amplitudes'],[0.,1.],rtol=1e-6,atol=1e-8)
  out=rt.fit(np.eye(2),[2.,3.],[0.,0.]);self.assertEqual(out['amplitudes'],[0.,0.])
  self.assertEqual(out['nnls_calls'],0)
 def test_rank_deficiency_and_scalar_feasibility(self):
  rng=np.random.default_rng(5)
  u=rng.normal(size=(80,10));u[:,9]=u[:,8]
  target=rng.normal(size=80);anchor=np.full(10,.2)
  orth,r=np.linalg.qr(u,mode='reduced');compressed=orth.T@target
  out=rt.fit(r,compressed,anchor);z=np.array(out['amplitudes'])
  self.assertTrue(np.isfinite(z).all());self.assertTrue((z>=0).all())
  self.assertLessEqual(np.linalg.norm(z),np.linalg.norm(anchor)*(1+1e-7))
  self.assertLessEqual(np.linalg.norm(u@z-target),np.linalg.norm(u@anchor-target)+1e-6)
 def test_scaling_response_does_not_change_physical_solution(self):
  a=np.array([[1.,.2],[0.,.05]]);b=np.array([1.,.7]);anchor=np.array([.8,.6])
  one=rt.fit(a,b,anchor);two=rt.fit(1e-8*a,1e-8*b,anchor)
  np.testing.assert_allclose(one['amplitudes'],two['amplitudes'],rtol=1e-6,atol=1e-8)
 def test_zero_response_does_not_invent_information(self):
  out=rt.fit(np.zeros((2,2)),[1.,2.],[.3,.4])
  self.assertEqual(out['amplitudes'],[.3,.4])
  self.assertEqual(out['solver_status'],'ZERO_RESPONSE_KEEP_SCALAR')

if __name__=='__main__':unittest.main()
