"""A89 orthogonality, coupled NNLS and immutable cache-resolution checks."""
from pathlib import Path
import tempfile,unittest
import numpy as np
from utils import diagonal_feedback as rt

class Tests(unittest.TestCase):
 def test_orthogonal_complete_bands_and_scalar_reduction(self):
  x=np.random.default_rng(12).normal(size=(1,3,8,8));parts=rt.haar_components(x)
  np.testing.assert_allclose(sum(parts),x,rtol=0,atol=1e-15)
  np.testing.assert_allclose(sum(1.3*p for p in parts),1.3*x,rtol=0,atol=1e-15)
  for i in range(4):
   for j in range(i):self.assertLess(abs(float((parts[i]*parts[j]).sum())),1e-12)
  changed=sum(g*p for g,p in zip((1.,.2,2.,.5),parts))
  self.assertGreater(np.linalg.norm(changed-x),1.)
 def test_scalar_limit_matches_existing_noise_rule(self):
  from utils.noise_signal_shrink import select
  r=np.array([2.,-1.,.5]);u=np.array([.4,-.2,1.]);sigma=.1
  rule=select(float(r@u),float(r@r),float(u@u),3,sigma)
  target=rule['weight']*r
  out=rt.fit([[u@u]],[u@target],[rule['gamma']],float(target@target))
  self.assertAlmostEqual(out['amplitudes'][0],rule['gamma'],places=12)
 def test_coupling_and_nonnegative_boundary(self):
  u=np.array([[1.,.5],[0.,1.]]);r=np.array([1.,-.2])
  out=rt.fit(u.T@u,u.T@r,[.5,.5],r@r);z=np.array(out['amplitudes'])
  np.testing.assert_allclose(z,[1.,0.],atol=1e-12)
  independent=np.maximum(0,(u.T@r)/np.diag(u.T@u))
  self.assertLess(np.linalg.norm(u@z-r),np.linalg.norm(u@independent-r))
  self.assertLess(out['target_objective'],out['scalar_target_objective'])
 def test_nullspace_tie_keeps_scalar_anchor(self):
  u=np.array([[1.,0.],[0.,0.]]);r=np.array([2.,1.])
  out=rt.fit(u.T@u,u.T@r,[.7,.8],r@r)
  np.testing.assert_allclose(out['amplitudes'],[2.,.8],atol=1e-12)
  out=rt.fit(np.zeros((4,4)),np.zeros(4),[.1,.2,.3,.4],2.)
  np.testing.assert_allclose(out['amplitudes'],[.1,.2,.3,.4],atol=0)
 def test_dependent_columns_and_noise_zero_target(self):
  out=rt.fit([[1.,1.],[1.,1.]],[2.,2.],[.1,1.],4.)
  np.testing.assert_allclose(out['amplitudes'],[.55,1.45],atol=1e-12)
  out=rt.fit(np.eye(4),np.zeros(4),np.zeros(4),0.)
  self.assertEqual(out['amplitudes'],[0.]*4)
 def test_reused_sr_binary_is_resolved_without_modifying_history(self):
  from utils.diag_diagonal_feedback import resolve_artifact,files
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);source=path/'A70';old=path/'A71';source.mkdir();old.mkdir()
   (source/'entry.pt').write_bytes(b'fixture')
   files.write(source/'run.json',dict(status='complete',artifacts={'entry.pt':files.digest(source/'entry.pt')}))
   files.write(old/'run.json',dict(status='complete',artifacts={},source_record=str(source/'run.json'),source_record_sha256=files.digest(source/'run.json')))
   before=files.digest(old/'run.json');bound={}
   self.assertEqual(resolve_artifact(old/'run.json','entry.pt',bound),str(source/'entry.pt'))
   self.assertEqual(files.digest(old/'run.json'),before)
   self.assertEqual(bound[str(source/'entry.pt')],files.digest(source/'entry.pt'))
   (source/'entry.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):resolve_artifact(old/'run.json','entry.pt',{})

if __name__=='__main__':unittest.main()
