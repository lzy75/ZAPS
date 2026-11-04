"""A91 direction-only moments, exact covariance identity, and real CLI gates."""
import itertools,subprocess,sys,unittest
from pathlib import Path
import numpy as np
from utils import directional_reliability as rt

class Tests(unittest.TestCase):
 def test_independent_band_reliability_without_global_reference(self):
  np.testing.assert_allclose(rt.select([10.,10.],[1.,9.]),[.9,.1])
  np.testing.assert_allclose(rt.select([10.,10.],[4.,36.]),[.6,0.])
 def test_preserves_native_amplitude_and_known_noise_scaling(self):
  E=np.array([3.,7.,10.]);N=np.array([1.,2.,8.]);weights=rt.select(E,N)
  np.testing.assert_allclose(rt.select(9*E,9*N),weights)
  self.assertFalse(np.allclose(rt.select(E,4*N),weights))
 def test_noise_propagation_identity_with_signed_learned_factor(self):
  A=np.array([[1.,2.],[.2,-.3],[-1.,.4]]);ell=np.array([-2.,.5,1.3])
  probes=np.array(list(itertools.product((-1.,1.),repeat=2)))
  actual=np.mean((probes@A.T*ell)**2,axis=0)
  expected=ell**2*np.sum(A*A,axis=1)
  np.testing.assert_allclose(actual,expected,rtol=1e-15)
 def test_zero_energy_and_noise_boundaries(self):
  np.testing.assert_array_equal(rt.select([0.,1.,1.],[0.,0.,2.]),[0.,1.,0.])
  for E,N in (([-1.],[1.]),([1.],[-1.]),([float('nan')],[1.])):
   with self.assertRaises(ValueError):rt.select(E,N)
 def test_operator_banks_reuse_images_and_noise_but_preserve_masks(self):
  cases=[dict(task=t,mask_seed=2001000+i,sigma=s) for t in ('super_resolution','gaussian_deblur','motion_deblur','inpainting') for i in (226,443,963) for s in (.05,.10)]
  keys={rt.calibration_key(c) for c in cases}
  self.assertEqual(len(keys),6)
  self.assertEqual(rt.PROBES,16)
  self.assertEqual(rt.calibration_seed('gaussian_deblur_shared'),rt.calibration_seed('gaussian_deblur_shared'))
 def test_original_subband_layout_is_complete(self):
  cover=np.zeros((256,256),int);bands=rt.subbands(256,256,3)
  self.assertEqual(len(bands),10)
  for _,(ys,xs) in bands:cover[ys,xs]+=1
  np.testing.assert_array_equal(cover,np.ones_like(cover))
 def test_entry_cli_is_importable_without_torch(self):
  project=Path(__file__).resolve().parents[1]
  r=subprocess.run([sys.executable,'utils/diag_directional_reliability.py','--help'],cwd=project,text=True,capture_output=True)
  self.assertEqual(r.returncode,0,r.stderr)
  self.assertIn('--gpus',r.stdout)

if __name__=='__main__':unittest.main()
