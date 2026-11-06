"""A94 proximal algebra, coupled objective and fixed protocol, no Torch needed."""
import contextlib,io,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils import joint_diagonal_response as r


class Tests(unittest.TestCase):
 def test_signed_and_nonnegative_prox_have_same_prior(self):
  np.testing.assert_allclose(r.prox(np.zeros(2),np.array([2.,-4.]),.5,1),[-2/3,4/3])
  np.testing.assert_allclose(r.prox(np.zeros(2),np.array([2.,-4.]),.5,1,True),[0,4/3])
  self.assertEqual(r.precision('SIGNED_DATA_ONLY'),0)
  self.assertEqual(r.precision('SIGNED_JOINT'),r.precision('NONNEG_JOINT'))
  np.testing.assert_array_equal(r.tangent_gradient(np.array([1.,3.]),'SCALAR_TIED'),[2.,2.])
 def test_joint_gradient_accounts_for_cross_component_response(self):
  B=np.array([[1.,1.],[0.,1.]]);y=np.array([1.,0.]);gamma=np.zeros(2)
  L=np.linalg.norm(B,2)**2
  for _ in range(300):gamma=r.prox(gamma,B.T@(B@gamma-y),1/L,1.)
  expected=np.linalg.solve(B.T@B+np.eye(2),B.T@y)
  np.testing.assert_allclose(gamma,expected,atol=1e-12)
  isolated=(B.T@y)/(np.sum(B*B,axis=0)+1)
  self.assertGreater(np.linalg.norm(gamma-isolated),.1)
 def test_majorization_rejects_oversized_numerical_step(self):
  self.assertFalse(r.bound_ok(10.,1.,-1.,1.,1.,1e-8))
  self.assertTrue(r.bound_ok(.5,1.,-1.,1.,1.,1e-8))
 def test_cli_and_summary_reuse_old_controls(self):
  project=Path(__file__).resolve().parents[1]
  process=subprocess.run([sys.executable,'utils/diag_joint_diagonal_response.py','--help'],cwd=project,capture_output=True,text=True)
  self.assertEqual(process.returncode,0,process.stderr)
  from utils import diag_joint_diagonal_response as driver
  self.assertTrue(all((project/n).is_file() for n in driver.NAMES))
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);a71=root/'a71';driver.files.write(a71/'summary.json',dict(groups=[dict(arm='SHRINK',sigma=.1)]))
   case=dict(key='one',task='inpainting',sigma=.1,image_id='00226',references={})
   for name in ('NATIVE_100','DIAG_ALIGN_100','PHYS_DIRECT_100'):
    path=root/'controls'/name/'run.json';driver.files.write(path,dict(metrics=dict(psnr=25.,ssim=.8,lpips=.2)));case['references'][name]=str(path)
   driver.files.write(root/'cases/one/SIGNED_JOINT/run.json',dict(status='complete',case=case,arm='SIGNED_JOINT',metrics=dict(psnr=26.,ssim=.9,lpips=.1),
      suffix_seconds=1.,solver_seconds=.4,counts=dict(forward_attempts=17,input_vjp_attempts=1),solver=dict(negative_fraction=.1,correction_norm=2.,native_norm=1.),audit=dict(clean_measurement_gain=2.,fitted_noise_gain=.1)))
   plan=dict(experiment='A94_joint_diagonal',a71_parent=str(a71),protocol={})
   with patch.object(driver,'check',return_value=plan),contextlib.redirect_stdout(io.StringIO()) as output:s=driver.summarize(root)
   self.assertEqual(s['new_diagnostic_forward_attempts'],17);self.assertEqual(s['new_input_vjp_attempts'],1)
   self.assertEqual(len(s['historical_A71_scalar']),1);self.assertLess(len(output.getvalue()),1200)

if __name__=='__main__':unittest.main()
