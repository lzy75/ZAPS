"""A96 factorial identities, zero-ray semantics and standalone entry (no Torch)."""
import contextlib,io,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils import direction_length_attribution as r


class Tests(unittest.TestCase):
 def test_crossed_vectors_match_requested_direction_and_length(self):
  native=np.array([1.,0.]);scalar=2*native;diagonal=np.array([3.,4.])
  ds=r.correction(native,scalar,diagonal,'DIAG_DIR_SCALAR_LEN');sd=r.correction(native,scalar,diagonal,'SCALAR_DIR_DIAG_LEN')
  np.testing.assert_allclose(ds,[1.2,1.6]);np.testing.assert_allclose(sd,[5.,0.])
  self.assertAlmostEqual(r.length(ds),r.length(scalar));self.assertAlmostEqual(r.length(sd),r.length(diagonal))
  np.testing.assert_array_equal(r.correction(native,scalar,diagonal,'DIAG_REPLAY'),diagonal)
  np.testing.assert_array_equal(r.correction(native,scalar,diagonal,'ZERO'),[0.,0.])
 def test_zero_scalar_keeps_defined_original_ray_and_factorial_interaction(self):
  np.testing.assert_array_equal(r.correction(np.array([1.,0.]),np.zeros(2),np.array([0.,3.]),'SCALAR_DIR_DIAG_LEN'),[3.,0.])
  np.testing.assert_array_equal(r.correction(np.array([1.,0.]),np.zeros(2),np.array([0.,3.]),'DIAG_DIR_SCALAR_LEN'),[0.,0.])
  with self.assertRaises(ValueError):r.correction(np.array([1.]),np.array([1.]),np.array([0.]),'DIAG_DIR_SCALAR_LEN')
  values=dict(SCALAR_REPLAY=1.,DIAG_REPLAY=4.,DIAG_DIR_SCALAR_LEN=2.,SCALAR_DIR_DIAG_LEN=3.,ZERO=0.)
  effects=r.factorial(values);self.assertEqual(effects['direction_at_scalar_length'],1.);self.assertEqual(effects['length_on_scalar_ray'],2.);self.assertEqual(effects['interaction'],0.)
 def test_cli_and_fixed_stage_summary(self):
  project=Path(__file__).resolve().parents[1]
  command=subprocess.run([sys.executable,'utils/diag_direction_length_attribution.py','--help'],cwd=project,capture_output=True,text=True)
  self.assertEqual(command.returncode,0,command.stderr)
  from utils import diag_direction_length_attribution as driver
  self.assertTrue(all((project/n).is_file() for n in driver.NAMES))
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary)
   for i,arm in enumerate(r.ARMS):
    stages={k:{name:1. for name in ('raw_mse','clamped_mse','coarse_LL3_mse','detail_mse','clean_measurement_mse','clamp_fraction')} for k in ('16','24','29','output')}
    driver.files.write(root/'cases/one'/arm/'run.json',dict(status='complete',case=dict(key='one',task='super_resolution',image_id='00226'),arm=arm,metrics=dict(psnr=float(i),ssim=.8,lpips=.2),counts=dict(forward_attempts=14),stages=stages,audit=dict(response_target_objective=1.,clean_measurement_gain=2.,fitted_noise_gain=.1)))
   with patch.object(driver,'check',return_value=dict(experiment='A96',protocol={})),contextlib.redirect_stdout(io.StringIO()):summary=driver.summarize(root)
   self.assertEqual(summary['actual_new_cost']['forward_attempts'],70);self.assertEqual(len(summary['contrasts']),1)

if __name__=='__main__':unittest.main()
