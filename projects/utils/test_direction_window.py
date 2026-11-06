"""A93 formal representation geometry and zero-label protocol checks."""
import contextlib,inspect,io,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils import direction_window as r


class Tests(unittest.TestCase):
 def test_within_band_and_sign_limits_are_distinct(self):
  bands=[('all',(slice(0,1),slice(0,2)))];q=np.array([[1.,1.]])
  # Within-band amplitude freedom helps although no sign changes are needed.
  v=r.coefficient_geometry(q,np.array([[1.,3.]]),bands)
  self.assertGreater(v['cosine_coefficient_nonnegative'],v['cosine_band_nonnegative'])
  self.assertAlmostEqual(v['cosine_coefficient_nonnegative'],1.)
  # Nonnegative diagonal cannot produce opposite signs even per coefficient.
  v=r.coefficient_geometry(q,np.array([[1.,-1.]]),bands)
  self.assertAlmostEqual(v['cosine_coefficient_nonnegative'],2**-.5)
  self.assertAlmostEqual(v['cosine_coefficient_signed'],1.)
  self.assertAlmostEqual(v['reference_energy_opposite_sign_fraction'],.5)
 def test_zero_source_is_not_claimed_representable(self):
  v=r.coefficient_geometry(np.array([[1.,0.]]),np.array([[1.,1.]]),[('all',(slice(0,1),slice(0,2)))])
  self.assertEqual(v['zero_source_count'],1)
  self.assertAlmostEqual(v['cosine_coefficient_signed'],2**-.5)
  self.assertAlmostEqual(v['reference_energy_zero_source_fraction'],.5)
 def test_call_budget_and_no_label_inputs(self):
  calls=r.expected_calls(list(range(30)))
  self.assertEqual(calls,[16,17,17,18]+list(range(18,30)))
  self.assertEqual(len(calls),16);self.assertEqual(24*2*16,768);self.assertEqual(24*2*2,96)
  self.assertNotIn('gt',inspect.signature(r.run).parameters)
  self.assertNotIn('gt',inspect.signature(r.geometry).parameters)
 def test_cli_import_without_torch(self):
  p=Path(__file__).resolve().parents[1]
  result=subprocess.run([sys.executable,'utils/diag_direction_window.py','--help'],cwd=p,capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--geometry-only',result.stdout)
 def test_summary_uses_cached_controls_without_charging_old_calls(self):
  from utils import diag_direction_window as driver
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);case=dict(key='one',task='inpainting',sigma=.1,image_id='00226',references={})
   for arm,psnr in (('NATIVE_100',25.),('DIAG_ALIGN_100',26.)):
    path=root/'parent'/arm/'run.json';driver.files.write(path,dict(metrics=dict(psnr=psnr,ssim=.8,lpips=.2),counts={'forward_attempts':14}))
    case['references'][arm]=str(path)
   driver.files.write(root/'cases/one/DIAG_ALIGN_3/run.json',dict(status='complete',case=case,arm='DIAG_ALIGN_3',
     metrics=dict(psnr=27.,ssim=.9,lpips=.1),suffix_seconds=1.,counts=dict(forward_attempts=16,input_vjp_attempts=2)))
   plan=dict(experiment='A93_direction_capacity_window',cases=[case],protocol={})
   with patch.object(driver,'check',return_value=plan),contextlib.redirect_stdout(io.StringIO()) as output:
    summary=driver.summarize(root)
   self.assertEqual(summary['new_diagnostic_forward_attempts'],16)
   self.assertEqual(summary['new_input_vjp_attempts'],2)
   self.assertEqual({x['reference']:x['delta']['psnr'] for x in summary['comparisons']},{'NATIVE_100':2.,'DIAG_ALIGN_100':1.})
   self.assertLess(len(output.getvalue()),1200)

if __name__=='__main__':unittest.main()
