"""A92 geometry/diagnostic contract checks without Torch."""
import contextlib,inspect,io,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from utils import overall_direction_audit as r


class Tests(unittest.TestCase):
 def test_cone_projection_maximizes_overall_cosine(self):
  # Orthogonal components with unequal energies and a reference outside cone.
  Q=np.diag([2.,.5,3.]);ref=np.array([1.,-2.,4.])
  coefficients=r.cone_coefficients(np.sum(Q*Q,axis=0),Q.T@ref)
  v=Q@coefficients;best=v@ref/np.linalg.norm(v)/np.linalg.norm(ref)
  self.assertEqual(coefficients[1],0.)
  for a in (0.,.1,1.,3.):
   for b in (0.,.1,1.,3.):
    for c in (0.,.1,1.,3.):
     candidate=Q@np.array([a,b,c])
     if np.linalg.norm(candidate):self.assertGreaterEqual(best+1e-12,candidate@ref/np.linalg.norm(candidate)/np.linalg.norm(ref))
 def test_zero_components_and_invalid_moments(self):
  np.testing.assert_array_equal(r.cone_coefficients([0.,1.],[0.,-1.]),[0.,0.])
  with self.assertRaises(ValueError):r.cone_coefficients([-1.],[0.])
 def test_paired_factorial_no_oracle_selector(self):
  self.assertEqual(len(r.ARMS),9);self.assertEqual(r.ARMS[0],'NATIVE_100')
  self.assertEqual({r.specification(a) for a in r.ARMS},{(k,s) for k in r.KINDS for s in r.SCALES})
  for f in (r.prefix,r.physical_reference,r.directions,r.continue_run):
   self.assertNotIn('gt',inspect.signature(f).parameters)
  self.assertEqual(24*(16+1+len(r.ARMS)*14),3432)
 def test_driver_cli(self):
  p=Path(__file__).resolve().parents[1]
  x=subprocess.run([sys.executable,'utils/diag_overall_direction_audit.py','--help'],cwd=p,capture_output=True,text=True)
  self.assertEqual(x.returncode,0,x.stderr);self.assertIn('--gpus',x.stdout)
 def test_summary_pairs_same_lengths_and_keeps_probe_cost(self):
  from utils import diag_overall_direction_audit as driver
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary)
   driver.files.write(root/'plan.json',dict(experiment='A92_overall_direction',sources={},inputs={},protocol={}))
   case=dict(key='one',task='inpainting',sigma=.1,image_id='00226')
   driver.files.write(root/'cases/one/prefix.json',dict(status='complete',counts={'forward_attempts':16},probe_counts={'forward_attempts':1,'input_vjp_attempts':1}))
   for arm in ('NATIVE_100','NATIVE_050','DIAG_ALIGN_050'):
    driver.files.write(root/'cases/one'/arm/'run.json',dict(status='complete',case=case,arm=arm,counts={'forward_attempts':14},
      metrics=dict(psnr=25.,ssim=.8,lpips=.2),continuation_seconds=1.,audit=dict(physical_gain=2.,clean_measurement_gain=1.,fitted_noise_gain=1.)))
   with contextlib.redirect_stdout(io.StringIO()):driver.summarize(root)
   summary=driver.files.read(root/'summary.json')
   self.assertEqual(summary['recorded_diagnostic_forwards'],59)
   self.assertEqual(summary['input_vjp_attempts'],1)
   self.assertTrue(any(x['arm']=='DIAG_ALIGN_050' and x['reference']=='NATIVE_050' and x['equal_length_orientation_test'] for x in summary['comparisons']))

if __name__=='__main__':unittest.main()
