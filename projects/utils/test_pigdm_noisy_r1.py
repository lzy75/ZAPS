"""A85R1 local gates: staged admission, unchanged parent, no retries/deletions."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from utils import pigdm_noisy_r1 as rules
from utils import diag_pigdm_noisy_r1 as driver
from utils import a85r1_session as session
from utils import historical_ffhq_gain as files

class Tests(unittest.TestCase):
 def test_protocol_budget_coefficient_and_only_two_arms(self):
  self.assertEqual(rules.ARMS,('BASE','SHRINK'))
  self.assertIn('sqrt(alpha_s*alpha_t)',rules.protocol()['formula'])
  self.assertEqual(2*len(rules.cases('ffhq','/fixed'))*rules.STEPS,2400)
  self.assertNotEqual(rules.root_for('ffhq'),__import__('utils.pigdm_noisy_transfer',fromlist=['root_for']).root_for('ffhq'))
 def test_full_grid_linear_surrogate_regression(self):
  # Algebraic constant-mode SR example, not a model reconstruction result.
  import numpy as np
  alpha=np.cumprod(1-np.linspace(.0001,.02,1000));grid=list(range(990,-1,-10))
  def roll(published):
   x=np.float32(1.)
   with np.errstate(over='ignore',invalid='ignore'):
    for i,t in enumerate(grid):
     a=float(alpha[t]);b=float(alpha[grid[i+1]]) if i+1<len(grid) else 1.
     k=(1-np.sqrt(1-a)*.05)/np.sqrt(a);m=k*x
     v=(.2-m)/(1+16*.05**2/(1-a));g=k*v
     q=np.sqrt(a*(b if published else 1))*g
     c1=np.sqrt(max(0.,(1-a/b)*(1-b)/(1-a)));c2=np.sqrt(max(0.,1-b-c1*c1))
     x=np.float32(np.sqrt(b)*m+c2*.05*x+q)
     if not np.isfinite(x):return False
   return bool(np.isfinite(x))
  self.assertFalse(roll(False));self.assertTrue(roll(True))
 def test_original_a85_bound_files_are_unchanged(self):
  for name in ('pigdm_noisy_transfer.py','pigdm_noisy_runtime.py','diag_pigdm_noisy_transfer.py','a85_session.py'):
   expected=subprocess.check_output(['git','show','c06a3f2:projects/utils/'+name],cwd=rules.PROJECTS.parent)
   self.assertEqual(expected,(rules.PROJECTS/'utils'/name).read_bytes())
 def test_baseline_admission_blocks_before_comparison(self):
  cases=rules.cases('ffhq','/fixed');plan=dict(dataset='ffhq',cases=cases)
  def good(*args):return dict(metrics=dict(psnr=25.,ssim=.8,raw_rms=.5))
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with patch.object(driver,'check',return_value=plan),patch.object(driver,'completed',side_effect=good):driver.qualify_baseline(root)
   self.assertTrue(files.read(root/'baseline_admission.json')['passed'])
   with patch.object(driver,'check',return_value=plan),patch.object(driver,'completed',return_value=dict(metrics=dict(psnr=6.,ssim=.01,raw_rms=4.))):
    with self.assertRaises(ValueError):driver.qualify_baseline(root)
   self.assertFalse(files.read(root/'baseline_admission.json')['passed'])
 def test_real_cli_and_phase_separation(self):
  for name in ('diag_pigdm_noisy_r1.py','a85r1_session.py'):
   r=subprocess.run([sys.executable,str(rules.PROJECTS/'utils'/name),'--help'],text=True,capture_output=True)
   self.assertEqual(r.returncode,0,r.stderr)
  cmd=session.worker_command('ffhq',Path('/fixed'),0,'UUID','baseline')
  self.assertEqual(cmd[-2:],['--phase','baseline'])
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);files.write(root/'workers/0_baseline/status.json',dict(status='complete'))
   self.assertEqual(session.status(root,0,'baseline'),'complete')
   self.assertEqual(session.status(root,0,'comparison'),'not_started')
 def test_no_update_of_original_module_or_gradient_calls_added(self):
  tree=ast.parse((rules.PROJECTS/'utils/pigdm_noisy_r1_runtime.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  self.assertEqual(sum(n.func.attr=='grad' for n in calls),1)
  self.assertEqual(sum(n.func.attr=='backward' for n in calls),0)

if __name__=='__main__':unittest.main()
