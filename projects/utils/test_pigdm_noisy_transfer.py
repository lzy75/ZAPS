"""Local design gates without Torch; actual numerical gates are separate REQUIRED CUDA."""
import ast
from pathlib import Path
import unittest
import numpy as np
from utils import pigdm_noisy_transfer as rules
from utils import native_ray_transfer as upstream
from utils import noise_signal_shrink as policy

class Tests(unittest.TestCase):
 def test_panel_budget_and_only_two_arms(self):
  for dataset in ('ffhq','imagenet'):
   cases=rules.cases(dataset,'/fixed')
   self.assertEqual(len(cases),12);self.assertEqual(len({c['key'] for c in cases}),12)
   self.assertEqual([sum(c['shard']==s for c in cases) for s in (0,1)],[6,6])
   self.assertEqual(len({c['image_id'] for c in cases}),3)
   self.assertEqual(2*len(cases)*rules.STEPS,2400)
  self.assertEqual(rules.ARMS,('BASE','SHRINK'))
  self.assertNotEqual(rules.root_for('ffhq'),rules.root_for('imagenet'))
 def test_original_sources_not_changed(self):
  self.assertEqual(upstream.verify('A84')['commit'],upstream.HOSTS['A84']['commit'])
 def test_noisy_variance_and_spectral_stability(self):
  alpha=np.cumprod(1-np.linspace(.0001,.02,1000))
  ve=(1-alpha)/alpha
  np.testing.assert_allclose(ve/(1+ve),1-alpha,rtol=1e-12)
  # Noise regularization bounds every inverse data singular gain by1/(2sqrt(lambda)).
  singular=np.linspace(0,1,10001);lam=.05**2/(1-alpha[990])
  gain=singular/(singular**2+lam)
  self.assertLessEqual(gain.max(),1/(2*np.sqrt(lam))+1e-12)
  self.assertLess(gain.max(),10.001)
 def test_observed_noise_energy_and_existing_policy(self):
  a=policy.select(5.,10.,4.,30,.05)
  self.assertAlmostEqual(a['gamma'],(1-30*.05**2/10)*5/4)
  self.assertEqual(policy.select(5.,10.,4.,3000,.05)['noise_energy'],7.5)
 def test_cli_has_real_entries_and_no_old_sources_written(self):
  import subprocess,sys
  for name in ('diag_pigdm_noisy_transfer.py','a85_session.py'):
   out=subprocess.run([sys.executable,str(rules.PROJECTS/'utils'/name),'--help'],capture_output=True,text=True)
   self.assertEqual(out.returncode,0,out.stderr);self.assertIn('--dataset',out.stdout)
  tree=ast.parse((rules.PROJECTS/'utils/pigdm_noisy_runtime.py').read_text())
  self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='backward' for n in ast.walk(tree)))

if __name__=='__main__':unittest.main()
