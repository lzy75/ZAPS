"""CPU-only A97 scope/cost and crossed-vector contracts; no model claims."""
import subprocess,sys,unittest,tempfile
from pathlib import Path
import numpy as np
from utils import full_unroll_response as r,response_target_lm as lm

class Tests(unittest.TestCase):
 def test_full_scope_and_all_solver_costs(self):
  b=r.budget();self.assertEqual(len(r.ARMS),6);self.assertEqual(b['arms'],72)
  self.assertEqual(b['native_grid_forwards'],2160);self.assertEqual(b['response_base_forwards'],1392)
  self.assertEqual(b['max_new_neural_forwards'],5511348)
  self.assertEqual(b['max_neural_solver_forwards'],(1392+1044)*lm.budget()['max_solver_forward'])
  self.assertEqual((b['terminal_scalar_solves'],b['terminal_diagonal_solves']),(48,36))
 def test_cross_vectors_keep_registered_direction_and_requested_length(self):
  q=np.array([3.,4.]);s=q*.7;d=np.array([-8.,6.])
  a=r.cross.correction(q,s,d,'DIAG_DIR_SCALAR_LEN');b=r.cross.correction(q,s,d,'SCALAR_DIR_DIAG_LEN')
  np.testing.assert_allclose(a,d*np.linalg.norm(s)/10)
  np.testing.assert_allclose(b,q*2)
 def test_cli_help_has_no_torch_dependency(self):
  script=Path(__file__).with_name('diag_full_unroll_response.py')
  result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--worker-index',result.stdout)
 def test_compact_solver_preserves_budget_status(self):
  result=r.compact(dict(status='OUTER_BUDGET',converged=False,trace=[1]))
  self.assertEqual(result,dict(status='OUTER_BUDGET',converged=False))
 def test_partial_summary_keeps_paid_costs_and_writes_comparison(self):
  from utils import diag_full_unroll_response as d
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d.files.write(root/'plan.json',dict(experiment='A97_full_unroll',sources={},inputs={},protocol={}))
   for arm,psnr,forwards in [('BASE',25.,30),('REAL_DIAG',26.,400)]:
    d.files.write(root/'cases'/'sample'/arm/'run.json',dict(status='complete',case=dict(key='sample',task='super_resolution',image_id='00226'),arm=arm,
     metrics=dict(psnr=psnr,ssim=.8,lpips=.1),suffix_seconds=2.,equivalent_full_method_forward=270+forwards,solver_stops={},counts=dict(forwards=forwards,forward_attempts=forwards)))
   s=d.summarize(root);self.assertEqual(s['actual_new_cost']['forwards'],430);self.assertEqual(s['completed_runs'],2)
   self.assertFalse(s['comparisons'][0]['positive_development_candidate'])
   self.assertEqual(len((root/'comparison.csv').read_text().splitlines()),3)

if __name__=='__main__':unittest.main()
