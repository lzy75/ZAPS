"""Minimal CPU-only coupled-descent, structure, binding and cost checks."""
import subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils import fast_response_refinement as r

class Linear:
 def __init__(self):self.a=np.array([[2.,1.,.3],[1.,3.,-.4]]);self.target=np.array([1.,-1.])
 def value(self,p):return self.a@p
 def linearize(self,p):return self.value(p),self.a.T@(self.value(p)-self.target)
 def jvp(self,p,v):return self.a@v

class Tests(unittest.TestCase):
 def test_coupled_exact_linear_length_and_budget(self):
  a=Linear();p=np.array([.3,.2,.1]);g=a.linearize(p)[1];eta=(g@g)/((a.a@g)@(a.a@g))
  one,info=r.solve(a,p,np.ones(3,bool),1);np.testing.assert_allclose(one,p-eta*g)
  three,more=r.solve(a,p,np.ones(3,bool),3)
  self.assertEqual(info['status'],'UPDATE_BUDGET');self.assertFalse(info['converged'])
  self.assertLessEqual(more['final_value'],info['final_value']);self.assertGreaterEqual(more['accepted_updates'],1)
 def test_inactive_coefficients_stay_at_scalar_initializer(self):
  p=np.array([.3,.2,.1]);value,info=r.solve(Linear(),p,np.array([True,False,True]),3)
  self.assertEqual(value[1],p[1]);self.assertLessEqual(info['final_value'],info['initial_value'])
  c=np.ones((1,3,32,32));mask=r.active_mask(c,True)
  self.assertEqual(int(mask.sum()),3*8*8);self.assertTrue(mask[...,:8,:8].all());self.assertFalse(mask[...,8:,:].any())
 def test_true_value_rejection_retains_last_iterate_and_records_both_trials(self):
  class Bad(Linear):
   def value(self,p):return np.array([100.,100.])
   def linearize(self,p):return np.zeros(2),np.ones(3)
  p=np.zeros(3);value,info=r.solve(Bad(),p,np.ones(3,bool),3)
  np.testing.assert_array_equal(value,p);self.assertEqual(info['status'],'TRIAL_BUDGET');self.assertEqual(len(info['trace']),2)
 def test_call_budget_and_cli_without_torch(self):
  b=r.budget();self.assertEqual(b['max_new_neural_forwards'],12576)
  self.assertEqual(b['per_suffix_forward_max'],dict(SCALAR_K0=59,DIAG_K1=175,DIAG_K3=407,COARSE_K3=407))
  self.assertEqual(b['max_new_neural_JVP'],2436)
  f=Path(__file__).with_name('diag_fast_response_refinement.py');v=subprocess.run([sys.executable,str(f),'--help'],capture_output=True,text=True)
  self.assertEqual(v.returncode,0,v.stderr);self.assertIn('--audit-only',v.stdout)
 def test_worker_reuses_original_model_and_checkpoint_guards_without_mutating_globals(self):
  from utils import diag_fast_response_refinement as d
  original_check,original_rules=d.old.check,d.old.rules
  def stub(root,index):self.assertIs(d.old.rules,r);self.assertIs(d.old.check,d.check)
  with patch.object(d.old,'worker',stub):d.worker(Path('/tmp'),0)
  self.assertIs(d.old.check,original_check);self.assertIs(d.old.rules,original_rules)
 def test_partial_summary_does_not_charge_cached_controls_as_new(self):
  from utils import diag_fast_response_refinement as d
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);case=dict(key='sample',task='super_resolution',image_id='00226');reference=root/'reference.json'
   row=dict(case=case,status='complete',arm='BASE',metrics=dict(psnr=25.,ssim=.8,lpips=.1),suffix_seconds=2.,counts=dict(forward_attempts=30,forwards=30))
   d.files.write(reference,row);case['a97_references']={'BASE':str(reference)}
   d.files.write(root/'plan.json',dict(experiment='A98_fast_response',sources={},inputs={},cases=[case],protocol={}))
   d.files.write(root/'cases'/'sample'/'SCALAR_K0'/'run.json',{**row,'arm':'SCALAR_K0','counts':dict(forward_attempts=59,forwards=59)})
   s=d.summarize(root);self.assertEqual(s['actual_new_cost']['forwards'],59);self.assertEqual(s['completed_runs'],1)

if __name__=='__main__':unittest.main()
