"""A95 exact nested scalar algebra, coupled solve and honest termination (no Torch)."""
import contextlib,io,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils import response_target_lm as r
from utils.noise_signal_shrink import select


class Linear:
 def __init__(self,B,target):self.B=B;self.target=target
 def value(self,p):return self.B@p
 def linearize(self,p):return self.value(p),self.B.T@(self.value(p)-self.target)
 def jvp(self,p,v):return self.B@v
 def vjp(self,p,v):return self.B.T@v


class Tests(unittest.TestCase):
 def test_original_scalar_is_exact_response_target_minimizer_and_scale_invariant(self):
  residual=np.array([1.,.2,-.3]);response=np.array([.4,.1,.2]);sigma=.05
  weight=r.noise_weight(float(residual@residual),3,sigma)
  expected=max(0.,float(response@(weight*residual))/(response@response))
  old=select(float(residual@response),float(residual@residual),float(response@response),3,sigma)
  self.assertAlmostEqual(old['gamma'],expected)
  scaled=select(float(residual@(7*response)),float(residual@residual),float((7*response)@(7*response)),3,sigma)
  self.assertAlmostEqual(scaled['gamma']*7,old['gamma'])
 def test_full_coupling_and_same_target_scalar_subset(self):
  B=np.array([[1.,.7],[.2,1.]]);target=np.array([.2,-.1]);unit=np.ones((2,1))/np.sqrt(2)
  scalar,si=r.lm(Linear(B@unit,target),np.zeros(1),lambda x:np.maximum(x,0))
  full,fi=r.lm(Linear(B,target),unit@scalar,grad_rtol=1e-6)
  self.assertTrue(si['converged']);self.assertTrue(fi['converged'])
  np.testing.assert_allclose(B@full,target,atol=2e-5)
  self.assertLessEqual(fi['final_value'],si['final_value']+1e-12)
  self.assertLess(full.min(),0)
  for t in fi['trace']:
   if t['accepted']:self.assertLess(t['new_value'],t['old_value'])
 def test_active_scalar_boundary_is_stationary(self):
  p,info=r.lm(Linear(np.eye(1),np.array([-1.])),np.zeros(1),lambda x:np.maximum(x,0))
  self.assertTrue(info['converged']);self.assertEqual(info['accepted_updates'],0);self.assertEqual(float(p[0]),0)
 def test_budget_is_not_convergence_and_cg_true_residual_is_checked(self):
  _,info=r.lm(Linear(np.eye(2),np.ones(2)),np.zeros(2),max_outer=0)
  self.assertEqual(info['status'],'OUTER_BUDGET');self.assertFalse(info['converged'])
  with self.assertRaises(RuntimeError):r.norm(np.array([np.nan]))
  B=np.array([[3.,1.],[1.,2.]])
  x,cg=r.cg(lambda v:B@v,np.array([1.,2.]),.3,maxiter=4,rtol=1e-8)
  self.assertTrue(cg['passed']);np.testing.assert_allclose((B+.3*np.eye(2))@x,[1,2],atol=1e-10)
 def test_cli_source_bindings_and_summary_do_not_charge_reused_controls(self):
  project=Path(__file__).resolve().parents[1]
  result=subprocess.run([sys.executable,'utils/diag_response_target_lm.py','--help'],cwd=project,capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
  from utils import diag_response_target_lm as driver
  self.assertTrue(all((project/n).is_file() for n in driver.NAMES))
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);case=dict(key='one',task='inpainting',image_id='00226',references={})
   for arm,value in [('REAL_SCALAR',25.),('REAL_DIAG',26.)]:
    driver.files.write(root/'cases/one'/arm/'run.json',dict(status='complete',case=case,arm=arm,metrics=dict(psnr=value,ssim=.8,lpips=.2),counts=dict(forward_attempts=20),suffix_seconds=1.,
     detail=dict(solver=dict(status='GRADIENT_TOLERANCE',converged=True),actual_target_objective=1.,target=dict(omega=.5))))
   with patch.object(driver,'check',return_value=dict(experiment='A95',protocol={})),contextlib.redirect_stdout(io.StringIO()):s=driver.summarize(root)
   self.assertEqual(s['actual_new_cost']['forward_attempts'],40)
   self.assertEqual(r.budget()['max_new_F'],55128)

if __name__=='__main__':unittest.main()
