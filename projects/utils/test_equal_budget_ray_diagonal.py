import subprocess,sys,unittest
from pathlib import Path
import numpy as np
from utils import equal_budget_ray_diagonal as r


class Linear:
    def __init__(self):
        self.a=np.array([[2.,1.],[.5,3.]])
        self.target=np.array([1.,-.5])
        self.calls=dict(linearize=0,jvp=0,value=0)
    def value(self,p):self.calls['value']+=1;return self.a@p
    def linearize(self,p):
        self.calls['linearize']+=1;v=self.a@p
        return v,self.a.T@(v-self.target)
    def jvp(self,p,v):self.calls['jvp']+=1;return self.a@v


class Tests(unittest.TestCase):
    def test_fixed_call_solver_accepts_descent(self):
        provider=Linear();p=np.array([.2,.1])
        result,info=r.solve(provider,p,lambda x:x)
        self.assertLess(info['final_value'],info['initial_value'])
        self.assertEqual(provider.calls,dict(linearize=1,jvp=1,value=2))
        self.assertEqual(len(info['trace']),2)
    def test_projection_can_restrict_to_nonnegative_ray(self):
        provider=Linear();p=np.array([.2]);provider.a=np.array([[2.],[.5]])
        result,info=r.solve(provider,p,lambda x:np.maximum(x,0))
        self.assertGreaterEqual(result.item(),0);self.assertEqual(provider.calls['value'],2)
    def test_exact_budget_and_cli_without_torch(self):
        b=r.budget();self.assertEqual(b['per_suffix_forward_exact'],175)
        self.assertEqual(b['max_new_neural_forwards'],4200)
        path=Path(__file__).with_name('diag_equal_budget_ray_diagonal.py')
        out=subprocess.run([sys.executable,str(path),'--help'],capture_output=True,text=True)
        self.assertEqual(out.returncode,0,out.stderr)


if __name__=='__main__':unittest.main()
