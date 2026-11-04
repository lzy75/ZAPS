"""Small local algebra/CLI gates; actual Torch gate is mandatory on server."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from utils.terminal_mask_counterfactual import parts,interventions
ROOT=Path(__file__).resolve().parents[1]

class Tests(unittest.TestCase):
    def test_partition_and_exact_measured_invariance(self):
        q=np.arange(12,dtype=float).reshape(3,4);mask=np.array([[0,1,0,1.]])
        m=q/10;qo,qu=parts(q,mask);out=interventions(m,q,mask)
        np.testing.assert_array_equal(qo+qu,q)
        np.testing.assert_array_equal(mask*qu,np.zeros_like(q))
        np.testing.assert_array_equal(mask*out['END_MISSING_RESTORE'],mask*m)
        np.testing.assert_array_equal(out['END_OBS_RESTORE']+out['END_MISSING_RESTORE']-m,out['END_BOTH_RESTORE'])
    def test_no_gt_or_nn_in_construction(self):
        import inspect
        self.assertNotIn('gt',inspect.signature(interventions).parameters)
        tree=ast.parse((ROOT/'utils/diag_terminal_mask_counterfactual.py').read_text())
        names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('backward',names);self.assertNotIn('grad',names);self.assertNotIn('_predict_eps',names)
    def test_real_cli(self):
        r=subprocess.run([sys.executable,str(ROOT/'utils/diag_terminal_mask_counterfactual.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--a71-root',r.stdout)

if __name__=='__main__':unittest.main()
