"""A60 refresh budget and real no-hidden-VJP check; no missing-Torch skips."""
import math
import unittest
from utils import direction_refresh_control as rules

class DesignTests(unittest.TestCase):
    def test_first_last_mandatory_and_budget(self):
        self.assertEqual([len(rules.refresh_steps(i)) for i in rules.INTERVALS],[30,16,9])
        self.assertEqual(10*sum(len(rules.refresh_steps(i)) for i in rules.INTERVALS),550)
        for i in rules.INTERVALS:
            self.assertEqual(rules.refresh_steps(i)[0],0);self.assertEqual(rules.refresh_steps(i)[-1],29)
        with self.assertRaises(ValueError):rules.refresh_steps(3)

    def test_worker_bindings_are_restored(self):
        from utils import diag_direction_refresh_value as d
        before=d.engine.check,d.engine.item,d.engine.sample
        with d.executor_bindings():self.assertIs(d.engine.sample,d.sample)
        self.assertEqual(before,(d.engine.check,d.engine.item,d.engine.sample))

class TorchTests(unittest.TestCase):
    def test_stale_step_recomputes_v_but_forbids_current_u(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        current=rules.current
        dm=SimpleNamespace(alphas_cumprod=torch.tensor([.9,.5]),_predict_eps=lambda x,t:torch.zeros_like(x))
        op=SimpleNamespace(base=SimpleNamespace(H=lambda x:x),H=lambda x:.5*(x+1),transpose=lambda x:.5*x)
        z=SimpleNamespace(dm=dm,A=op);x=torch.tensor([[1.,2.]]);y=torch.zeros_like(x)
        cached=torch.tensor([[3.,4.]]);cache=dict(u=cached,u_RMS=math.sqrt(12.5),step=0,t=2)
        def field(z,y,raw,t,p):return dict(clean=raw.detach().clamp(-1,1),q=torch.ones_like(raw))
        with patch.object(current.base,'field_from_mean',field),patch.object(current.algebra,'input_vjp',side_effect=AssertionError('Hidden current VJP')):
            f,g,row,newcache=rules.sensed_field(z,y,x,1,0,None,.05,1,2,cache,before_backward=lambda:(_ for _ in ()).throw(AssertionError('Hidden backward callback')))
        self.assertEqual(row['backward_calls'],0);self.assertFalse(row['refresh_due'])
        self.assertTrue(row['cg']['passed']);self.assertTrue(row['resolved']);self.assertIs(newcache,cache)
        self.assertLessEqual(row['norm_rule_relative_error'],2e-6)
        expected=row['coefficient']*row['v_RMS']*math.sqrt(x.numel())
        self.assertAlmostEqual(float(f['q'].double().norm()),expected,delta=expected*2e-6)

if __name__=='__main__':unittest.main()
