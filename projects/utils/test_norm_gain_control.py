"""A54 gain allocation leakage and actual Torch normalization checks, no skips."""
import unittest
import numpy as np
from utils import norm_gain_control as rules


def cohort():return [dict(task=t,image_id=i,key=t+'_'+i) for t in rules.TASKS for i in rules.IMAGES]


class NumpyTests(unittest.TestCase):
    def test_executor_bindings_are_process_local_and_restored(self):
        from utils import diag_norm_gain_value as driver
        engine=driver.engine;before=(engine.rules,engine.item,engine.check)
        with driver.executor_bindings():
            self.assertIs(engine.rules,rules);self.assertIs(engine.item,driver.item);self.assertIs(engine.check,driver.check)
        self.assertEqual(before,(engine.rules,engine.item,engine.check))

    def test_held_image_excluded_from_time_and_constant(self):
        cases=cohort();curves={c['key']:[.1+.1*rules.IMAGES.index(c['image_id'])+.01*k for k in range(30)] for c in cases}
        a=rules.allocate(cases,curves);held=cases[0]['key'];curves[held]=[30.]*30;b=rules.allocate(cases,curves)
        for name in ('TIME_GAIN','CONST_GAIN'):self.assertEqual(a['assignments'][held][name],b['assignments'][held][name])
        self.assertNotIn(held,b['assignments'][held]['references'])
        self.assertNotEqual(held,b['assignments'][held]['donor'])
        self.assertEqual(len(set(b['assignments'][held]['CONST_GAIN'])),1)

    def test_invalid_gains_rejected(self):
        cases=cohort()
        for bad in (0.,-1.,float('nan')):
            curves={c['key']:[1.]*30 for c in cases};curves[cases[0]['key']][0]=bad
            with self.assertRaises(ValueError):rules.allocate(cases,curves)

    def test_current_gain_preserves_physical_target_norm(self):
        rng=np.random.default_rng(54)
        for _ in range(40):
            v,u=rng.normal(size=(2,11));k=.31;gain=np.linalg.norm(v)/np.linalg.norm(u)
            target=rules.numpy_target(u,k,gain)
            self.assertAlmostEqual(float(np.linalg.norm(target)),k*np.linalg.norm(v),places=12)
            self.assertAlmostEqual(float(np.linalg.norm(rules.numpy_target(u,k,2*gain))),2*k*np.linalg.norm(v),places=12)


class TorchTests(unittest.TestCase):
    def test_current_gain_matches_a53_and_foreign_gain_is_not_clipped(self):
        import torch
        from utils import current_score_runtime_r2 as runtime
        q=torch.tensor([[1.,.2]],dtype=torch.float64);v=torch.tensor([[1.,1.]],dtype=torch.float64)
        u=torch.tensor([[2.,1.]],dtype=torch.float64);r=runtime.gram_readout(q,v,u,.5,1.)
        g=r['v_RMS']/r['u_RMS']
        for used in (g,2*g,.2*g):
            out,gate=rules.controlled_target(q,v,u,.5,r,used)
            np.testing.assert_allclose(out.numpy(),rules.numpy_target(u.numpy(),.5,used))
            self.assertAlmostEqual(gate['relative_dose_to_current'],used/g,places=12)
            self.assertFalse(gate['cosine_used_for_action'])
        torch.testing.assert_close(q,torch.tensor([[1.,.2]],dtype=torch.float64))


if __name__=='__main__':unittest.main()
