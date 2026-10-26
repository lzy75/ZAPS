"""A53 LOO leakage, bounded geometry and same-state A52 equivalence tests."""
import unittest
import numpy as np
from utils import angular_feedback_control as rules


def cohort():
    return [dict(task=t,image_id=i,key=t+'_'+i) for t in rules.TASKS for i in rules.IMAGES]


class NumpyTests(unittest.TestCase):
    def test_held_image_does_not_enter_own_time_control(self):
        cases=cohort();curves={c['key']:[.1+.1*rules.IMAGES.index(c['image_id'])]*30 for c in cases}
        old=rules.allocate(cases,curves);held=cases[0]['key'];curves[held]=[-.8]*30
        new=rules.allocate(cases,curves)
        self.assertEqual(old['assignments'][held]['TIME_COS'],new['assignments'][held]['TIME_COS'])
        self.assertNotIn(held,new['assignments'][held]['references'])
        for c in cases:
            a=new['assignments'][c['key']]
            self.assertNotEqual(a['donor'],c['key']);self.assertTrue(a['donor'].startswith(c['task']))
            self.assertEqual(a['NORM_ONLY'],[1.]*30)

    def test_missing_or_nonfinite_calibration_is_rejected(self):
        cases=cohort();curves={c['key']:[.5]*30 for c in cases}
        for bad in (None,float('nan'),1.2):
            curves[cases[0]['key']][0]=bad
            with self.assertRaises(ValueError):rules.allocate(cases,curves)

    def test_current_cosine_matches_a52_and_controls_preserve_current_norms(self):
        rng=np.random.default_rng(53)
        for _ in range(40):
            v,u=rng.normal(size=(2,9));k=.23;c=float(v@u/np.linalg.norm(v)/np.linalg.norm(u))
            target=rules.numpy_target(v,u,k,c)
            np.testing.assert_allclose(target,k*(v@u)/(u@u)*u,rtol=1e-12,atol=1e-12)
            for injected in (-1.,0.,.6,1.):
                controlled=rules.numpy_target(v,u,k,injected)
                self.assertAlmostEqual(float(np.linalg.norm(controlled)),k*np.linalg.norm(v)*abs(injected),places=12)


class TorchTests(unittest.TestCase):
    def test_same_state_factorization_and_signed_control_match_numpy(self):
        import torch
        from utils import current_score_runtime_r2 as runtime
        q=torch.tensor([[1.,.2]],dtype=torch.float64)
        v=torch.tensor([[1.,1.]],dtype=torch.float64);u=torch.tensor([[2.,1.]],dtype=torch.float64)
        reading=runtime.gram_readout(q,v,u,.5,1.)
        for c in (reading['cosine_v_u'],-.4,0.,1.):
            target,gate=rules.controlled_target(q,v,u,.5,reading,c)
            np.testing.assert_allclose(target.numpy(),rules.numpy_target(v.numpy(),u.numpy(),.5,c))
            self.assertLessEqual(gate['factorization_relative_error'],2e-6)


if __name__=='__main__':unittest.main()
