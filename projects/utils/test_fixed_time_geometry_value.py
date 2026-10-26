"""A59 analytic field/unit and same-image exclusion checks."""
import math
import unittest
import numpy as np
from utils import fixed_time_geometry_value as rules

class DesignTests(unittest.TestCase):
    def test_executor_restores_previous_worker(self):
        from utils import diag_fixed_time_geometry_value as d
        before=d.engine.check,d.engine.item,d.engine.sample
        with d.executor_bindings():self.assertIs(d.engine.sample,d.sample)
        self.assertEqual(before,(d.engine.check,d.engine.item,d.engine.sample))

    def test_rotation_field_has_known_directional_sensitivity(self):
        scales=(.001,.002)
        pairs=[(s,np.array([1.,s]),np.array([1.,-s])) for s in scales]
        result=rules.finite_geometry(np.array([1.,0.]),pairs)
        self.assertTrue(result['usable'])
        for scale,row in zip(scales,result['scales']):
            self.assertAlmostEqual(row['omega_per_input_RMS'],1/math.sqrt(1+scale*scale))
            self.assertAlmostEqual(row['kappa_per_Euclidean_input'],row['omega_per_input_RMS']/math.sqrt(2))
            self.assertAlmostEqual(row['relative_sensitivity'],1.)
            self.assertAlmostEqual(row['finite_scale_nonlinearity'],0.)
        self.assertEqual(rules.budget(),dict(replay=300,probes=120,continuation=340,total=760,per_case=76))

    def test_unresolved_rows_retained_and_same_image_excluded(self):
        rows=[]
        for i in range(5):
            for task in ('super_resolution','inpainting'):
                for step in rules.STEPS:
                    rows.append(dict(image_id=str(i),task=task,step=step,center_readout=dict(resolved=False),geometry=dict(usable=False),contrast={n:float(i) for n in ('psnr','ssim','lpips','loss')}))
        result=rules.analyze(rows)
        self.assertEqual(result['total_states'],30);self.assertEqual(result['usable_states'],0)
        first=result['predictions'][0]['values']['psnr']
        self.assertAlmostEqual(first['STAGE'],2.5)
        self.assertEqual(first['GEOMETRY'],first['PHYSICAL'])
        self.assertEqual(len(result['predictions']),30)

class TorchTests(unittest.TestCase):
    def test_observer_taps_one_actual_vjp_without_recomputing(self):
        import torch
        from unittest.mock import patch
        from utils import diag_fixed_time_geometry_value as d
        actual=d.current.algebra.input_vjp;calls=[]
        def source_field(z,y,x,t,s,positions,sigma,cosine,mask,backward):
            inp=x.detach().requires_grad_(True);raw=2*inp
            backward();u=d.current.algebra.input_vjp(raw,inp,torch.ones_like(raw))
            return dict(clean=raw.detach(),q=u),1.,dict(backward_calls=1)
        with patch.object(d.current,'sensed_field',source_field):
            field,row,u=d.observed_field(None,None,torch.ones(3),1,0,None,.05,None,lambda:calls.append(1))
        torch.testing.assert_close(u,torch.full((3,),2.))
        self.assertEqual(calls,[1]);self.assertIs(d.current.algebra.input_vjp,actual)

if __name__=='__main__':unittest.main()
