"""Check the observation geometries, native entry points, and fixed job grid."""
import unittest

import torch

from utils import diag_a145_temporal_panel as trial
from utils import dps_missing_tasks_runtime, dps_multiplier_transfer_runtime, pigdm_noisy_r1_runtime
from utils.pigdm_noisy_runtime import Operator


class TemporalPanelTests(unittest.TestCase):
    def test_mask_and_pool_projectors(self):
        for task in ('inpainting','super_resolution'):
            op = Operator(task,'cpu',size=16,seed=145)
            x = torch.zeros(1,3,16,16)
            counts = {}
            trial.native.initialize_counts(counts)
            geometry = trial.method.Geometry(task,op,x,counts)
            self.assertLessEqual(geometry.check(x)['null_relative_leakage'],2e-6)
            impulse = torch.randn(x.shape,generator=torch.Generator().manual_seed(145))
            torch.testing.assert_close(geometry.observable(geometry.observable(impulse)),
                                       geometry.observable(impulse),atol=1e-6,rtol=0)

    def test_resizer_uses_exact_native_rowspace(self):
        from utils.historical_ffhq_runtime import make_operator

        op = make_operator('cpu',size=32)
        x = torch.zeros(1,3,32,32)
        counts = {}
        trial.native.initialize_counts(counts)
        geometry = trial.method.Geometry('super_resolution',op,x,counts)
        check = geometry.check(x)
        self.assertLessEqual(check['null_relative_leakage'],2e-6)
        self.assertLessEqual(check['separable_H_relative_l2'],2e-6)

    def test_blur_uses_soft_complement(self):
        op = Operator('gaussian_deblur','cpu',size=64)
        x = torch.zeros(1,3,64,64)
        counts = {}
        trial.native.initialize_counts(counts)
        geometry = trial.method.Geometry('gaussian_deblur',op,x,counts)
        geometry.check(x)
        constant = torch.ones_like(x)
        torch.testing.assert_close(geometry.weak(constant),torch.zeros_like(x),atol=1e-6,rtol=0)
        checker = (torch.arange(64)%2-.5).expand_as(x)
        self.assertGreater(float(geometry.weak(checker).norm()),.9*float(checker.norm()))

    def test_all_sampler_hooks_compile(self):
        for host,runtime,num_active in (
            ('dps',dps_missing_tasks_runtime,100),('dps',dps_multiplier_transfer_runtime,100),
            ('pigdm',pigdm_noisy_r1_runtime,10)):
            _,active,warm = trial.schedule(host)
            fn = trial.native_instrument(runtime,host,active,warm,lambda *_:None,lambda *_:None)
            self.assertEqual(fn.__globals__['positions'],set(active))
            self.assertEqual(fn.__globals__['warm'],set(warm))
            self.assertEqual(len(active),num_active)
        _,active,warm = trial.schedule('zaps')
        fn = trial.zaps_sampler(None,None,'ZERO_WEAK',.05,active,warm,None,{},[])
        self.assertEqual(fn.__globals__['active'],set(active))
        self.assertEqual(len(active),5)


if __name__=='__main__':
    unittest.main()
