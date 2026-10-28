"""Mandatory server Torch gates; no skip on missing Torch."""
import unittest
import torch
from utils import counted_adjoint_repair as repair
from utils import historical_ffhq_runtime as engine
from utils import paper_clock_runtime as common
from utils.test_paper_clock_runtime import TinyDiffusion
from utils.current_score_runtime_r2 import input_vjp_parameters


class NumericTests(unittest.TestCase):
    def test_real_counted_exact_adjoint_default_and_explicit_size_preserve_gradient(self):
        ref=engine.make_operator('cpu',size=16,mode='exact')
        count=dict(base_H_actions=0,base_adjoint_actions=0)
        with repair.installed(): counted=engine.make_operator('cpu',count,size=16,mode='exact')
        y=torch.arange(48,dtype=torch.float32).reshape(1,3,4,4)/48
        for size in (None,(16,16)):
            a=y.clone().requires_grad_();b=y.clone().requires_grad_()
            first=ref.transpose(a,output_size=size);second=counted.transpose(b,output_size=size)
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(first.square().sum(),a)[0],
                                       torch.autograd.grad(second.square().sum(),b)[0],rtol=0,atol=0)
        self.assertEqual(count['guidance_mapping_actions'],2)
        self.assertEqual(count['base_adjoint_actions'],2)

    def test_all_five_arms_use_production_counted_constructor_end_to_end(self):
        for arm in engine.rules.ARMS:
            with self.subTest(arm=arm):
                model=TinyDiffusion()
                with repair.installed(),input_vjp_parameters(model.model),common.counters(model.model) as count:
                    op=engine.make_operator('cpu',count,size=16,mode='exact' if arm=='EXACT_IEEE' else 'legacy_bicubic')
                    out,details=engine.execute(model,op,torch.full((1,3,4,4),.15),1003,arm,count,size=16,level=1)
                engine.validate_counts(count,arm)
                self.assertTrue(torch.isfinite(out).all());self.assertIsNone(model.model.weight.grad)
                self.assertGreater(count['guidance_mapping_actions'],0)
                if arm=='EXACT_IEEE':self.assertEqual(count['base_adjoint_actions'],count['guidance_mapping_actions'])


if __name__=='__main__':unittest.main()
