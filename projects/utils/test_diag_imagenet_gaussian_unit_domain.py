"""Independent protocol contracts; optional tiny tensor tests, no pretrained UNet."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import diag_imagenet_gaussian_unit_domain as audit
from utils import test_diag_imagenet_original_reference as original_tests


class ContractTests(unittest.TestCase):
    def reference(self):
        return dict(experiment="original_imagenet_reference_cached_audit",status="complete",
            method_nfe=0,diagnostic_nfe=0,model_loaded=False,config=original_tests.ContractTests().config(),
            gates={k:dict(passed=True) for k in ("source_identity","cached_mechanics","cached_output_metrics_and_loss")},
            baseline=dict(task="gaussian_deblur",seed=1001,historical_method_nfe=300,new_method_nfe=0),
            measurement_domain=dict(model_domain=[-1,1],configured_sigma=.05),decision=dict(author_parity=False))

    def test_only_completed_original_reference_is_admitted(self):
        audit.validate_reference(self.reference())
        for field,value in (("status","failed"),("method_nfe",300),("model_loaded",True)):
            record=self.reference();record[field]=value
            with self.assertRaises(ValueError):audit.validate_reference(record)
        for key,value in (("lr",.05),("zeta_init",.0400298),("eta",.75),("use_learned_var",True)):
            record=self.reference();record["config"][key]=value
            with self.assertRaises(ValueError):audit.validate_reference(record)
        record=self.reference();record["gates"]["cached_mechanics"]["passed"]=False
        with self.assertRaises(ValueError):audit.validate_reference(record)

    def test_multimetric_decision_never_certifies_author_or_state_parity(self):
        old=dict(psnr=19.6,ssim=.3168,lpips=.6505)
        for candidate,label in ((dict(psnr=20.,ssim=.33,lpips=.63),"SUPPORTS_GAUSSIAN_MEASUREMENT_PROTOCOL_SENSITIVITY"),
                (dict(psnr=18.,ssim=.3,lpips=.8),"NO_COMPREHENSIVE_QUALITY_SUPPORT_FOR_UNIT_PROTOCOL"),
                (dict(psnr=24.,ssim=.6,lpips=.7),"MIXED_OR_INCONCLUSIVE")):
            result=audit.interpret(candidate,old)
            self.assertEqual(result["classification"],label)
            for key in ("author_parity","benchmark_reproduced","state_gain","original_reference_overwritten"):
                self.assertFalse(result[key])

    def test_one_training_call_no_extra_sampling_no_gt_selection(self):
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        training=[n for n in calls if getattr(n.func,"attr",None)=="optimize"]
        self.assertEqual(len(training),1)
        self.assertTrue(any(k.arg=="x0_gt" and isinstance(k.value,ast.Constant) and k.value.value is None
                            for k in training[0].keywords))
        self.assertFalse(any(getattr(n.func,"attr",None) in ("sample","p_sample_loop","_reverse_diffusion") for n in calls))


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable; affine/VJP/tiny-training tests NOT run")
class TensorTests(unittest.TestCase):
    def test_nonsymmetric_adjoint_and_no_output_size_api(self):
        import torch
        class Matrix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("matrix",torch.tensor([[1.,2.],[0.,3.]],dtype=torch.float64))
            def H(self,x):return x@self.matrix.T
            def transpose(self,r):return r@self.matrix
            def forward(self,x):raise AssertionError("No stochastic base forward allowed")
        base=Matrix();operator=audit.unit_operator(base)
        x=torch.tensor([[.2,-.4]],dtype=torch.float64,requires_grad=True)
        r=torch.tensor([[.7,-.5]],dtype=torch.float64)
        torch.testing.assert_close(operator(x),base.H((x+1)/2))
        grad=torch.autograd.grad((operator(x)*r).sum(),x)[0]
        torch.testing.assert_close(operator.transpose(r),grad)
        self.assertFalse(torch.allclose(operator.transpose(r),.5*base.H(r)))

    def test_zero_padding_offset_noise_field_and_gates_without_rng(self):
        import torch
        from modules.degradations import GaussianBlurOperator
        base=GaussianBlurOperator(kernel_size=3,sigma=1.,noise_sigma=.05)
        operator=audit.unit_operator(base)
        gt=torch.linspace(-1,1,64).reshape(1,1,8,8)
        direction=torch.cos(torch.arange(64).float()).reshape_as(gt)
        noise=.05*torch.sin(torch.arange(64).float()).reshape_as(gt)
        old_y=base.H(gt)+noise
        before=torch.get_rng_state().clone()
        y,actual_noise,standard,gate=audit.measurement_and_gates(base,operator,gt,old_y,direction)
        self.assertTrue(gate["passed"],gate)
        self.assertGreater(gate["zero_padding_H1_max_difference_from_one"],.1)
        self.assertEqual(gate["score_nfe"],0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(actual_noise,noise,atol=1e-7,rtol=2e-5)
        torch.testing.assert_close(standard*.05,actual_noise)
        torch.testing.assert_close(y,base.H((gt+1)/2)+actual_noise)
        self.assertFalse(torch.allclose(y,(old_y+1)/2))
        with patch.object(operator,"transpose",side_effect=lambda r:base.transpose(r)):
            _,_,_,bad=audit.measurement_and_gates(base,operator,gt,old_y,direction)
        self.assertFalse(bad["passed"])

    def test_actual_core_300_call_original_training_contract(self):
        import torch
        from modules import zaps_algorithm as core
        from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS
        from utils.diag_zaps_first_direction import model_rng_guard
        class Base(torch.nn.Module):
            def H(self,x):return x
            def transpose(self,r):return r
        class DWT(torch.nn.Module):
            def __init__(self,*args,**kwargs):super().__init__()
            def analysis(self,x):return x
            def synthesis(self,x):return x
        class Score(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1))
            def forward(self,x,t):return self.weight*x
        score=Score()
        dm=SimpleNamespace(device="cpu",num_steps=1000,model=score,
            alphas_cumprod=torch.linspace(.999,.5,1000),_predict_eps=score)
        config=original_tests.ContractTests().config()
        with patch.object(core,"OrthogonalDWT2D",DWT):
            zaps=LastUnrollSnapshotZAPS(dm,audit.unit_operator(Base()),channels=1,img_size=4,**config)
        self.assertTrue(torch.equal(zaps.zeta,torch.full_like(zaps.zeta,.2)))
        self.assertTrue(torch.equal(zaps.D,torch.full_like(zaps.D,.2)))
        y=torch.full((1,1,4,4),.6)
        visited=[]
        hook=score.register_forward_pre_hook(lambda m,args:visited.append(int(args[1][0])))
        torch.manual_seed(1001)
        try:
            with model_rng_guard(score,"cpu") as calls:
                losses=zaps.optimize(y,verbose=False,x0_gt=None)
        finally:hook.remove()
        tail=torch.get_rng_state().clone()
        torch.manual_seed(1001)
        noise=torch.randn(1,1,4,4)
        for _ in range(290):torch.randn_like(noise)
        self.assertTrue(torch.equal(tail,torch.get_rng_state()))
        self.assertEqual(len(losses),10)
        self.assertEqual(zaps.audit_calls,10)
        self.assertEqual(len(calls),300)
        self.assertEqual(zaps._last_nfe,300)
        self.assertEqual(visited,zaps.tau.flip(0).tolist()*10)
        self.assertTrue(torch.equal(zaps.audit_state["init_noise"],noise))
        self.assertTrue(torch.isfinite(zaps._last_opt_x0).all())
        self.assertIsNone(score.weight.grad)
        self.assertIsNotNone(zaps.zeta.grad)
        self.assertIsNotNone(zaps.D.grad)


if __name__=="__main__":unittest.main()
