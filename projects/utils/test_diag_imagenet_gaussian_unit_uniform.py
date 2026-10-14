"""Paired schedule contracts; tensor tests use tiny score, never pretrained UNet."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import diag_imagenet_gaussian_unit_uniform as audit
from utils import test_diag_imagenet_original_reference as originals


class ContractTests(unittest.TestCase):
    def parent(self):
        return dict(experiment="imagenet_gaussian_unit_measurement_original_settings",status="complete",new_nfe=300,
            diagnostic_nfe=0,optimizer_steps=10,config=originals.ContractTests().config(),
            task_config=dict(kernel_size=61,sigma=3.,noise_sigma=.05),
            resolved_protocol=dict(model_state_domain=[-1,1],measurement_domain=[0,1],configured_sigma=.05,
                operator="H((x+1)/2)",jacobian_adjoint=".5 H^T(r)",normalized_noise="from archived y-H(gt), no redraw",
                observation_clipped=False,state=False,final_mode="last_opt_pre_final_adam_update",
                physical_noise_changed=True,new_final_sample=False),
            gates={k:dict(passed=True) for k in ("identity","affine_measurement","cached_metrics_loss","noise_preflight","training_pairing")},
            loss_history=[.01]*10,gradients=[dict(joint_norm=.1)]*10,
            results=[dict(policy="cached_original_model_domain"),dict(policy="unit_protocol_original_initialization",
                new_nfe=300,loss_domain="unit",psnr=21.7256,ssim=.5387,lpips=.3958,final_mse=.01,final_residual=20.)])

    def test_requires_completed_original_unit_parent_and_all_gates(self):
        self.assertAlmostEqual(audit.validate_parent(self.parent())["psnr"],21.7256)
        for field,value in (("status","failed"),("new_nfe",330),("optimizer_steps",9)):
            other=self.parent();other[field]=value
            with self.assertRaises(ValueError):audit.validate_parent(other)
        for key in self.parent()["gates"]:
            other=self.parent();other["gates"][key]["passed"]=False
            with self.assertRaises(ValueError):audit.validate_parent(other)
        for field,value in (("lr",.05),("zeta_init",.04),("eta",.75)):
            other=self.parent();other["config"][field]=value
            with self.assertRaises(ValueError):audit.validate_parent(other)
        for field,value in (("state",True),("jacobian_adjoint","H^T(r)"),("configured_sigma",.025)):
            other=self.parent();other["resolved_protocol"][field]=value
            with self.assertRaises(ValueError):audit.validate_parent(other)

    def test_exact_grid_matches_historical_uniform(self):
        expected=[0,34,69,103,138,172,207,241,276,310,344,379,413,448,482,
            517,551,586,620,655,689,723,758,792,827,861,896,930,965,999]
        self.assertEqual(audit.uniform_grid(),expected)
        self.assertEqual(len(set(audit.uniform_grid())),30)
        self.assertEqual(audit.uniform_grid(),[round(999*i/29) for i in range(30)])

    def test_runtime_mismatch_stops_but_gpu_count_mapping_is_separate(self):
        runtime={k:False for k in audit.RUNTIME_KEYS}
        runtime.update(torch="test",cuda="test",gpu_name="test",visible_cuda_count=4)
        current=copy.deepcopy(runtime);current["visible_cuda_count"]=1
        self.assertTrue(audit.check_runtime(runtime,current)["passed"])
        current["matmul_allow_tf32"]=True
        self.assertFalse(audit.check_runtime(runtime,current)["passed"])
        current.pop("gpu_name")
        self.assertIn("gpu_name",audit.check_runtime(runtime,current)["differences"])

    def test_delta_is_always_irregular_minus_uniform_and_no_parity_claim(self):
        uniform=dict(psnr=21.,ssim=.5,lpips=.4)
        for paper,label in ((dict(psnr=21.3,ssim=.51,lpips=.39),"IRREGULAR_PSNR_ADVANTAGE_WITHOUT_MATERIAL_AUXILIARY_REGRESSION"),
                (dict(psnr=20.8,ssim=.48,lpips=.43),"UNIFORM_PSNR_ADVANTAGE_PERSISTS"),
                (dict(psnr=21.04,ssim=.5,lpips=.4),"PSNR_CLOSE_ON_THIS_IMAGE"),
                (dict(psnr=21.3,ssim=.48,lpips=.43),"MIXED_PSNR_AND_PERCEPTUAL_TRADEOFF")):
            result=audit.interpret(paper,uniform)
            self.assertEqual(result["classification"],label)
            self.assertAlmostEqual(result["irregular_minus_uniform"]["psnr"],paper["psnr"]-21)
            self.assertFalse(result["author_parity"] or result["benchmark_reproduced"] or result["state_gain"])

    def test_one_training_call_no_observation_regeneration_or_extra_sample(self):
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        optimize=[n for n in calls if getattr(n.func,"attr",None)=="optimize"]
        self.assertEqual(len(optimize),1)
        self.assertTrue(any(k.arg=="x0_gt" and isinstance(k.value,ast.Constant) and k.value.value is None
                            for k in optimize[0].keywords))
        forbidden={"sample","p_sample_loop","measurement_and_gates","randn","randn_like"}
        names={getattr(n.func,"attr",getattr(n.func,"id",None)) for n in calls}
        self.assertFalse(names & forbidden)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable; operator and tiny-training tests NOT run")
class TensorTests(unittest.TestCase):
    def test_operator_gate_and_corrupted_adjoint_without_rng(self):
        import torch
        from modules.degradations import GaussianBlurOperator
        base=GaussianBlurOperator(kernel_size=3,sigma=1.,noise_sigma=.05)
        operator=audit.unit_operator(base)
        x=torch.linspace(-1,1,64).reshape(1,1,8,8)
        direction=torch.cos(torch.arange(64).float()).reshape_as(x)
        before=torch.get_rng_state().clone()
        gate=audit.operator_gate(operator,x,direction)
        self.assertTrue(gate["passed"],gate)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with patch.object(operator,"transpose",side_effect=lambda r:base.transpose(r)):
            self.assertFalse(audit.operator_gate(operator,x,direction)["passed"])

    def test_actual_core_grid_only_change_keeps_rng_and_300_call_budget(self):
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
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1))
            def forward(self,x,t):return self.weight*x
        score=Score()
        dm=SimpleNamespace(device="cpu",num_steps=1000,model=score,
            alphas_cumprod=torch.linspace(.999,.5,1000),_predict_eps=score)
        config=originals.ContractTests().config()
        snapshots=[]
        for grid in (None,audit.uniform_grid()):
            with patch.object(core,"OrthogonalDWT2D",DWT):
                zaps=LastUnrollSnapshotZAPS(dm,audit.unit_operator(Base()),channels=1,img_size=4,**config)
            if grid is not None:zaps.tau=torch.tensor(grid,dtype=torch.long)
            self.assertTrue(torch.equal(zaps.zeta,torch.full_like(zaps.zeta,.2)))
            self.assertTrue(torch.equal(zaps.D,torch.full_like(zaps.D,.2)))
            visited=[]
            hook=score.register_forward_pre_hook(lambda m,args:visited.append(int(args[1][0])))
            torch.manual_seed(1001)
            try:
                with model_rng_guard(score,"cpu") as calls:
                    losses=zaps.optimize(torch.full((1,1,4,4),.6),verbose=False,x0_gt=None)
            finally:hook.remove()
            self.assertEqual(visited,zaps.tau.flip(0).tolist()*10)
            self.assertEqual(len(calls),300)
            self.assertEqual(zaps._last_nfe,300)
            self.assertEqual(len(losses),10)
            self.assertTrue(torch.isfinite(zaps._last_opt_x0).all())
            self.assertIsNone(score.weight.grad)
            snapshots.append((zaps.audit_state,torch.get_rng_state().clone()))
        self.assertTrue(torch.equal(snapshots[0][0]["init_noise"],snapshots[1][0]["init_noise"]))
        self.assertTrue(torch.equal(snapshots[0][0]["cpu_rng"],snapshots[1][0]["cpu_rng"]))
        self.assertTrue(torch.equal(snapshots[0][1],snapshots[1][1]))
        self.assertFalse(torch.equal(snapshots[0][0]["tau"],snapshots[1][0]["tau"]))


if __name__=="__main__":unittest.main()
