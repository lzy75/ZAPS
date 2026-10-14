"""State/domain contracts and optional actual-core toy tests, no pretrained model."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from utils import diag_state_domain_paired as audit
from utils import test_diag_imagenet_tuned_state as previous


class ContractTests(unittest.TestCase):
    def test_live_rng_schema_is_translated_for_cuda_restore(self):
        live=dict(cpu="cpu-state",cuda=["sampling-state","other-state"])
        saved=audit.saved_rng_view(live)
        self.assertEqual(saved,dict(cpu_rng="cpu-state",cuda_rng=["sampling-state","other-state"]))
        self.assertIs(saved["cuda_rng"],live["cuda"])

    def test_original_initialization_and_frozen_per_dataset_config(self):
        for dataset,lr,zeta in (("imagenet",.001,.2),("ffhq",.01,.1)):
            c=audit.base_config(dataset)
            self.assertEqual((c["lr"],c["zeta_init"],c["d_init"]),(lr,zeta,.2))
            self.assertEqual((c["num_steps"],c["num_epochs"]),(30,10))
            self.assertFalse(c["use_learned_var"] or c["surrogate_score_jacobian"])
        with self.assertRaises(ValueError):audit.base_config("unknown")

    def test_factorial_has_no_cross_domain_state_comparison(self):
        rows=[];arms={}
        for domain,policy,response in audit.arms_for("ffhq"):
            p=20+(2 if domain=="unit" else 0)+(.3 if domain=="unit" else .2)*(policy=="state")
            rows.append(dict(domain=domain,policy=policy,psnr=p,ssim=.5,lpips=.4))
            arms[domain+"_"+policy]=dict(changed_points_after_pilot=20)
        result=audit.summarize(rows,arms)
        self.assertAlmostEqual(result["unit_minus_model_null"]["psnr"],2.)
        self.assertAlmostEqual(result["state_by_domain"]["model"]["state_minus_null"]["psnr"],.2)
        self.assertAlmostEqual(result["state_by_domain"]["unit"]["state_minus_null"]["psnr"],.3)
        self.assertAlmostEqual(result["state_domain_interaction"]["psnr"],.1)
        self.assertEqual(result["domain_null_quality_classification"],"UNIT_PROTOCOL_QUALITY_ADVANTAGE_ON_THIS_IMAGE")
        self.assertEqual(len(audit.arms_for("imagenet")),2)

    def test_quality_tradeoff_drift_and_nonactivation_are_not_hidden(self):
        null=dict(psnr=21.7,ssim=.54,lpips=.4);state=dict(psnr=21.9,ssim=.54,lpips=.4)
        self.assertEqual(audit.assess_state(null,state,10,null)["classification"],"SINGLE_IMAGE_STATE_GAIN_CANDIDATE")
        self.assertEqual(audit.assess_state(null,state,0,null)["classification"],"STATE_NOT_ACTIVATED")
        self.assertEqual(audit.assess_state(null,state,10,dict(null,psnr=21.5))["classification"],"INCONCLUSIVE_NULL_DRIFT")
        self.assertEqual(audit.assess_state(null,dict(state,lpips=.41),10)["classification"],"PSNR_ONLY_TRADEOFF")
        self.assertEqual(audit.assess_state(null,dict(state,psnr=21.6),10)["classification"],"NO_CLEAR_COMPREHENSIVE_STATE_GAIN")

    def test_single_training_entry_no_extra_sample_and_no_gt_passed(self):
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        training=[n for n in calls if getattr(n.func,"attr",None)=="optimize"]
        self.assertEqual(len(training),1)
        self.assertTrue(any(k.arg=="x0_gt" and isinstance(k.value,ast.Constant) and k.value.value is None
                            for k in training[0].keywords))
        self.assertFalse(any(getattr(n.func,"attr",None) in ("sample","p_sample_loop") for n in calls))
        self.assertEqual(sum(300 for _ in audit.arms_for("ffhq")),1200)
        self.assertEqual(sum(300 for _ in audit.arms_for("imagenet")),600)

    def test_frozen_controller_null_pilot_and_two_indicators(self):
        grid=previous.SchedulerTests.grid
        for dataset in ("imagenet","ffhq"):
            for response in (0.,.2):
                s=audit.scheduler_for(dataset,grid,previous.alpha_table(),response)
                self.assertEqual(s.cfg.profile_gate_mode,"veto_only")
                self.assertEqual(s.cfg.weight_mode,"identity")
                self.assertEqual(s.cfg.profile_cosine_gate,.25 if dataset=="imagenet" else .35)
                for epoch in range(2):
                    s.reset();t=999;visited=[]
                    while not s.done():
                        visited.append(t)
                        if hasattr(s,"set_time"):s.set_time(t)
                        s.update_state(math.exp(2.-t/1000)+epoch*.1,.7 if s.used>=2 else float("nan"))
                        h=s.select_step(t)
                        if not s.done():t-=h
                    self.assertEqual(len(visited),30)
                    self.assertEqual((visited[0],visited[-1]),(999,0))
                    self.assertTrue(all(a>b for a,b in zip(visited,visited[1:])))
                    if response==0. or epoch==0:self.assertEqual(visited,grid)
        with self.assertRaises(ValueError):audit.scheduler_for("imagenet",grid,previous.alpha_table(),.5)
        with self.assertRaises(ValueError):audit.scheduler_for("unknown",grid,previous.alpha_table(),.2)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable; actual-core forward/VJP/training tests NOT run")
class TensorTests(unittest.TestCase):
    def setup_case(self,dataset="imagenet"):
        import torch
        from modules import zaps_algorithm as core
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
        score=Score();alpha=torch.tensor(previous.alpha_table())
        dm=SimpleNamespace(device="cpu",num_steps=1000,model=score,alphas_cumprod=alpha,_predict_eps=score)
        operator=audit.unit_operator(Base());cls=audit.observed_class()
        def construct():
            with patch.object(core,"OrthogonalDWT2D",DWT):
                z=cls(dm,operator,channels=1,img_size=2,**audit.base_config(dataset))
            z.tau=torch.tensor(previous.SchedulerTests.grid[::-1]);return z
        def scheduler(z,response):return audit.scheduler_for(dataset,z.tau.flip(0).tolist(),alpha.tolist(),response)
        return construct,scheduler,dm

    def test_actual_core_both_controllers_pass_fixed_repeat_null_vjp_gate(self):
        import torch
        from utils.diag_zaps_sampler_parity import rng_state
        for dataset in ("imagenet","ffhq"):
            construct,scheduler,dm=self.setup_case(dataset)
            torch.manual_seed(1001);initial=torch.randn(1,1,2,2);rng=rng_state("cpu")
            record=dict(diagnostic_nfe=0,planned_diagnostic_nfe=90)
            gate=audit.gate_unroll(construct,scheduler,torch.full_like(initial,.6),initial,rng,"cpu",record)
            self.assertTrue(gate["passed"],gate)
            self.assertEqual(record["diagnostic_nfe"],90)
            self.assertIsNone(dm.model.weight.grad)

    def test_passive_all_epoch_trace_preserves_null_training_and_rng(self):
        import torch
        from utils.diag_zaps_sampler_parity import rng_state,same_rng
        construct,scheduler,dm=self.setup_case()
        a,b=construct(),construct();y=torch.full((1,1,2,2),.6);s=scheduler(b,0.)
        hook=dm.model.register_forward_pre_hook(lambda m,inputs:s.set_time(int(inputs[1][0])))
        try:
            torch.manual_seed(1001);loss_a=a.optimize(y,verbose=False,x0_gt=None);tail=rng_state("cpu")
            torch.manual_seed(1001);loss_b=b.optimize(y,verbose=False,x0_gt=None,scheduler=s)
        finally:hook.remove()
        self.assertEqual(loss_a,loss_b)
        self.assertTrue(same_rng(tail,rng_state("cpu")))
        self.assertTrue(torch.equal(a._last_opt_x0,b._last_opt_x0))
        self.assertEqual(b._last_nfe,300);self.assertEqual(len(b.audit_trace),300)
        self.assertEqual([r["parameter_index"] for r in b.audit_trace],list(range(29,-1,-1))*10)
        self.assertEqual(b.audit_state["scheduler_used"],30)
        self.assertIsNone(dm.model.weight.grad)
        self.assertFalse(torch.equal(b.audit_state["zeta"],b.zeta.detach()))

    def test_ffhq_affine_physical_noise_and_empirical_mapping_contract(self):
        import torch
        from utils.diag_ffhq_regression import TransposeMode
        class Base(torch.nn.Module):
            scale_factor=4
            def H(self,x):return torch.nn.functional.avg_pool2d(x,4)
        legacy=TransposeMode(Base(),"legacy_bicubic");unit=audit.unit_operator(legacy)
        x=torch.linspace(-1,1,64).reshape(1,1,8,8);noise=torch.full((1,1,2,2),.05)
        ym=legacy.H(x)+noise;yu=unit.H(x)+noise
        torch.testing.assert_close(yu,(legacy.H(x)+legacy.H(torch.ones_like(x)))/2+noise)
        torch.testing.assert_close(unit.transpose(noise),.5*legacy.transpose(noise))
        self.assertFalse(torch.allclose(yu,(ym+1)/2))


if __name__=="__main__":unittest.main()
