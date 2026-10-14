"""Torch-free controller/protocol tests; optional actual-core CPU null parity."""
import copy
import importlib.util
import math
import unittest
from unittest.mock import patch
from utils import diag_imagenet_tuned_state as audit


def alpha_table():
    return [.999-(.999-.00004)*t/999 for t in range(1000)]


class SchedulerTests(unittest.TestCase):
    grid = [round(999*(29-k)/29) for k in range(30)]

    def run_epoch(self, scheduler, residual_scale=1.):
        scheduler.reset()
        t, visited, rows = 999, [], []
        while not scheduler.done():
            visited.append(t)
            scheduler.set_time(t)
            u = scheduler.u_by_t[t]
            scheduler.update_state(residual_scale*math.exp(2-u), .7 if scheduler.used >= 2 else float("nan"))
            h = scheduler.select_step(t)
            rows.append(scheduler.snapshot())
            if not scheduler.done():
                t -= h
        return visited, rows

    def test_null_always_exact_grid_and_pilot_stays_fixed(self):
        s = audit.NoiseProfileScheduler(self.grid, alpha_table(), 0.)
        for scale in (1., .7, 2.):
            visited, _ = self.run_epoch(s, scale)
            self.assertEqual(visited, self.grid)
        self.assertAlmostEqual(s.pilot_log_r[0], 2.)

    def test_same_noise_profile_not_step_number_has_zero_response(self):
        s = audit.NoiseProfileScheduler(self.grid, alpha_table(), .2)
        self.run_epoch(s)
        s.reset()
        # Deliberately query u=.4 under reverse-call slot 1, not slot 12.
        t = min(range(1000), key=lambda t: abs(s.u_by_t[t]-.4))
        s.set_time(999); s.update_state(math.exp(2.), float("nan")); s.select_step(999)
        s.set_time(t); s.update_state(math.exp(2-s.u_by_t[t]), .7)
        self.assertAlmostEqual(s.profile_log_ratio, 0., places=10)
        self.assertAlmostEqual(s.profile_residual_signal, 0., places=10)

    def test_bounded_budget_and_confidence_never_reverses_physical_sign(self):
        s = audit.NoiseProfileScheduler(self.grid, alpha_table(), .2)
        self.run_epoch(s)
        s.reset(); t=999
        for k in range(30):
            s.set_time(t)
            u=s.u_by_t[t]
            s.update_state(math.exp(2-u+(.9 if k%2 else -.9)), (-.9 if k%2 else .9))
            h=s.select_step(t)
            self.assertGreaterEqual(s.step_modifier,.8)
            self.assertLessEqual(s.step_modifier,1.2)
            self.assertGreaterEqual(s.profile_confidence,.75)
            self.assertLessEqual(s.profile_confidence,1.)
            self.assertGreaterEqual((s.state_score-.5)*s.profile_residual_signal, -1e-15)
            if k < 29:
                self.assertGreater(h,0); t-=h
        self.assertEqual(t,0)

    def test_pilot_interpolation_and_invalid_inputs(self):
        self.assertAlmostEqual(audit.interpolate([0.,1.], [2.,4.], .25), 2.5)
        self.assertTrue(math.isnan(audit.interpolate([0.,1.], [float("nan"),4.], .25)))
        self.assertEqual(audit.interpolate([0.,1.], [float("nan"),4.], 1.),4.)
        with self.assertRaises(ValueError):
            audit.NoiseProfileScheduler(self.grid, [.5]*1000, .2)
        s = audit.NoiseProfileScheduler(self.grid,alpha_table(),.2)
        with self.assertRaises(RuntimeError):
            s.update_state(1.,.7)


class ProtocolTests(unittest.TestCase):
    def parent(self):
        return dict(status="complete", experiment="first_zeta_initialization_joint_training",
            new_nfe=300, optimizer_steps=10, final_mode="last_opt_pre_final_update",
            first_zeta_init=.04002983177868308,
            config=dict(num_steps=30, num_epochs=10, lr=.001, zeta_init=.2, d_init=.2,
                        eta=1., use_learned_var=False, sampler_mode="ddpm", wave="db4", level=3),
            gates={k:dict(passed=True) for k in ("zero_nfe_noise_preflight","training_pairing","cached_metrics")},
            decision=dict(classification="SUPPORTS_FIRST_INITIALIZATION_SENSITIVITY"),
            results=[dict(policy="first_init_joint_training")])

    def test_failed_or_changed_parent_rejected(self):
        original=self.parent(); audit.validate_parent(original)
        for key,value in (("status","failed"),("new_nfe",330),("first_zeta_init",.04)):
            other=copy.deepcopy(original); other[key]=value
            with self.assertRaises(ValueError):
                audit.validate_parent(other)
        other=copy.deepcopy(original); other["config"]["lr"]=.05
        with self.assertRaises(ValueError):
            audit.validate_parent(other)

    def test_parity_cannot_hide_failure_by_large_repeat_floor(self):
        small=dict(output=0.,zeta_grad=5e-6,D_grad=6e-6)
        self.assertTrue(audit.parity_decision(small,small,True)["passed"])
        large=dict(output=0.,zeta_grad=.01,D_grad=.01)
        self.assertFalse(audit.parity_decision(large,large,True)["passed"])
        self.assertFalse(audit.parity_decision(small,small,False)["passed"])

    def test_metric_tradeoffs_and_parent_drift_not_hidden(self):
        null=dict(psnr=21.36,ssim=.44,lpips=.51)
        state=dict(psnr=21.5,ssim=.44,lpips=.51)
        self.assertEqual(audit.interpret(null,state,null)["classification"],"SINGLE_IMAGE_STATE_GAIN_CANDIDATE")
        state["lpips"]+=.01
        self.assertEqual(audit.interpret(null,state,null)["classification"],"PSNR_ONLY_TRADEOFF")
        parent=dict(null,psnr=21.)
        self.assertEqual(audit.interpret(null,state,parent)["classification"],"INCONCLUSIVE_NULL_DRIFT")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; actual-core CPU test not run")
class TensorTests(unittest.TestCase):
    def test_actual_core_fixed_vs_null_output_gradients_rng_and_indices(self):
        import torch
        from types import SimpleNamespace
        from modules.zaps_algorithm import ZAPS
        from utils.diag_zaps_sampler_parity import rng_state,same_rng
        class Operator(torch.nn.Module):
            def H(self,x): return x
            def transpose(self,x): return x
        class DWT(torch.nn.Module):
            def __init__(self,*a,**kw): super().__init__()
            def analysis(self,x): return x
            def synthesis(self,x): return x
        class Score(torch.nn.Module):
            def forward(self,x,t): return .1*x
        model=Score()
        dm=SimpleNamespace(device="cpu",num_steps=1000,alphas_cumprod=torch.tensor(alpha_table()),
            model=model,_predict_eps=model)
        with patch("modules.zaps_algorithm.OrthogonalDWT2D",DWT):
            z=ZAPS(dm,Operator(),img_size=2,channels=1,num_steps=30,num_epochs=10,lr=.001,
                   use_learned_var=False)
        with torch.no_grad(): z.zeta[-1]=.04002983177868308
        initial=torch.full((1,1,2,2),.3); y=torch.full_like(initial,.1)
        results=[]
        for adaptive in (False,True):
            torch.manual_seed(1001)
            indices=[]; original=z._tweedie_estimate
            def tweedie(x,eps,ab,index):
                indices.append(index); return original(x,eps,ab,index)
            s=audit.NoiseProfileScheduler(z.tau.flip(0).tolist(),alpha_table(),0.)
            hook=model.register_forward_pre_hook(lambda m,args:s.set_time(int(args[1][0])))
            try:
                with patch.object(z,"_tweedie_estimate",side_effect=tweedie):
                    counter=[0]
                    if adaptive: out=z._reverse_diffusion_adaptive(y,s,nfe_counter=counter,init_noise=initial)
                    else: out=z._reverse_diffusion(y,nfe_counter=counter,init_noise=initial)
                    grads=torch.autograd.grad((out-y).square().mean(),(z.zeta,z.D))
                    results.append((out.detach(),grads,rng_state("cpu"),indices,counter[0]))
            finally: hook.remove()
        a,b=results
        self.assertTrue(torch.equal(a[0],b[0]))
        for x,y in zip(a[1],b[1]): self.assertTrue(torch.equal(x,y))
        self.assertTrue(same_rng(a[2],b[2])); self.assertEqual(a[3],b[3]); self.assertEqual(a[4],30)


if __name__ == "__main__":
    unittest.main()
