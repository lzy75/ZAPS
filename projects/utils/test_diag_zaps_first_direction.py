"""Protocol tests and optional small CPU direction/algebra/RNG tests."""
import copy
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import diag_zaps_first_direction as audit


class ProtocolTests(unittest.TestCase):
    def records(self):
        from utils.test_diag_zaps_first_guidance import ProtocolTests as SourceTests
        source = SourceTests()
        guidance = source.guidance()
        first = {"status": "complete", "experiment": "first_guidance_norm_frozen_path",
                 "retraining": False, "optimizer_steps": 0, "new_nfe": 60,
                 "source_trace": "/tmp/trace", "identity": source.parent(guidance)["identity"],
                 "calibration": {"factor": .2},
                 "gates": {k: {"passed": True} for k in ("archive_fingerprints", "archived_loss", "pairing")},
                 "results": [{"policy": p} for p in ("raw_baseline", "first_norm_matched")],
                 "calls": [{"policy": p, "nfe": 30, "rows": [{}]*30}
                           for p in ("raw_baseline", "first_norm_matched")],
                 "intervention": {"scaled_times": [999]}}
        return first, guidance

    def test_completed_same_identity_parent_accepted(self):
        audit.validate_first_run(*self.records(), "/tmp/trace")

    def test_failed_changed_factor_or_extra_step_rejected(self):
        first, guidance = self.records()
        changes = []
        for key, value in (("status", "failed"), ("new_nfe", 300), ("source_trace", "/tmp/other")):
            changed = copy.deepcopy(first)
            changed[key] = value
            changes.append(changed)
        changed = copy.deepcopy(first)
        changed["intervention"]["scaled_times"] = [999, 916]
        changes.append(changed)
        changed = copy.deepcopy(first)
        changed["calibration"]["factor"] = .3
        changes.append(changed)
        changed = copy.deepcopy(first)
        changed["gates"]["pairing"]["passed"] = False
        changes.append(changed)
        changed = copy.deepcopy(first)
        changed["identity"]["image_sha256"] = "b"*64
        changes.append(changed)
        for parent in changes:
            with self.assertRaises(ValueError):
                audit.validate_first_run(parent, guidance, "/tmp/trace")

    def test_positive_psnr_with_bad_lpips_is_not_success(self):
        baseline = dict(psnr=20., ssim=.4, lpips=.5)
        candidate = dict(psnr=21., ssim=.5, lpips=.6)
        self.assertEqual(audit.interpret_direction(baseline, candidate)["classification"], "MIXED_OR_INCONCLUSIVE")
        candidate["lpips"] = .4
        self.assertEqual(audit.interpret_direction(baseline, candidate)["classification"],
                         "SUPPORTS_FIRST_STEP_DIRECTION_SENSITIVITY")
        self.assertEqual(audit.interpret_direction(baseline, baseline)["classification"],
                         "NO_FINAL_QUALITY_SUPPORT_FOR_EXACT_DIRECTION")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; tensor direction tests not run")
class TensorTests(unittest.TestCase):
    def case(self):
        import torch
        from modules.zaps_algorithm import ddpm_posterior_step

        class Score(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.matrix = torch.nn.Parameter(torch.tensor([[.3, .1], [-.2, .4]]))

            def forward(self, x, t):
                return (x.reshape(-1, 2) @ self.matrix.T).reshape_as(x)

        class DWT:
            def analysis(self, x):
                return x

            def synthesis(self, x):
                return x

        class Operator:
            def H(self, x):
                return 2*x

            def transpose(self, x):
                return 2*x

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tau = torch.tensor(list(range(29))+[999])
                self.A, self.dwt = Operator(), DWT()
                self.surrogate_score_jacobian = False
                self.device = "cpu"
                score = Score()
                self.dm = SimpleNamespace(model=score, alphas_cumprod=torch.full((1000,), .7))
                self.dm._predict_eps = score
                self.zeta = torch.nn.Parameter(torch.full((30,), .2))
                self.D = torch.nn.Parameter(torch.full((30, 1, 1, 2), .2))

            def _tweedie_estimate(self, x, eps, ab, index):
                return ((x-(1-ab).sqrt()*eps.detach()) / ab.sqrt()).clamp(-1, 1)

            def _reverse_diffusion(self, y, nfe_counter, init_noise, record_indicators):
                x = init_noise
                for i in range(29, -1, -1):
                    t = int(self.tau[i])
                    with torch.no_grad():
                        eps = self.dm._predict_eps(x, self.tau[i].reshape(1))
                    nfe_counter[0] += 1
                    ab = self.dm.alphas_cumprod[t]
                    x0 = self._tweedie_estimate(x, eps, ab, i)
                    uncond = ddpm_posterior_step(x, x0, t, int(self.tau[i-1]) if i else -1,
                                                self.dm.alphas_cumprod)
                    v = self.A.transpose(y-self.A.H(x0))
                    hv = self.dwt.synthesis(self.D[i] * self.dwt.analysis(v))
                    x = uncond + self.zeta[i]*(v+(1-ab)*hv)/ab.sqrt()
                return x

        core = Core()
        torch.manual_seed(42)
        initial = torch.tensor([[[[.25, -.1]]]])
        y = torch.zeros_like(initial)
        state = {"tau": core.tau.clone(), "zeta": core.zeta.detach().clone(),
                 "D": core.D.detach().clone(), "cpu_rng": torch.get_rng_state(), "cuda_rng": None,
                 "init_noise": initial.clone()}
        with torch.no_grad():
            eps = core.dm._predict_eps(initial, torch.tensor([999]))
            ab = core.dm.alphas_cumprod[999]
            x0 = core._tweedie_estimate(initial, eps, ab, 29)
            v = core.A.transpose(y-core.A.H(x0))
            raw = (v+(1-ab)*core.D[29]*v)/ab.sqrt()
            jacobian = (torch.eye(2)-(1-ab).sqrt()*core.dm.model.matrix)/ab.sqrt()
            exact = (v.reshape(-1, 2) @ jacobian).reshape_as(v)
            calibration = dict(exact_norm=float(exact.double().norm()), approximate_norm=float(raw.double().norm()))
            calibration["factor"] = calibration["exact_norm"]/calibration["approximate_norm"]
        capture = {"t": 999, "index": 29, "x": initial.clone(), "epsilon": eps.clone()}
        return core, y, state, initial, calibration, capture

    def test_exact_probe_sign_norm_single_call_and_no_weight_grad(self):
        import torch
        core, y, state, initial, calibration, capture = self.case()
        accounting = {"new_nfe": 0}
        with patch("utils.diag_zaps_guidance_clipping.restore_sampling_rng",
                   side_effect=lambda s,d,i: torch.set_rng_state(s["cpu_rng"])):
            target, matched, clipped, data = audit.first_vjp(core, y, state, capture, calibration,
                                                            "cpu", {"saved_sampling_index": None}, accounting)
        self.assertEqual(accounting["new_nfe"], 1)
        self.assertAlmostEqual(float(target.double().norm()), float(matched.double().norm()), places=5)
        self.assertGreater(data["cosine_approx_exact"], 0)
        self.assertIsNone(core.dm.model.matrix.grad)
        self.assertEqual(len(core.dm.model._forward_pre_hooks), 0)
        self.assertTrue(torch.equal(core.zeta, state["zeta"]))

    def test_null_bitwise_exact_replacement_equal_norm_and_rng(self):
        import torch
        from utils.diag_zaps_parameter_gradient import evaluate_replay
        from utils.diag_zaps_first_guidance import first_guidance_scope
        from utils.diag_zaps_sampler_parity import same_rng, rng_state
        from modules.zaps_algorithm import ddpm_posterior_step
        core, y, state, initial, calibration, capture = self.case()
        kwargs = dict(device="cpu", saved_index=None)
        base, base_data, base_rng = evaluate_replay(core, y, state, initial, **kwargs)
        original_transpose, original_synthesis = core.A.transpose, core.dwt.synthesis
        observation = {}
        with audit.first_direction_scope(core, None, calibration["approximate_norm"], observation, enabled=False):
            null, null_data, null_rng = evaluate_replay(core, y, state, initial, **kwargs)
        self.assertTrue(torch.equal(base, null))
        self.assertTrue(same_rng(base_rng, null_rng))
        with first_guidance_scope(core, calibration["factor"], calibration["approximate_norm"], {}):
            matched_out, matched_data, matched_rng = evaluate_replay(core, y, state, initial, **kwargs)
        with patch("utils.diag_zaps_guidance_clipping.restore_sampling_rng",
                   side_effect=lambda s,d,i: torch.set_rng_state(s["cpu_rng"])):
            target, matched, clipped, probe = audit.first_vjp(core, y, state, capture, calibration,
                "cpu", {"saved_sampling_index": None}, {"new_nfe": 0})
        torch.set_rng_state(state["cpu_rng"])
        with torch.no_grad():
            uncond = ddpm_posterior_step(initial, clipped, 999, 28, core.dm.alphas_cumprod)
            reference_rms = float((uncond+core.zeta[29]*matched).square().mean().sqrt())
            for _ in range(28):
                torch.randn_like(initial)
        expected_rng = rng_state("cpu")
        self.assertAlmostEqual(reference_rms, matched_data["rows"][1]["input_x_rms"], places=5)
        intervention = {}
        with audit.model_rng_guard(core.dm.model, "cpu") as guarded:
            with audit.first_direction_scope(core, target, calibration["approximate_norm"], intervention):
                output, data, end_rng = evaluate_replay(core, y, state, initial, **kwargs)
        self.assertTrue(same_rng(matched_rng, end_rng))
        self.assertTrue(same_rng(expected_rng, end_rng))
        self.assertEqual(guarded, [True]*30)
        self.assertEqual(data["rows"][0], matched_data["rows"][0])
        self.assertEqual(intervention["changed_times"], [999])
        self.assertTrue(intervention["algebra_gate"]["passed"])
        self.assertAlmostEqual(intervention["applied_correction_norm"], probe["reference_correction_norm"], places=5)
        self.assertEqual(core.A.transpose, original_transpose)
        self.assertEqual(core.dwt.synthesis, original_synthesis)
        self.assertTrue(torch.equal(core.zeta, state["zeta"]))
        self.assertTrue(torch.equal(core.D, state["D"]))
        self.assertEqual(len(core.dm.model._forward_pre_hooks), 0)
        self.assertEqual(len(core.dm.model._forward_hooks), 0)

    def test_invalid_norm_or_random_score_rejected_and_hooks_restored(self):
        import torch
        core, y, state, initial, calibration, capture = self.case()
        original_transpose, original_synthesis = core.A.transpose, core.dwt.synthesis
        with self.assertRaises(RuntimeError):
            with audit.first_direction_scope(core, torch.zeros_like(initial), 99., {}):
                vector = core.A.transpose(initial)
                core.dwt.synthesis(core.D[29]*vector)
        self.assertEqual(core.A.transpose, original_transpose)
        self.assertEqual(core.dwt.synthesis, original_synthesis)
        original_forward = core.dm.model.forward

        def noisy(x,t):
            return original_forward(x,t)+torch.randn_like(x)

        with patch.object(core.dm.model, "forward", side_effect=noisy):
            with self.assertRaisesRegex(RuntimeError, "consumed RNG"):
                with audit.model_rng_guard(core.dm.model, "cpu"):
                    core.dm._predict_eps(initial, torch.tensor([999]))
        self.assertEqual(len(core.dm.model._forward_pre_hooks), 0)
        self.assertEqual(len(core.dm.model._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
