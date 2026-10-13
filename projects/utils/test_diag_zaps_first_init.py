"""Contracts for the final bounded initialization check; CUDA run stays server-side."""
import math
import importlib.util
import unittest
from unittest.mock import patch
from utils import diag_zaps_first_init as audit


class ContractTests(unittest.TestCase):
    def test_one_derived_initial_value(self):
        self.assertAlmostEqual(audit.initial_value(.2, .20014915889341542), .04002983177868308)
        for original, factor in ((.2, 1), (.2, 0), (.2, math.nan), (.2, 1e-8)):
            with self.assertRaises(ValueError):
                audit.initial_value(original, factor)

    def test_direction_parent_requires_completed_matching_source_and_gates(self):
        from utils.test_diag_zaps_first_direction import ProtocolTests
        first, guidance = ProtocolTests().records()
        first["config"] = {"lr": .001}
        record = {"status": "complete", "new_nfe": 31, "experiment": "first_direction_fixed_norm_frozen_path",
                  "optimizer_steps": 0, "retraining": False, "source_first_run": "/tmp/first/run.json",
                  "config": first["config"], "identity": first["identity"],
                  "gates": {k: {"passed": True} for k in ("cached_terminal_loss", "first_input_fingerprint",
                      "calibrated_first_next_rms", "pairing", "equal_correction_norm", "replay_first_input", "cached_metrics")}}
        audit.validate_direction_parent(record, "/tmp/first/run.json", first)
        record["gates"]["pairing"]["passed"] = False
        with self.assertRaises(ValueError):
            audit.validate_direction_parent(record, "/tmp/first/run.json", first)

    def test_all_outcomes_close_the_single_image_round(self):
        raw = dict(psnr=19.6, ssim=.31, lpips=.65)
        frozen = dict(psnr=20.75, ssim=.395, lpips=.582)
        for candidate, expected in ((dict(psnr=20.8, ssim=.4, lpips=.58), "SUPPORTS_FIRST_INITIALIZATION_SENSITIVITY"),
            (dict(psnr=20., ssim=.33, lpips=.62), "PARTIAL_GAIN_NOT_RETAINING_FROZEN_BENEFIT"),
            (dict(psnr=21., ssim=.4, lpips=.7), "NO_COMPREHENSIVE_INITIALIZATION_GAIN")):
            result = audit.interpret(candidate, raw, frozen)
            self.assertEqual(result["classification"], expected)
            self.assertIn("Stop this single-image", result["next_action"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; tensor training tests not run")
class TensorTests(unittest.TestCase):
    def test_noise_preflight_matches_29_draws_per_epoch_and_rejects_changed_seed(self):
        import torch
        def fake_rng(device):
            current = torch.get_rng_state().clone()
            return {"cpu": current, "cuda": [current.clone()]}
        torch.manual_seed(1001)
        fixed = torch.randn(1, 1, 2, 2)
        for _ in range(9*29):
            torch.randn_like(fixed)
        state = dict(init_noise=fixed.clone(), cpu_rng=torch.get_rng_state().clone(),
                     cuda_rng=[torch.get_rng_state().clone()])
        for _ in range(29):
            torch.randn_like(fixed)
        expected = fake_rng("cpu")
        with patch("utils.diag_learning_rate_ablation.set_seed", side_effect=torch.manual_seed), \
             patch("utils.diag_zaps_sampler_parity.rng_state", side_effect=fake_rng), \
             patch.object(torch.cuda, "current_device", return_value=0):
            gate, tail = audit.rng_preflight(1001, state, "cpu", 0)
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["score_nfe"], 0)
            self.assertEqual(gate["diagnostic_draws"], 291)
            self.assertTrue(torch.equal(tail["cpu"], expected["cpu"]))
            bad, _ = audit.rng_preflight(1002, state, "cpu", 0)
            self.assertFalse(bad["passed"])

    def test_passive_observer_preserves_300_call_training_values_gradients_and_rng(self):
        import torch
        from types import SimpleNamespace
        from modules.zaps_algorithm import ZAPS, ddpm_posterior_step
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.diag_zaps_sampler_parity import rng_state, same_rng

        class Operator:
            def H(self, x):
                return x

            def transpose(self, x):
                return x

        class Score(torch.nn.Module):
            def forward(self, x, t):
                return .1*x

        class Snapshot(torch.nn.Module):
            # Use the actual optimizer, loss, projection and last_opt implementation.
            optimize = ZAPS.optimize

            def __init__(self):
                super().__init__()
                self.device, self.channels, self.img_size = "cpu", 1, 2
                self.num_epochs, self.lr, self.eta = 10, .001, 1.
                self.tau = torch.linspace(0, 999, 30).round().long()
                self.zeta = torch.nn.Parameter(torch.full((30,), .2))
                with torch.no_grad():
                    self.zeta[-1] = audit.initial_value(.2, .20014915889341542)
                self.D = torch.nn.Parameter(torch.full((30, 1, 2, 2), .2))
                self.A, self.dm = Operator(), SimpleNamespace(model=Score())
                self.ab = torch.linspace(.99, .5, 1000)
                self.audit_calls, self.audit_state = 0, None

            def _tweedie_estimate(self, x, eps, ab, index):
                return ((x-(1-ab).sqrt()*eps.detach())/ab.sqrt()).clamp(-1, 1)

            def _reverse_diffusion(self, y, nfe_counter, eta_override, init_noise):
                self.audit_calls += 1
                if self.audit_calls == 10:
                    self.audit_state = dict(zeta=self.zeta.detach().clone(), D=self.D.detach().clone(),
                        init_noise=init_noise.clone(), cpu_rng=torch.get_rng_state().clone())
                x = init_noise
                for i in range(29, -1, -1):
                    t = int(self.tau[i])
                    with torch.no_grad():
                        eps = self.dm.model(x, self.tau[i].reshape(1))
                    nfe_counter[0] += 1
                    x0 = self._tweedie_estimate(x, eps, self.ab[t], i)
                    uncond = ddpm_posterior_step(x, x0, t, int(self.tau[i-1]) if i else -1, self.ab)
                    v = self.A.transpose(y-self.A.H(x0))
                    x = uncond+self.zeta[i]*(v+(1-self.ab[t])*self.D[i]*v)/self.ab[t].sqrt()
                return x

        record = {"epochs": []}
        observed_type = audit.make_observed_training(Snapshot, record, lambda: None)
        original, observed = Snapshot(), observed_type()
        y = torch.full((1, 1, 2, 2), .2)
        torch.manual_seed(1001)
        loss_a = original.optimize(y, verbose=False)
        tail_a = rng_state("cpu")
        torch.manual_seed(1001)
        with model_rng_guard(observed.dm.model, "cpu") as calls:
            loss_b = observed.optimize(y, verbose=False)
        self.assertEqual(loss_a, loss_b)
        self.assertEqual(len(calls), 300)
        self.assertEqual(observed._last_nfe, 300)
        self.assertEqual(len(record["epochs"]), 10)
        self.assertTrue(same_rng(tail_a, rng_state("cpu")))
        for name in ("zeta", "D", "_last_opt_x0"):
            self.assertTrue(torch.equal(getattr(original, name), getattr(observed, name)))
        self.assertTrue(torch.equal(original.zeta.grad, observed.zeta.grad))
        self.assertTrue(torch.equal(original.D.grad, observed.D.grad))
        self.assertAlmostEqual(record["epochs"][-1]["first_zeta_used"], float(observed.audit_state["zeta"][-1]))
        self.assertAlmostEqual(record["epochs"][0]["first_zeta_used"], .04002983177868308, places=7)


if __name__ == "__main__":
    unittest.main()
