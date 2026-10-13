"""Protocol/decision tests; optional tiny CPU replay tests when torch exists."""

import copy
import importlib.util
import math
import unittest

from utils import diag_zaps_parameter_gradient as audit


def source_records():
    from utils.test_diag_zaps_clamp_ablation import records
    guidance, saved = records()
    guidance["task"] = "gaussian_deblur"
    for key in ("checkpoint_sha256", "measurement_sha256", "operator_source_sha256",
                "core_sha256", "image_sha256"):
        guidance[key] = "a" * 64
    saved["arguments"] = {"task": "gaussian_deblur"}
    arm = saved["results"]["irregular_15_10_5"]
    arm["config"].update(wave="db4", level=3, zeta_init=.2, d_init=.2)
    arm.update(gates={"passed": True}, rows=[{"t": 999}], archived_loss_history=[.003] * 10)
    return guidance, saved, {"arguments": {"dataset": "imagenet"}}


def linear_pairs(slopes):
    result = []
    for h, slope in zip(audit.DELTAS, slopes):
        result.append({"requested_delta": h, "plus_value": .2 + h, "minus_value": .2 - h,
                       "plus_loss64": .003 + slope * h, "minus_loss64": .003 - slope * h})
    return result


class ContractTests(unittest.TestCase):
    def test_agreed_protocol_passes(self):
        audit.validate_protocol(*source_records(), audit.PROBE_TIMES, audit.DELTAS)

    def test_failed_audit_or_changed_setting_is_rejected(self):
        guidance, saved, source = source_records()
        invalid = []
        for key, value in (("lr", .05), ("eta", .75), ("wave", "haar"), ("level", 2)):
            changed = copy.deepcopy(saved)
            changed["results"]["irregular_15_10_5"]["config"][key] = value
            invalid.append((guidance, changed, source))
        changed = copy.deepcopy(guidance)
        changed["status"] = "failed"
        invalid.append((changed, saved, source))
        changed = copy.deepcopy(saved)
        changed["results"]["irregular_15_10_5"]["gates"]["passed"] = False
        invalid.append((guidance, changed, source))
        changed = copy.deepcopy(source)
        changed["arguments"]["dataset"] = "ffhq"
        invalid.append((guidance, saved, changed))
        for args in invalid:
            with self.assertRaises(ValueError):
                audit.validate_protocol(*args, audit.PROBE_TIMES, audit.DELTAS)
        with self.assertRaises(ValueError):
            audit.validate_protocol(guidance, saved, source, [999, 333, 0], audit.DELTAS)

    def test_indices_are_original_time_not_reverse_loop_position(self):
        self.assertEqual(audit.parameter_indices([0, 143, 333, 999], [999, 333, 143]), [3, 2, 1])
        for grid, times in (([0, 999, 333], [333]), ([0, 333, 999], [143]),
                            ([0, 333, 999], [333, 333])):
            with self.assertRaises(ValueError):
                audit.parameter_indices(grid, times)


class DecisionTests(unittest.TestCase):
    def classify(self, gradient, slopes, floor=1e-10):
        return audit.classify_probe(gradient, linear_pairs(slopes), floor)

    def test_stable_agreement_conflict_and_scale_mismatch(self):
        self.assertEqual(self.classify(.001, [.001] * 3)["classification"], "LOCAL_AGREEMENT")
        self.assertEqual(self.classify(-.001, [.001] * 3)["classification"], "DIRECTION_CONFLICT")
        self.assertEqual(self.classify(.0005, [.001] * 3)["classification"], "SCALE_MISMATCH")
        self.assertEqual(self.classify(0, [.001] * 3)["classification"],
                         "AUTOGRAD_NEAR_ZERO_WITH_FD_SIGNAL")

    def test_near_zero_or_one_resolved_scale_is_not_a_pass(self):
        for slope in (0, 1e-8, 8e-7):
            self.assertEqual(self.classify(slope, [slope] * 3)["classification"],
                             "INCONCLUSIVE_SMALL_SIGNAL")

    def test_cannot_cherry_pick_two_agreeing_scales(self):
        for slopes in ([.001, -.001, .001], [.001, .0011, .01]):
            self.assertEqual(self.classify(.001, slopes)["classification"],
                             "INCONCLUSIVE_SCALE_DEPENDENT")

    def test_noise_floor_prevents_false_sign_conflict(self):
        self.assertEqual(self.classify(-.001, [.001] * 3, floor=1e-5)["classification"],
                         "INCONCLUSIVE_SMALL_SIGNAL")

    def test_invalid_differences_fail_instead_of_becoming_agreement(self):
        for gradient in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.classify(gradient, [.001] * 3)
        pairs = linear_pairs([.001] * 3)
        pairs[0]["plus_value"] = pairs[0]["minus_value"]
        with self.assertRaises(ValueError):
            audit.classify_probe(.001, pairs, 1e-10)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; tensor replay tests not run")
class ReplayTests(unittest.TestCase):
    def make_case(self):
        import torch
        from types import SimpleNamespace

        class Score(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
                self.fail = False

            def forward(self, x, t):
                if self.fail and int(t[0]) == 333:
                    raise RuntimeError("synthetic model failure")
                return self.weight * x

        class Operator(torch.nn.Module):
            def H(self, x):
                return 2 * x

            def transpose(self, residual):
                return 2 * residual

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A = Operator()
                self.dm = SimpleNamespace(model=Score())
                self.device = "cpu"
                self.tau = torch.tensor([0, 143, 333, 999])
                self.zeta = torch.nn.Parameter(torch.full((4,), .2))
                self.D = torch.nn.Parameter(torch.full((4,), .2))

            def _tweedie_estimate(self, x, eps, ab, index):
                return ((x - .2 * eps.detach()) / ab.sqrt()).clamp(-1, 1)

            def _reverse_diffusion(self, y, nfe_counter, init_noise, record_indicators):
                x = init_noise
                for i in range(3, -1, -1):
                    with torch.no_grad():
                        eps = self.dm.model(x, self.tau[i].reshape(1))
                    nfe_counter[0] += 1
                    x0 = self._tweedie_estimate(x, eps, torch.tensor(.8), i)
                    v = self.A.transpose(y - self.A.H(x0))
                    x = .8 * x0 + self.zeta[i] * (v + .1 * self.D[i] * v)
                    x = x + .01 * torch.randn_like(x)
                return x

        core = Core()
        torch.manual_seed(37)
        state = {"tau": core.tau.clone(), "zeta": core.zeta.detach().clone(),
                 "D": core.D.detach().clone(), "cpu_rng": torch.get_rng_state(), "cuda_rng": None}
        return core, torch.zeros(1, 1, 1, 2), state, torch.full((1, 1, 1, 2), .25)

    def test_value_RNG_and_parameters_rollback_with_and_without_grad(self):
        import torch
        from utils.diag_zaps_sampler_parity import same_rng
        core, y, state, initial = self.make_case()
        kwargs = dict(device="cpu", saved_index=None)
        out, first, rng_a = audit.evaluate_replay(core, y, state, initial, gradient=True, **kwargs)
        repeat, second, rng_b = audit.evaluate_replay(core, y, state, initial, **kwargs)
        self.assertTrue(torch.equal(out, repeat))
        self.assertTrue(same_rng(rng_a, rng_b))
        self.assertEqual(first["nfe"], 4)
        self.assertEqual(first["visited"], [999, 333, 143, 0])
        self.assertIsNotNone(first["gradients"])
        self.assertTrue(torch.equal(core.zeta, state["zeta"]))
        self.assertTrue(torch.equal(core.D, state["D"]))
        self.assertIsNone(core.dm.model.weight.grad)

    def test_perturbation_changes_values_but_not_sampling_draws(self):
        import torch
        from utils.diag_zaps_sampler_parity import same_rng
        core, y, state, initial = self.make_case()
        kwargs = dict(device="cpu", saved_index=None)
        plus, upper, rng_a = audit.evaluate_replay(core, y, state, initial, probe=(1, .001), **kwargs)
        minus, lower, rng_b = audit.evaluate_replay(core, y, state, initial, probe=(1, -.001), **kwargs)
        self.assertFalse(torch.equal(plus, minus))
        self.assertTrue(same_rng(rng_a, rng_b))
        self.assertGreater(upper["effective_value"], lower["effective_value"])
        self.assertTrue(torch.equal(core.zeta, state["zeta"]))

    def test_exception_counts_attempted_calls_and_removes_hook_and_perturbation(self):
        import torch
        core, y, state, initial = self.make_case()
        core.dm.model.fail = True
        accounting = {"new_nfe": 0}
        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            audit.evaluate_replay(core, y, state, initial, device="cpu", saved_index=None,
                                  probe=(1, .001), accounting=accounting)
        self.assertEqual(accounting["new_nfe"], 2)
        self.assertEqual(len(core.dm.model._forward_pre_hooks), 0)
        self.assertTrue(torch.equal(core.zeta, state["zeta"]))
        self.assertTrue(torch.equal(core.D, state["D"]))


if __name__ == "__main__":
    unittest.main()
