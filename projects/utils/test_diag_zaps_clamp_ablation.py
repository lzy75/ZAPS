"""Clamp-ablation source contracts and tiny differentiable wrapper tests."""

import copy
import importlib.util
import unittest

from utils import diag_zaps_clamp_ablation as audit


def records():
    names = ("irregular_15_10_5", "uniform_30")
    guidance = {"status": "complete", "results": {}}
    saved = {"results": {}}
    for name in names:
        guidance["results"][name] = {
            "replay_gate": {"passed": True}, "archive_fingerprint_gate": {"passed": True},
            "rows": [{"t": 999}],
            "probe_gates": [{key: {"passed": True} for key in
                             ("epsilon_vs_captured", "independent_graph_epsilon", "frozen_epsilon_clamp_chain")}],
        }
        saved["results"][name] = {"config": {
            "num_steps": 30, "num_epochs": 10, "lr": .001, "eta": 1.,
            "use_learned_var": False, "sampler_mode": "ddpm", "surrogate_score_jacobian": False,
        }}
    return guidance, saved


class ContractTests(unittest.TestCase):
    def test_completed_audited_source_is_accepted(self):
        audit.validate_source(*records())

    def test_failed_missing_probe_or_changed_policy_is_rejected(self):
        guidance, saved = records()
        invalid = []
        bad = copy.deepcopy(guidance)
        bad["status"] = "failed"
        invalid.append((bad, saved))
        bad = copy.deepcopy(guidance)
        bad["results"]["uniform_30"]["probe_gates"] = []
        invalid.append((bad, saved))
        bad = copy.deepcopy(guidance)
        bad["results"]["uniform_30"]["archive_fingerprint_gate"]["passed"] = False
        invalid.append((bad, saved))
        for key, value in (("lr", .05), ("eta", .75), ("num_epochs", 20), ("use_learned_var", True)):
            bad_saved = copy.deepcopy(saved)
            bad_saved["results"]["uniform_30"]["config"][key] = value
            invalid.append((guidance, bad_saved))
        for pair in invalid:
            with self.assertRaises(ValueError):
                audit.validate_source(*pair)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; differentiable wrapper tests not run")
class WrapperTests(unittest.TestCase):
    def make_core(self):
        import torch

        class Operator(torch.nn.Module):
            def H(self, x):
                return 2 * x

            def transpose(self, residual):
                return 2 * residual

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A = Operator()
                self.tau = torch.tensor([0, 999])
                self.zeta = torch.nn.Parameter(torch.tensor([.2, .2]))
                self.D = torch.nn.Parameter(torch.tensor([.2, .2]))

            def _tweedie_estimate(self, x, epsilon, ab, index):
                return (x / ab.sqrt()).clamp(-1, 1)

            def _reverse_diffusion(self, y, init_noise):
                x = init_noise
                for index in (1, 0):
                    ab = torch.tensor(.5)
                    x0 = self._tweedie_estimate(x, torch.zeros_like(x), ab, index)
                    v = self.A.transpose(y - self.A.H(x0))
                    # A symmetric non-diagonal toy B checks mask placement.
                    guided = v + .5*self.D[index]*(v + v.flip(-1))
                    x = x0 + self.zeta[index]*guided + .01*torch.randn_like(x)
                return x
        return Core

    def test_null_wrapper_is_exact_and_mask_keeps_RNG_consumption(self):
        import torch
        Core = self.make_core()
        Wrapped = audit.make_clamp_zaps(Core)
        y, initial = torch.zeros(1, 1, 1, 2), torch.tensor([[[[.2, 2.]]]])
        torch.manual_seed(31)
        reference = Core()._reverse_diffusion(y, initial)
        reference_rng = torch.get_rng_state()
        torch.manual_seed(31)
        null = Wrapped(use_clamp_mask=False)
        null_output = null._reverse_diffusion(y, initial)
        self.assertTrue(torch.equal(reference, null_output))
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        torch.manual_seed(31)
        masked = Wrapped(use_clamp_mask=True)
        masked_output = masked._reverse_diffusion(y, initial)
        self.assertFalse(torch.equal(masked_output, reference))
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        self.assertEqual([r["t"] for r in masked.clamp_rows], [999, 0])
        self.assertEqual(masked.clamp_rows[0]["raw_x0_clip_fraction"], .5)

    def test_masked_wrapper_retains_zeta_D_gradients(self):
        import torch
        Wrapped = audit.make_clamp_zaps(self.make_core())
        zaps = Wrapped(use_clamp_mask=True)
        y = torch.zeros(1, 1, 1, 2)
        output = zaps._reverse_diffusion(y, torch.tensor([[[[.2, 2.]]]]))
        (y-zaps.A.H(output)).square().sum().backward()
        for parameter in (zaps.zeta, zaps.D):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()), 0.)

    def test_scope_masks_adjoint_before_non_diagonal_B_and_restores_on_error(self):
        import torch
        zaps = self.make_core()()
        zaps.tau = torch.tensor([999])
        y = torch.zeros(1, 1, 1, 2)
        zaps._clamp_audit_y = y
        original_tweedie, original_transpose = zaps._tweedie_estimate, zaps.A.transpose
        with audit.clamp_guidance_scope(zaps, True, []):
            x0 = zaps._tweedie_estimate(torch.tensor([[[[.2, 2.]]]]), torch.zeros_like(y), torch.tensor(.5), 0)
            v = zaps.A.transpose(y-zaps.A.H(x0))
            self.assertEqual(float(v[..., 1]), 0.)
            guided = v + .5*zaps.D[0]*(v+v.flip(-1))
            self.assertNotEqual(float(guided[..., 1]), 0.)  # masking AFTER B would incorrectly zero this
        self.assertEqual(zaps._tweedie_estimate, original_tweedie)
        self.assertEqual(zaps.A.transpose, original_transpose)
        with self.assertRaises(RuntimeError):
            with audit.clamp_guidance_scope(zaps, True, []):
                zaps.A.transpose(y)  # invalid order exercises cleanup
        self.assertEqual(zaps._tweedie_estimate, original_tweedie)
        self.assertEqual(zaps.A.transpose, original_transpose)

    def test_pairing_compares_selected_sampling_generator_not_all_GPU_counts(self):
        import torch
        old = {"init_noise": torch.ones(2), "cpu_rng": torch.tensor([1]),
               "cuda_rng": [torch.tensor([2]), torch.tensor([3])]}
        new = {"init_noise": torch.ones(2), "cpu_rng": torch.tensor([1]), "cuda_rng": [torch.tensor([3])]}
        self.assertTrue(all(audit.training_pairing(old, new, 1, 0).values()))
        new["cuda_rng"][0] = torch.tensor([9])
        self.assertFalse(audit.training_pairing(old, new, 1, 0)["epoch10_sampling_cuda_rng_equal"])


if __name__ == "__main__":
    unittest.main()
