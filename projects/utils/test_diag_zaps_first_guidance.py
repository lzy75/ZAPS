"""First-step calibration contract; tensor scope tests require local torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import diag_zaps_first_guidance as audit


class ProtocolTests(unittest.TestCase):
    def guidance(self):
        return {"results": {audit.ARM: {"rows": [
            {"t": 999, "exact_norm": 2., "raw_norm": 10., "raw_norm_ratio": 5.}
        ]}}, **{k: "a"*64 for k in audit.IDENTITIES}}

    def parent(self, guidance):
        return {"status": "complete", "experiment": "archived_terminal_zeta_direction_audit",
                "schedule": audit.ARM, "retraining": False, "optimizer_steps": 0, "new_nfe": 600,
                "source_guidance": "/tmp/guidance", "identity": {k: guidance[k] for k in audit.IDENTITIES},
                "gates": {"autograd_vs_no_grad": {"passed": True}, "archived_loss": {"passed": True},
                          "archive_fingerprints": {"autograd": {"passed": True}, "no_grad": {"passed": True}}}}

    def test_factor_is_norm_derived_not_manual(self):
        self.assertEqual(audit.norm_calibration(self.guidance())["factor"], .2)

    def test_invalid_norms_duplicates_and_ratios_rejected(self):
        for key, value in (("exact_norm", 0), ("raw_norm", float("nan")), ("raw_norm_ratio", 9)):
            g = self.guidance()
            g["results"][audit.ARM]["rows"][0][key] = value
            with self.assertRaises(ValueError):
                audit.norm_calibration(g)
        g = self.guidance()
        g["results"][audit.ARM]["rows"] *= 2
        with self.assertRaises(ValueError):
            audit.norm_calibration(g)

    def test_parent_identity_and_gate_required(self):
        g = self.guidance()
        audit.validate_parent(self.parent(g), g, Path("/tmp/guidance"))
        for field, value in (("status", "failed"), ("new_nfe", 300), ("source_guidance", "/tmp/other")):
            p = self.parent(g)
            p[field] = value
            with self.assertRaises(ValueError):
                audit.validate_parent(p, g, "/tmp/guidance")
        p = self.parent(g)
        p["gates"]["archive_fingerprints"]["no_grad"]["passed"] = False
        with self.assertRaises(ValueError):
            audit.validate_parent(p, g, "/tmp/guidance")

    def test_mixed_metrics_are_not_success(self):
        b = dict(psnr=20., ssim=.5, lpips=.5)
        early = ({"input_x_rms": 10, "raw_x0_clip_fraction": .9},
                 {"input_x_rms": 4, "raw_x0_clip_fraction": .8})
        for lpips, expected in ((.48, "SUPPORTS_FIRST_STEP_AMPLITUDE_SENSITIVITY"),
                                (.52, "MIXED_OR_INCONCLUSIVE")):
            self.assertEqual(audit.interpret(b, dict(psnr=21, ssim=.6, lpips=lpips), early)["classification"], expected)

    def test_auto_parent_skips_failed_or_wrong_identity(self):
        g = self.guidance()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, status in (("001", "complete"), ("003", "failed"), ("002", "complete")):
                parent = self.parent(g)
                parent["status"] = status
                if name == "002":
                    parent["identity"]["image_sha256"] = "b" * 64
                path = root / ("parameter_gradient_" + name)
                path.mkdir()
                (path / "run.json").write_text(json.dumps(parent), encoding="utf-8")
            chosen, _ = audit.select_parent(root, g, "/tmp/guidance")
            self.assertEqual(chosen.parent.name, "parameter_gradient_001")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; tensor scope tests not run")
class ScopeTests(unittest.TestCase):
    def case(self):
        import torch
        class Operator:
            def transpose(self, x):
                return x * 2
        return SimpleNamespace(A=Operator(), tau=torch.tensor([0, 333, 999]),
                               dm=SimpleNamespace(alphas_cumprod=torch.ones(1000)))

    def test_only_first_scaled_identity_exact_and_rng_untouched(self):
        import torch
        for factor in (1., .2):
            zaps = self.case()
            original = zaps.A.transpose
            x = torch.tensor([3., 4.])
            rng = torch.get_rng_state().clone()
            observation = {}
            with patch("utils.diag_zaps_guidance_clipping.approximate_vjp", side_effect=lambda z,v,a,i:v):
                with audit.first_guidance_scope(zaps, factor, 10., observation):
                    values = [zaps.A.transpose(x) for _ in range(3)]
            self.assertTrue(torch.equal(values[0], original(x) * factor))
            self.assertTrue(all(torch.equal(v, original(x)) for v in values[1:]))
            self.assertEqual(zaps.A.transpose, original)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(observation["scaled_times"], [] if factor == 1 else [999])

    def test_failure_restores_operator(self):
        import torch
        zaps = self.case()
        original = zaps.A.transpose
        with patch("utils.diag_zaps_guidance_clipping.approximate_vjp", side_effect=lambda z,v,a,i:v):
            with self.assertRaises(RuntimeError):
                with audit.first_guidance_scope(zaps, .2, 99., {}):
                    zaps.A.transpose(torch.ones(2))
        self.assertEqual(zaps.A.transpose, original)

    def test_nested_passive_replay_null_identity_intervention_and_rollback(self):
        import torch
        from utils.test_diag_zaps_parameter_gradient import ReplayTests
        from utils.diag_zaps_parameter_gradient import evaluate_replay
        from utils.diag_zaps_sampler_parity import same_rng
        core, y, state, initial = ReplayTests().make_case()
        core.dm.alphas_cumprod = torch.ones(1000)
        x0 = core._tweedie_estimate(initial, initial * .3, torch.tensor(.8), 3)
        expected = float(core.A.transpose(y - core.A.H(x0)).double().norm())
        kwargs = dict(device="cpu", saved_index=None)
        baseline, base, rng = evaluate_replay(core, y, state, initial, **kwargs)
        for factor in (1., .2):
            observation = {}
            accounting = {"new_nfe": 0}
            with patch("utils.diag_zaps_guidance_clipping.approximate_vjp", side_effect=lambda z,v,a,i:v):
                with audit.first_guidance_scope(core, factor, expected, observation):
                    output, data, end = evaluate_replay(core, y, state, initial, accounting=accounting, **kwargs)
            self.assertEqual(torch.equal(output, baseline), factor == 1)
            self.assertTrue(same_rng(rng, end))
            self.assertEqual(data["rows"][0], base["rows"][0])
            self.assertEqual(accounting["new_nfe"], 4)
            self.assertTrue(torch.equal(core.zeta, state["zeta"]))
            self.assertTrue(torch.equal(core.D, state["D"]))
            self.assertEqual(len(core.dm.model._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
