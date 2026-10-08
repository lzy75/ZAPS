"""CPU checks for the sampler audit; numerical checks require PyTorch.

Run: python -m unittest utils.test_diag_zaps_sampler_parity -v
No model weights or CUDA are required. PyTorch numerical tests skip explicitly
when PyTorch is unavailable; they are not claimed as GPU validation.
"""

import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from utils import diag_zaps_sampler_parity as audit


class ReferenceLoaderTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("_zaps_sampler_dps_reference", None)

    def test_missing_reference_is_not_silently_replaced(self):
        with self.assertRaises(FileNotFoundError):
            audit.load_dps_reference(audit.REPO_ROOT / "missing-reference")

    def test_private_namespace_and_unchanged_path(self):
        source = audit.REPO_ROOT / "DPS/guided_diffusion/gaussian_diffusion.py"
        before = list(sys.path)
        module = types.SimpleNamespace(__file__=str(source))
        with patch.object(audit.importlib, "import_module", return_value=module) as importer:
            result, metadata = audit.load_dps_reference(audit.REPO_ROOT / "DPS")
        self.assertIs(result, module)
        importer.assert_called_once_with("_zaps_sampler_dps_reference.gaussian_diffusion")
        self.assertEqual(before, sys.path)
        self.assertEqual(len(metadata["source_sha256"]), 2)

    def test_import_failure_restores_search_path(self):
        before = list(sys.path)
        with patch.object(audit.importlib, "import_module", side_effect=ImportError("fixture")):
            with self.assertRaisesRegex(ImportError, "fixture"):
                audit.load_dps_reference(audit.REPO_ROOT / "DPS")
        self.assertEqual(before, sys.path)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; numerical tests not run")
class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        # Execute the existing helper definitions, without importing wavelet/model
        # dependencies. The function bodies are unchanged AST nodes from core.
        tree = ast.parse((audit.PROJECTS_ROOT / "modules/zaps_algorithm.py").read_text())
        posterior = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "ddpm_posterior_step")
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZAPS")
        learned = next(n for n in klass.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_learned_log_var")
        module = ast.Module(body=[posterior, learned], type_ignores=[])
        namespace = {"torch": torch}
        exec(compile(module, "core-helper-test", "exec"), namespace)
        cls.posterior = staticmethod(namespace["ddpm_posterior_step"])
        cls.learned = staticmethod(namespace["_learned_log_var"])
        cls.dps, _ = audit.load_dps_reference(audit.REPO_ROOT / "DPS")

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name.startswith("_zaps_sampler_dps_reference"):
                del sys.modules[name]

    def test_both_variance_policies_and_terminal_noise_mask(self):
        torch = self.torch
        betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
        ab = torch.cumprod(1 - betas, 0).float()
        tau = torch.tensor([0, 24, 333, 667, 916, 999])
        zaps = types.SimpleNamespace(tau=tau, dm=types.SimpleNamespace(alphas_cumprod=ab))
        # Synthetic inputs isolate numerical mechanics; no pretrained-model claim.
        for policy in ("fixed_small", "learned_range"):
            reference = self.dps.DDPM(use_timesteps=tau.tolist(), betas=betas.numpy(),
                                      model_mean_type="epsilon", model_var_type=policy,
                                      dynamic_threshold=False, clip_denoised=True,
                                      rescale_timesteps=False)
            rows = []
            torch.manual_seed(12)
            for index in range(len(tau) - 1, -1, -1):
                t = int(tau[index])
                s = int(tau[index - 1]) if index else -1
                x = torch.randn(1, 3, 8, 8)
                eps = torch.randn_like(x)
                var = torch.rand_like(x) * 2 - 1
                x0 = ((x - (1-ab[t]).sqrt() * eps) / ab[t].sqrt()).clamp(-1, 1)
                llv = self.learned(zaps, var, t, s) if policy == "learned_range" and s >= 0 else None
                captured = {"eps": eps, "var": var, "t": t}
                with patch("builtins.print"):
                    audit.check_step(zaps, reference, captured, self.posterior,
                                     x, x0, t, s, ab, 1.0, llv, "ddpm", rows, 2e-5, 2e-5)
            self.assertTrue(all(r["passed"] for r in rows), rows)
            self.assertEqual(rows[-1]["noise_draws"], 0)
            self.assertEqual(rows[-1]["effective_std_rms"], 0.0)

    def test_corrupt_standard_deviation_is_detected(self):
        torch = self.torch
        betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
        ab = torch.cumprod(1 - betas, 0).float()
        tau = torch.tensor([0, 999])
        zaps = types.SimpleNamespace(tau=tau)
        ref = self.dps.DDPM(use_timesteps=tau.tolist(), betas=betas.numpy(),
                            model_mean_type="epsilon", model_var_type="fixed_small",
                            dynamic_threshold=False, clip_denoised=True, rescale_timesteps=False)
        x, eps = torch.ones(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)
        x0 = (x / ab[999].sqrt()).clamp(-1, 1)

        def wrong_posterior(*args, **kwargs):
            kwargs["eta"] *= 0.75
            return self.posterior(*args, **kwargs)

        rows = []
        with patch("builtins.print"):
            audit.check_step(zaps, ref, {"eps": eps, "var": None, "t": 999},
                             wrong_posterior, x, x0, 999, 0, ab, 1.0, None, "ddpm",
                             rows, 2e-5, 2e-5)
        self.assertFalse(rows[0]["checks"]["std"]["passed"])
        self.assertFalse(rows[0]["passed"])

    def test_wrong_time_mapping_is_detected(self):
        torch = self.torch
        zaps = types.SimpleNamespace(tau=torch.tensor([0, 999]))
        x = torch.zeros(1, 3, 4, 4)
        ab = torch.tensor([0.9999] * 999 + [0.00004])

        class WrongReference:
            def p_mean_variance(self, model, x, index):
                model(x, torch.zeros_like(index))

        with self.assertRaisesRegex(RuntimeError, "wrong original t"):
            audit.check_step(zaps, WrongReference(), {"eps": x, "var": None, "t": 999},
                             self.posterior, x, x, 999, 0, ab, 1.0, None, "ddpm", [], 2e-5, 2e-5)


if __name__ == "__main__":
    unittest.main()
