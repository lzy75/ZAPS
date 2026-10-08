"""Low-step diagnostic contracts; real DPS tests require PyTorch/dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

from utils import diag_dps_lowstep_variance as audit
from utils.test_diag_dps_same_observation import DIFFUSION, CONDITIONING


GRIDS = {
    "irregular_15_10_5": [0, 24, 48, 71, 95, 119, 143, 166, 190, 214, 238, 262,
                           285, 309, 333, 334, 371, 408, 445, 482, 518, 555, 592,
                           629, 666, 667, 750, 833, 916, 999],
    "uniform_30": [round(999 * k / 29) for k in range(30)],
}


def archive(expected):
    return {
        **expected, "status": "complete",
        "dps": {"nfe": 1000, "psnr": 23.1528, "ssim": 0.6301, "lpips": 0.3999},
        "diffusion_config": copy.deepcopy(DIFFUSION),
        "conditioning_config": copy.deepcopy(CONDITIONING),
        "gates": dict.fromkeys(("source_xT_equal", "seed_xT_equal", "pure_H_equal",
                                "full_timestep_grid", "model_grad_not_accumulated"), True),
    }


class ContractTests(unittest.TestCase):
    def test_four_arms_preserve_original_grids(self):
        variants = audit.build_variants(GRIDS)
        self.assertEqual(len(variants), 4)
        self.assertEqual(sum(len(v["grid"]) for v in variants), 120)
        for schedule in GRIDS:
            arms = [v for v in variants if v["schedule"] == schedule]
            self.assertEqual({v["variance"] for v in arms}, {"fixed_small", "learned_range"})
            for arm in arms:
                self.assertEqual(arm["grid"], GRIDS[schedule])
                self.assertIsNot(arm["grid"], GRIDS[schedule])

    def test_reject_missing_duplicate_wrong_endpoint_grid(self):
        invalid = [dict(list(GRIDS.items())[:1])]
        for replacement in ([0] * 30, list(range(30)), GRIDS["uniform_30"][:-1]):
            invalid.append({**GRIDS, "uniform_30": replacement})
        for grids in invalid:
            with self.subTest(grids=grids), self.assertRaises(ValueError):
                audit.build_variants(grids)

    def test_baseline_requires_matching_identity_policy_and_gates(self):
        expected = {"seed": 1001, "checkpoint_sha256": "checkpoint", "measurement_sha256": "y"}
        valid = archive(expected)
        self.assertEqual(audit.baseline_mismatches(valid, expected), [])
        changes = [
            ("status", "failed"), ("seed", 1000), ("checkpoint_sha256", "other"),
            ("measurement_sha256", "new y"),
            ("diffusion_config", {**DIFFUSION, "model_var_type": "fixed_small"}),
            ("gates", {**valid["gates"], "seed_xT_equal": False}),
            ("dps", {**valid["dps"], "nfe": 30}),
            ("dps", {**valid["dps"], "psnr": float("nan")}),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                self.assertTrue(audit.baseline_mismatches({**valid, key: value}, expected))

    def test_auto_find_skips_failed_newer_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"source_trace": str(root), "seed": 1001}
            for suffix, status in (("100", "complete"), ("200", "failed")):
                directory = root / ("dps_same_observation_" + suffix)
                directory.mkdir()
                (directory / "run.json").write_text(json.dumps({**archive(expected), "status": status}))
            found, _ = audit.find_baseline(root, None, expected)
            self.assertEqual(found.name, "dps_same_observation_100")
            with self.assertRaises(RuntimeError):
                audit.find_baseline(root, root / "dps_same_observation_200", expected)
            with self.assertRaises(RuntimeError):
                audit.find_baseline(root, None, {**expected, "seed": 2000})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; real sampler test not run")
class SamplerTests(unittest.TestCase):
    def tearDown(self):
        import sys
        for name in list(sys.modules):
            if name.startswith("_zaps_sampler_dps_reference"):
                del sys.modules[name]

    def test_original_DPS_respacing_and_variance_arms_share_draw_consumption(self):
        import importlib
        import torch
        dps, _ = audit.load_dps_reference(audit.REPO_ROOT / "DPS")
        module = importlib.import_module(dps.__package__ + ".condition_methods")

        class Operator:
            def H(self, x):
                return x

        condition = module.get_conditioning_method(
            "ps", operator=audit.PureOperatorAdapter(Operator()),
            noiser=types.SimpleNamespace(__name__="gaussian"), scale=0.3,
        )
        initial = torch.full((1, 3, 4, 4), 0.1)
        end_rng = None
        for variant in audit.build_variants(GRIDS):
            torch.manual_seed(1001)
            calls = []

            def toy_model(x, t):
                calls.append(int(t[0]))
                return torch.cat((torch.zeros_like(x), torch.zeros_like(x)), dim=1)

            sampler = dps.DDPM(
                use_timesteps=variant["grid"], betas=dps.get_named_beta_schedule("linear", 1000),
                model_mean_type="epsilon", model_var_type=variant["variance"],
                dynamic_threshold=False, clip_denoised=True, rescale_timesteps=False,
            )
            result = sampler.p_sample_loop(
                model=toy_model, x_start=initial.clone(), measurement=torch.zeros_like(initial),
                measurement_cond_fn=condition.conditioning, record=False, save_root="unused",
            )
            self.assertEqual(calls, list(reversed(variant["grid"])))
            self.assertTrue(torch.isfinite(result).all())
            current_rng = audit.rng_state("cpu")
            if end_rng is not None:
                self.assertTrue(audit.same_rng(end_rng, current_rng))
            end_rng = current_rng


if __name__ == "__main__":
    unittest.main()
