"""Contract checks; PyTorch tests skip explicitly on hosts without PyTorch."""

import copy
import importlib.util
from pathlib import Path
import types
import unittest

from utils import diag_dps_same_observation as audit


DIFFUSION = {
    "sampler": "ddpm", "steps": 1000, "noise_schedule": "linear",
    "model_mean_type": "epsilon", "model_var_type": "learned_range",
    "dynamic_threshold": False, "clip_denoised": True,
    "rescale_timesteps": False, "timestep_respacing": 1000,
}
CONDITIONING = {"method": "ps", "params": {"scale": 0.3}}


class ContractTests(unittest.TestCase):
    def test_adapter_never_calls_noisy_forward(self):
        class Operator:
            def H(self, data):
                return 2 * data

            def forward(self, data):
                raise AssertionError("Must not draw another measurement noise")

        adapter = audit.PureOperatorAdapter(Operator())
        self.assertEqual(adapter.forward(3), 6)
        self.assertEqual(adapter.forward(3, noisy_measurement=100), 6)
        self.assertEqual(adapter.forward(3, noisy_measurement=-100), 6)
        with self.assertRaises(ValueError):
            adapter.forward(3, mask="wrong task")

    def test_original_dps_policy_is_accepted(self):
        audit.validate_dps_config(DIFFUSION, CONDITIONING)
        config = {**DIFFUSION, "timestep_respacing": "1000"}
        audit.validate_dps_config(config, CONDITIONING)

    def test_no_silent_sampler_or_guidance_changes(self):
        for key, value in (("steps", 30), ("model_var_type", "fixed_small"),
                           ("clip_denoised", False), ("timestep_respacing", "30")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_dps_config({**DIFFUSION, key: value}, CONDITIONING)
        for scale in (0.2, float("nan")):
            condition = copy.deepcopy(CONDITIONING)
            condition["params"]["scale"] = scale
            with self.assertRaises(ValueError):
                audit.validate_dps_config(DIFFUSION, condition)

    def test_file_fingerprint_is_stable(self):
        source = Path(audit.__file__)
        self.assertEqual(audit.file_sha256(source), audit.file_sha256(source))
        self.assertEqual(len(audit.file_sha256(source)), 64)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; numerical test not run")
class GradientTests(unittest.TestCase):
    def tearDown(self):
        import sys
        for name in list(sys.modules):
            if name.startswith("_zaps_sampler_dps_reference"):
                del sys.modules[name]

    def test_actual_official_loop_forwards_noisy_measurement_safely(self):
        import importlib
        import numpy as np
        import torch
        dps, _ = audit.load_dps_reference(audit.REPO_ROOT / "DPS")
        module = importlib.import_module(dps.__package__ + ".condition_methods")

        class Operator:
            def H(self, data):
                return 2 * data

            def forward(self, data):
                raise AssertionError("No new measurement noise")

        condition = module.get_conditioning_method(
            "ps", operator=audit.PureOperatorAdapter(Operator()),
            noiser=types.SimpleNamespace(__name__="gaussian"), scale=0.3,
        )
        sampler = dps.DDPM(use_timesteps=[0, 1], betas=np.array([0.01, 0.02]),
                           model_mean_type="epsilon", model_var_type="learned_range",
                           dynamic_threshold=False, clip_denoised=True,
                           rescale_timesteps=False)
        calls = []

        def toy_model(x, t):
            calls.append(int(t[0]))
            return torch.cat((torch.zeros_like(x), torch.zeros_like(x)), dim=1)

        initial = torch.full((1, 3, 4, 4), 0.1)
        result = sampler.p_sample_loop(
            model=toy_model, x_start=initial, measurement=torch.zeros_like(initial),
            measurement_cond_fn=condition.conditioning, record=False, save_root="unused",
        )
        self.assertEqual(calls, [1, 0])
        self.assertTrue(torch.isfinite(result).all())

    def test_official_PS_uses_pure_H_and_true_input_gradient(self):
        import torch
        source = audit.REPO_ROOT / "DPS/guided_diffusion/condition_methods.py"
        spec = importlib.util.spec_from_file_location("_same_observation_ps_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Operator:
            def H(self, data):
                return 2 * data

            def forward(self, data):
                raise AssertionError("No noisy forward allowed")

        condition = module.get_conditioning_method(
            "ps", operator=audit.PureOperatorAdapter(Operator()),
            noiser=types.SimpleNamespace(__name__="gaussian"), scale=0.3,
        )
        previous = torch.tensor([[[[0.2, 0.7], [0.5, 0.8]]]], requires_grad=True)
        observation = torch.ones_like(previous)
        x0 = previous / 2
        unconditional = torch.zeros_like(previous)
        result, distance = condition.conditioning(
            x_prev=previous, x_t=unconditional.clone(), x_0_hat=x0,
            measurement=observation, noisy_measurement=torch.zeros_like(previous),
        )
        expected_gradient = (previous.detach() - observation) / (previous.detach() - observation).norm()
        self.assertTrue(torch.allclose(result, -0.3 * expected_gradient))
        self.assertTrue(torch.allclose(distance, (previous.detach() - observation).norm()))


if __name__ == "__main__":
    unittest.main()
