"""Probe selection contracts and synthetic VJP checks (PyTorch when available)."""

import importlib.util
import math
import types
import unittest
from unittest.mock import Mock, patch

from utils import diag_zaps_guidance_clipping as audit


class ContractTests(unittest.TestCase):
    def test_legacy_and_explicit_saved_CUDA_generator_mapping(self):
        state = {"cuda_rng": ["gpu0 state", "gpu1 state"]}
        self.assertEqual(audit.saved_cuda_selection(state, "cuda")["saved_sampling_index"], 0)
        self.assertEqual(audit.saved_cuda_selection(state, "cuda:1")["saved_sampling_index"], 1)
        self.assertEqual(audit.saved_cuda_selection(state, None, 1)["saved_sampling_index"], 1)
        metadata = {**state, "sampling_cuda_index": 1, "sampling_device": "cuda"}
        self.assertEqual(audit.saved_cuda_selection(metadata, "cuda")["saved_sampling_index"], 1)
        self.assertEqual(audit.saved_cuda_selection({"cuda_rng": ["only"]}, None)["saved_sampling_index"], 0)

    def test_ambiguous_missing_CPU_or_invalid_generator_is_not_guessed(self):
        for state, original, override in (({"cuda_rng": [0, 1]}, None, None),
                                         ({"cuda_rng": None}, "cuda", None),
                                         ({"cuda_rng": [0]}, "cpu", None),
                                         ({"cuda_rng": [0]}, "cuda:1", None),
                                         ({"cuda_rng": [0]}, "cuda", 3)):
            with self.subTest(state=state, original=original, override=override), self.assertRaises(RuntimeError):
                audit.saved_cuda_selection(state, original, override)

    def test_restore_does_not_require_equal_visible_GPU_counts_or_touch_other_GPU(self):
        fake_torch = types.SimpleNamespace(set_rng_state=Mock(), cuda=types.SimpleNamespace(set_rng_state=Mock()))
        state = {"cpu_rng": "cpu state", "cuda_rng": ["saved GPU0", "saved GPU1"]}
        with patch.dict("sys.modules", {"torch": fake_torch}):
            audit.restore_sampling_rng(state, "cuda:0", 1)
        fake_torch.set_rng_state.assert_called_once_with("cpu state")
        fake_torch.cuda.set_rng_state.assert_called_once_with("saved GPU1", device="cuda:0")

    def test_archive_scalar_fingerprint_rejects_wrong_noise_path(self):
        archived = [{"t": 999, "input_x_rms": 1., "residual_norm": 20., "raw_x0_clip_fraction": .01}]
        self.assertTrue(audit.archive_fingerprint_check(archived, archived)["passed"])
        altered = [{**archived[0], "input_x_rms": 1.03}]
        self.assertFalse(audit.archive_fingerprint_check(altered, archived)["passed"])
        self.assertFalse(audit.archive_fingerprint_check([], archived)["passed"])
        # A one-pixel clip count difference is not a new continuous trajectory.
        near = [{**archived[0], "raw_x0_clip_fraction": .01 + 1/(3*256*256)}]
        self.assertTrue(audit.archive_fingerprint_check(near, archived)["passed"])

    def test_original_times_map_to_nearest_grid_indices_once(self):
        grid = [0, 34, 138, 344, 655, 827, 930, 999]
        selected = audit.probe_indices(grid, [999, 916, 833, 667, 333, 143, 0, 999])
        self.assertEqual(selected, [7, 6, 5, 4, 3, 2, 0])
        self.assertEqual([grid[i] for i in selected], [999, 930, 827, 655, 344, 138, 0])

    def test_bad_grid_and_out_of_range_probes_rejected(self):
        for grid, requested in (([], [0]), ([0, 0], [0]), ([999, 0], [0]),
                                 ([0, 999], []), ([0, 999], [-1]), ([0, 999], [1000])):
            with self.subTest(grid=grid, requested=requested), self.assertRaises(ValueError):
                audit.probe_indices(grid, requested)

    def test_region_summary_does_not_treat_undefined_direction_as_evidence(self):
        rows = [
            {"t": 999, "clip_fraction": .8, "raw_relative_error": 3., "masked_relative_error": 1.},
            {"t": 667, "clip_fraction": .4, "raw_relative_error": 2., "masked_relative_error": 4.},
            {"t": 0, "clip_fraction": 1., "raw_relative_error": None, "masked_relative_error": None},
        ]
        summary = audit.region_summary(rows, lambda t: t >= 600)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["masked_relative_error_improved_count"], 1)
        self.assertEqual(summary["median_raw_relative_error"], 2.5)
        self.assertEqual(audit.region_summary(rows, lambda t: t == 0), {"count": 0})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; synthetic VJP tests not run")
class NumericalTests(unittest.TestCase):
    def test_mask_is_applied_before_noncommuting_wavelet_Jacobian(self):
        import torch
        w = torch.tensor([[1., 1.], [-1., 1.]], dtype=torch.float64) / math.sqrt(2.)
        zaps = types.SimpleNamespace(
            D=torch.tensor([[.3, -.6]], dtype=torch.float64),
            dwt=types.SimpleNamespace(analysis=lambda x: w.T @ x, synthesis=lambda x: w @ x),
        )
        ab = torch.tensor(.25, dtype=torch.float64)
        vector, mask = torch.tensor([1., 2.], dtype=torch.float64), torch.tensor([1., 0.], dtype=torch.float64)
        b = (torch.eye(2, dtype=torch.float64) + (1-ab) * w @ torch.diag(zaps.D[0]) @ w.T) / ab.sqrt()
        actual = audit.approximate_vjp(zaps, vector, ab, 0, mask)
        self.assertTrue(torch.allclose(actual, b.T @ (mask * vector)))
        self.assertFalse(torch.allclose(actual, mask * (b.T @ vector)))

    def test_frozen_clamp_gradient_and_endpoint_convention(self):
        import torch
        raw = torch.tensor([-2., -1., .3, 1., 2.], dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(raw.clamp(-1, 1).sum(), raw)[0]
        self.assertTrue(torch.equal(gradient, (raw.detach().abs() <= 1).double()))

    def test_exact_known_score_Jacobian_matches_masked_probe(self):
        import torch
        calls = []
        ab = torch.tensor(.5, dtype=torch.float64)
        x = torch.tensor([[[[.1, 2.]]]], dtype=torch.float64)
        k = .2

        class SingleUseCheckpoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data):
                ctx.factor = k
                return k * data

            @staticmethod
            def backward(ctx, grad):
                factor = ctx.factor
                del ctx.factor  # mirrors guided-diffusion checkpoint lifetime
                return factor * grad

        def epsilon(data, t):
            calls.append(int(t[0]))
            return SingleUseCheckpoint.apply(data)

        zaps = types.SimpleNamespace(
            device="cpu", tau=torch.tensor([0]),
            dm=types.SimpleNamespace(alphas_cumprod=ab[None], _predict_eps=epsilon),
            A=types.SimpleNamespace(H=lambda data: data, transpose=lambda data: data),
            D=torch.full((1, 1, 1, 2), -k / math.sqrt(1-float(ab)), dtype=torch.float64),
            dwt=types.SimpleNamespace(analysis=lambda data: data, synthesis=lambda data: data),
            zeta=torch.tensor([.2], dtype=torch.float64),
        )
        observation = torch.full_like(x, .3)
        row, gates = audit.probe(zaps, observation, {"index": 0, "t": 0, "x": x, "epsilon": k*x})
        self.assertEqual(calls, [0, 0])  # independent single-use checkpoint graphs
        self.assertTrue(all(gate["passed"] for gate in gates.values()))
        self.assertAlmostEqual(row["clip_fraction"], .5)
        self.assertAlmostEqual(row["masked_cosine"], 1., places=12)
        self.assertLess(row["masked_relative_error"], 1e-12)
        self.assertGreater(row["raw_relative_error"], row["masked_relative_error"])
        # The raw exact VJP uses the SAME clipped residual, not a different loss.
        self.assertAlmostEqual(row["exact_raw_same_v_relative_error"], row["raw_relative_error"], places=12)

    def test_zero_direction_is_undefined_not_artificially_perfect(self):
        import torch
        metrics = audit.direction_metrics(torch.ones(2), torch.zeros(2))
        self.assertIsNone(metrics["cosine"])
        self.assertIsNone(metrics["relative_error"])
        self.assertIsNone(metrics["norm_ratio"])
        with self.assertRaises(RuntimeError):
            audit.direction_metrics(torch.tensor([float("nan")]), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
