"""A30 final-label identity and actual kernel/budget preflight; no checkpoint loaded."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import final_allocation_value as r
try:
    import torch
except ImportError:
    torch = None

GRID = [999, 989, 978, 967, 956, 944, 931, 917, 903, 888, 872, 854, 835, 815, 793,
        769, 742, 712, 679, 641, 597, 546, 484, 409, 319, 218, 124, 54, 14, 0]


def fixtures():
    records, points = [], []
    for image in range(20, 26):
        for seed in (1000, 2000):
            for step in r.STEPS:
                records.append(dict(case=dict(image_id=f'{image:05d}', sampling_seed=seed), step=step,
                    t=GRID[step], m=GRID[step]-1, baseline=dict(psnr=20., ssim=.7, lpips=.2, loss=1.),
                    metrics=dict(psnr=20.1, ssim=.701, lpips=.199, loss=.99)))
                points.append(dict(image_id=f'{image:05d}', seed=seed, step=step, R=2., q_rms=.2,
                    move_rms=.5, clip=.1, history_turn=.3,
                    probes={d: dict(usable=True, L_dir=2., L_mag=4.) for d in ('q', 'travel')},
                    actions=[dict(factor=f, gamma=g, mean=dict(loss=10-g), stochastic=dict(loss=10-g, psnr=20+g))
                             for f in (.5, 1., 1.5) for g in (.5, 1., 2.)],
                    refinements=[dict(factor=f, mean_error=.5*f, stochastic_error=2., stochastic_fine=dict(psnr=22.)) for f in (1., 1.5)]))
    return records, points


class AnalysisTests(unittest.TestCase):
    def test_budget_is_counterfactual_301_and_only_864_new_calls(self):
        self.assertEqual(r.CALLS_PER_CASE, 72)
        self.assertEqual(12*r.CALLS_PER_CASE, 864)
        for k in r.STEPS:
            # Cached old last-unroll prefix+center, new midpoint+suffix.
            self.assertEqual(270+(k+1)+(30-k), 301)

    def test_beating_C_does_not_imply_state_allocation_headroom(self):
        records, points = fixtures(); result = r.analyze(records, points)
        self.assertEqual(result['classification'], 'NO_MATERIAL_FINAL_ALLOCATION_HEADROOM')
        self.assertAlmostEqual(result['same_301_call_oracle_headroom_psnr'], 0.)
        self.assertGreater(result['layers'][0]['mean_delta_vs_C']['psnr'], 0.)

    def test_risk_label_uses_original_interval_not_large_span(self):
        records, points = fixtures(); first = r.analyze(records, points)
        altered = copy.deepcopy(points)
        for p in altered:
            next(f for f in p['refinements'] if f['factor'] == 1.5)['mean_error'] = 12345.
        self.assertEqual(first, r.analyze(records, altered))

    def test_incomplete_or_duplicated_identity_is_rejected(self):
        records, points = fixtures()
        with self.assertRaises(ValueError): r.analyze(records[:-1], points)
        records[-1] = records[0]
        with self.assertRaises(ValueError): r.analyze(records, points)


@unittest.skipIf(torch is None, 'Torch kernel contract runs on server, no UNet loaded')
class TorchTests(unittest.TestCase):
    def test_frozen_field_final_parity_actual_core_suffix_and_query_count(self):
        from utils import diag_final_allocation_value as d
        from modules.diffusion_model import _precompute_diffusion_params
        alpha = _precompute_diffusion_params()['alphas_cumprod']
        class Identity:
            H = staticmethod(lambda x: x)
            transpose = staticmethod(lambda x: x)
            analysis = staticmethod(lambda x: x)
            synthesis = staticmethod(lambda x: x)
        z = SimpleNamespace(A=Identity(), dwt=Identity(), zeta=torch.full((30,), .001),
                            D=torch.ones(30), dm=SimpleNamespace(alphas_cumprod=alpha))
        positions = SimpleNamespace(nominal=GRID)
        gen = torch.Generator().manual_seed(3021)
        x = torch.randn((1, 3, 8, 8), generator=gen)*.1; clean = torch.full_like(x, .2); y = torch.full_like(x, .3)
        inputs, cleans, qs, noises = [], [], [], {}
        for j, time_ in enumerate(GRID):
            inputs.append(x.clone()); cleans.append(clean.clone())
            q = d.P.coupled.frozen_guidance(z, y, clean, time_, 29-j); qs.append(q)
            if j == 29:
                x = d.P.terminal.terminal_output(z, x, dict(clean=clean, q=q))
            else:
                noises[j] = torch.randn(x.shape, generator=gen)
                x = d.P.coupled.posterior(x, clean, time_, GRID[j+1], alpha, noises[j])+q
        original = x.clone(); cache = dict(latent_inputs=torch.cat(inputs), clean=torch.cat(cleans), guidance=torch.cat(qs))
        before = torch.get_rng_state().clone()
        for step in r.STEPS:
            calls = []
            def evaluate(value, time_):
                calls.append(time_)
                q = d.P.coupled.frozen_guidance(z, y, clean, time_, 29-GRID.index(time_)) if time_ in GRID else torch.zeros_like(clean)
                return dict(clean=clean, q=q)
            auxiliary = torch.randn(clean.shape, generator=gen)
            actual, info = d.refine_and_continue(z, y, cache, positions, noises, step, auxiliary, evaluate)
            torch.testing.assert_close(actual, original, atol=0., rtol=0.)
            self.assertEqual(len(calls), 30-step)
            self.assertEqual(calls, [info['m']]+GRID[step+1:])
            self.assertLessEqual(info['frozen_error'], 1e-6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
