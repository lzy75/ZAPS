"""Cache/time identity, tangent-angle geometry, and optional real-core RNG parity."""
import copy
import importlib.util
import math
from types import SimpleNamespace
import unittest

from utils import diag_cached_direction as audit


def trace_fixture():
    grid = list(range(29))+[999]
    rows = [dict(step=k, t=t, t_prev=grid[28-k] if k<29 else -1, parameter_index=29-k,
                 noise_u=k/29, residual_norm=30.-k, raw_x0_clip_fraction=.05,
                 psnr=20.+k*.1, ssim=.8, lpips=.2, residual_noise_ratio=30.-k,
                 cosine_clean=None if k<2 else .2, cosine_latent=None if k==0 else .1,
                 cosine_mean=None if k==0 else .3) for k, t in enumerate(grid[::-1])]
    return dict(experiment='passive_fixed_parent_state_trajectory', status='complete', method_nfe=0,
                diagnostic_nfe=90, resolved_config=dict(kind='historical_development', seed=1000,
                    image='/data/00000.png', transpose='legacy_bicubic', grid=grid),
                gates={k: dict(passed=True) for k in ('observer', 'archive')}, rows=rows)


class ContractTests(unittest.TestCase):
    def test_complete_source_and_reject_failed_wrong_mapping_or_time(self):
        source = trace_fixture(); audit.validate_trace(source)
        variants = []
        for key, value in [('status', 'failed'), ('method_nfe', 300), ('diagnostic_nfe', 30)]:
            changed = copy.deepcopy(source); changed[key] = value; variants.append(changed)
        changed = copy.deepcopy(source); changed['rows'][0]['parameter_index'] = 0; variants.append(changed)
        changed = copy.deepcopy(source); changed['rows'][2]['noise_u'] = 0.; variants.append(changed)
        changed = copy.deepcopy(source); changed['gates']['archive']['passed'] = False; variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError): audit.validate_trace(variant)

    def test_unit_tangent_turn_line_reversal_circle_and_missing(self):
        self.assertEqual(audit.angular_readouts(1., .1)['tangent_turn'], 0.)
        self.assertEqual(audit.angular_readouts(-1., .1)['tangent_turn'], 2.)
        self.assertEqual(audit.angular_readouts(None, .1)['turn_per_u'], None)
        for h in (.1, .01):
            self.assertAlmostEqual(audit.angular_readouts(math.cos(h), h)['turn_per_u'], 2*math.sin(h/2)/h, places=10)
        for c, h in [(0., 0.), (float('nan'), .1), (1.1, .1)]:
            with self.assertRaises(ValueError): audit.angular_readouts(c, h)

    def test_conditioned_analysis_preserves_original_neighbors(self):
        rows = trace_fixture()['rows']
        for row in rows: row['raw_x0_clip_fraction'] = 1.
        for k in (0, 5, 10, 29): rows[k]['raw_x0_clip_fraction'] = .01
        summary = audit.direction_summary(rows)
        part = summary['scopes']['low_clip_exploratory']
        self.assertEqual(part['n'], 4)
        # k0 is undefined, k29 has no next point: only k5->6 and k10->11.
        self.assertEqual(part['associations']['mean_vs_next_change_psnr']['n'], 2)
        self.assertEqual(summary['controller_suitability'], 'NOT_ESTABLISHED')


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'pywt', 'numpy')),
                     'PyTorch/PyWavelets unavailable; real-core decomposition/RNG tests NOT executed')
class TensorTests(unittest.TestCase):
    def test_tiny_turn_not_lost_to_cosine_rounding(self):
        import torch
        a, b = torch.tensor([1., 0.], dtype=torch.float64), torch.tensor([1., 1e-9], dtype=torch.float64)
        c, readout = audit.vector_turn(a, b, .1)
        self.assertEqual(c, 1.)
        self.assertAlmostEqual(readout['tangent_turn'], 1e-9, delta=1e-15)
        self.assertAlmostEqual(readout['turn_per_u'], 1e-8, delta=1e-15)

    def test_cached_decomposition_reconstructs_real_core_without_score_replay(self):
        import torch
        from modules.zaps_algorithm import ZAPS
        from utils.diag_state_trajectory import capture_clean
        class Score(torch.nn.Module):
            def forward(self, x, t): return x*.1+t.reshape(-1, 1, 1, 1)*.00001
        class Operator(torch.nn.Module):
            def H(self, x): return (x+1)*.5
            def transpose(self, r): return .5*r
        score = Score(); calls = []
        def predict(x, t): calls.append(int(t[0])); return score(x, t)
        dm = SimpleNamespace(device='cpu', num_steps=1000, model=score,
             alphas_cumprod=torch.cumprod(1-torch.linspace(.0001, .02, 1000), dim=0), _predict_eps=predict)
        z = ZAPS(dm, Operator(), num_steps=4, schedule=(2, 1, 1), channels=1, img_size=8,
                 wave='haar', level=1, use_learned_var=False, sampler_mode='ddpm')
        initial = torch.linspace(-.2, .2, 64).reshape(1, 1, 8, 8); y = torch.ones_like(initial)*.5
        rng = torch.get_rng_state().clone(); captures = []
        with capture_clean(z, captures): output, _, _ = z.sample(y, init_noise=initial)
        expected_rng = torch.get_rng_state().clone(); calls.clear(); torch.set_rng_state(rng)
        with torch.no_grad():
            for k, cap in enumerate(captures):
                i = cap['index']; t = int(z.tau[i]); s = int(z.tau[i-1]) if i else -1
                step = audit.cached_step(cap['x'], cap['clean'], y, z.A, z.dwt, z.zeta[i], z.D[i], t, s, dm.alphas_cumprod)
                expected = captures[k+1]['x'] if k<3 else output
                torch.testing.assert_close(step['next'], expected, rtol=0., atol=0.)
                self.assertTrue(torch.isfinite(step['mean_increment']).all())
                if s == -1: self.assertEqual(float(step['noise_increment'].norm()), 0.)
        self.assertEqual(calls, []); self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))


if __name__ == '__main__': unittest.main()
