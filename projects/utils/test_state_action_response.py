"""A28 grouped-information tests and actual Torch kernel/query contract; no UNet weights."""
from types import SimpleNamespace
import unittest
import numpy as np
from utils import state_action_response as r

try:
    import torch
except ImportError:
    torch = None


GRID = [999, 989, 978, 967, 956, 944, 931, 917, 903, 888, 872, 854, 835, 815, 793,
        769, 742, 712, 679, 641, 597, 546, 484, 409, 319, 218, 124, 54, 14, 0]


class AnalysisTests(unittest.TestCase):
    def test_candidates_cover_relative_spans_preserve_anchor_and_strict_interior(self):
        from utils.fixed_count_state_positions import noise_u
        alpha = np.cumprod(1-np.linspace(1e-4, .02, 1000)).astype(np.float32)
        u = noise_u(alpha)
        for k in r.STEPS:
            times = r.next_times(u, GRID[k], GRID[k+1])
            self.assertEqual(times[1]['s'], GRID[k+1])
            self.assertTrue(times[0]['s'] > times[1]['s'] > times[2]['s'])
            self.assertTrue(all(0 <= a['s'] < GRID[k]-1 for a in times))
        with self.assertRaises(ValueError):
            r.next_times(u, 1, 0)

    def test_group_exclusion_holds_out_both_seeds_and_fits_stage_on_training_only(self):
        rows = [dict(image_id=str(i), seed=seed, step=stage, features={}, labels={'y': 100*i+stage+seed})
                for i in range(3) for seed in (0, 1) for stage in (0, 2)]
        first = r.cross_validate(rows, (), 'y')
        held = [p for p in first['predictions'] if p['image_id'] == '2']
        self.assertEqual([p['prediction'] for p in held], [50.5, 52.5, 50.5, 52.5])
        for row in rows:
            if row['image_id'] == '2':
                row['labels']['y'] += 10000
        second = r.cross_validate(rows, (), 'y')
        self.assertEqual([p['prediction'] for p in second['predictions'] if p['image_id'] == '2'],
                         [p['prediction'] for p in held])

    def test_stage_only_feature_cannot_masquerade_as_state_increment(self):
        rows = [dict(image_id=str(i), seed=seed, step=s, features={'f': 100*s}, labels={'y': i+s+seed})
                for i in range(6) for seed in (0, 1) for s in (2, 10, 19)]
        a = r.cross_validate(rows, (), 'y'); b = r.cross_validate(rows, ('f',), 'y')
        self.assertEqual(a, b)

    def test_missing_responses_are_retained_and_block_premature_success(self):
        self.assertEqual(r.analyze([])['classification'], 'INSUFFICIENT_RESOLVED_STATE_RESPONSE')
        self.assertAlmostEqual(r.scale_agreement([1., 1.]), 0.)
        self.assertGreater(r.scale_agreement([1., 2.]), .1)
        with self.assertRaises(ValueError):
            r.scale_agreement([float('nan'), 1.])

    def test_targets_compare_same_noise_endpoints_and_direction_units(self):
        point = dict(image_id='x', seed=1, step=2, R=2., q_rms=.2, move_rms=.5, clip=.1, history_turn=.3,
                     probes={d: dict(usable=True, L_dir=2., L_mag=4.) for d in ('q', 'travel')},
                     actions=[dict(factor=f, gamma=g, mean=dict(loss=10-g), stochastic=dict(loss=10-g, psnr=20+g))
                              for f in r.SPANS for g in r.GAMMAS],
                     refinements=[dict(factor=1.5, mean_error=.5, stochastic_error=2., stochastic_fine=dict(psnr=22.))])
        result = r.derive(point)
        self.assertAlmostEqual(result['labels']['large_step_error'], np.log(3.))
        self.assertEqual(result['labels']['guidance_loss_gain'], 1.)
        self.assertEqual(result['labels']['guidance_psnr_gain'], 1.)
        self.assertEqual(result['labels']['large_refinement_psnr_gain'], 1.)
        self.assertAlmostEqual(result['features']['q_dir'], np.log(2.))

    def test_full_analysis_and_paid_response_survive_unresolved_local_probes(self):
        points = []
        for image in range(6):
            for seed in (1000, 2000):
                value = image+1.+.01*(seed == 2000); gain = np.log1p(.5*value)
                for step in r.STEPS:
                    points.append(dict(image_id=str(image), seed=seed, step=step, R=2., q_rms=.2,
                        move_rms=.5, clip=.1, history_turn=.3,
                        probes={d: dict(usable=True, L_dir=value, L_mag=value) for d in ('q', 'travel')},
                        actions=[dict(factor=f, gamma=g, mean=dict(loss=10-g*gain),
                                      stochastic=dict(loss=10-g*gain, psnr=20+.01*g*gain))
                                 for f in r.SPANS for g in r.GAMMAS],
                        refinements=[dict(factor=1.5, mean_error=.5*value, stochastic_error=.5*value,
                                          stochastic_fine=dict(psnr=21.))]))
        result = r.analyze(points)
        self.assertEqual(result['resolved_states'], 72)
        self.assertTrue(result['local_supported'])
        self.assertEqual(result['classification'], 'LOCAL_STATE_RESPONSE_CANDIDATE_ONLY')
        for p in points:
            p['probes']['q']['usable'] = False
        result = r.analyze(points)
        self.assertEqual(result['resolved_states'], 0)
        self.assertEqual(len(result['unresolved_states']), 72)
        self.assertTrue(result['paid_response_supported'])
        self.assertEqual(result['classification'], 'PAID_RESPONSE_PROBE_CANDIDATE_ONLY')


@unittest.skipIf(torch is None, 'Torch contract must run on server; no checkpoint needed')
class TorchTests(unittest.TestCase):
    def test_actual_field_kernel_count_rng_precision_and_gamma_pairing(self):
        from utils import diag_state_action_response as d
        from utils.fixed_count_state_positions import Positions, noise_u
        from modules.diffusion_model import _precompute_diffusion_params
        from utils.diag_clean_response_precision import precision_state
        alpha = _precompute_diffusion_params()['alphas_cumprod']
        positions = Positions(noise_u(alpha.numpy()), GRID)
        class Identity:
            H = staticmethod(lambda x: x)
            transpose = staticmethod(lambda x: x)
            analysis = staticmethod(lambda x: x)
            synthesis = staticmethod(lambda x: x)
        class Model(torch.nn.Module):
            def forward(self, x, t):
                return .1*x
        module = Model(); calls = []
        hook = module.register_forward_pre_hook(lambda _, inputs: calls.append(int(inputs[1][0])))
        z = SimpleNamespace(A=Identity(), dwt=Identity(), zeta=torch.full((30,), .01),
                            D=torch.ones(30), dm=SimpleNamespace(alphas_cumprod=alpha, _predict_eps=module))
        gen = torch.Generator().manual_seed(2821)
        x, noise, aux, gt = [torch.randn((1, 3, 8, 8), generator=gen)*.1 for _ in range(4)]
        y = torch.full_like(x, .2); step = 24; t = GRID[step]
        def evaluate(value, time, kind):
            return d.fields.field(z, y, value, time, positions)
        f = evaluate(x, t, 'cache'); calls.clear()
        before_rng = torch.get_rng_state().clone(); before_policy = precision_state(torch)
        with torch.no_grad():
            point = d.diagnose_point(z, y, x, positions, step, f['clean'], f['q'], noise, aux,
                                     f['q'], gt, evaluate)
        self.assertEqual(len(calls), r.CALLS_PER_POINT)
        self.assertEqual(len(point['actions']), 9)
        self.assertEqual(len(point['refinements']), 3)
        self.assertTrue(all(g['passed'] for g in point['gates'].values()))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(before_policy, precision_state(torch))
        self.assertTrue(all(np.isfinite(a['stochastic']['psnr']) for a in point['actions']))
        hook.remove()

    def test_tiny_direction_is_unresolved_without_dividing_by_zero(self):
        from utils.diag_state_action_response import fixed_probe
        x = torch.zeros((1, 3, 4, 4))
        result = fixed_probe(x, x, x, 0., lambda v: {'q': torch.zeros_like(v)}, 0.)
        self.assertFalse(result['usable'])
        self.assertTrue(all(np.isfinite(p['L_dir']) for p in result['pairs']))

    def test_physical_readout_does_not_depend_on_ground_truth(self):
        from utils.diag_state_action_response import metrics
        z = SimpleNamespace(A=SimpleNamespace(H=lambda x: x))
        prediction = torch.full((1, 3, 4, 4), .2); y = torch.zeros_like(prediction)
        a = metrics(z, y, prediction, y)
        b = metrics(z, y, prediction, torch.ones_like(prediction))
        self.assertEqual(a['loss'], b['loss'])
        self.assertNotEqual(a['psnr'], b['psnr'])


if __name__ == '__main__':
    unittest.main()
