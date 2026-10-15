"""Source selection/statistics tests, plus real small-core CPU parity if available."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from utils import diag_state_trajectory as audit


def parent_fixture():
    grid = list(range(29)) + [999]
    config = dict(num_steps=30, num_epochs=10, eta=1., sampler_mode='ddpm', use_learned_var=False,
                  surrogate_score_jacobian=False, lr=.001)
    return dict(status='complete', experiment='ffhq_four_task_public_settings_unit_measurements',
                input_manifest=[dict(path='/data/00000.png', seed=1000, sha256='image')],
                arms={'000_super_resolution': dict(training_gate_passed=True, config=config, grid=grid,
                      pre_snapshot_sha256='pre', output_sha256='output')},
                gates={'000_super_resolution': dict(passed=True)}, resolved_protocol=dict(state=False),
                observations={'000_super_resolution': dict(measurement_sha256='y')},
                checkpoint=dict(sha256='checkpoint'), source_sha256={},
                results=[dict(task='super_resolution', image='/data/00000.png', psnr=26., ssim=.8, lpips=.1)])


class SourceAndStatisticsTests(unittest.TestCase):
    def test_completed_protocol_and_reject_failed_state_wrong_image(self):
        parent = parent_fixture()
        self.assertEqual(audit.describe_parent(parent, '/tmp/run')['transpose'], 'exact')
        for key, value in [('status', 'failed'), ('resolved_protocol', dict(state=True)),
                           ('input_manifest', [dict(path='/data/00001.png', seed=1000, sha256='x')])]:
            changed = copy.deepcopy(parent); changed[key] = value
            with self.assertRaises((ValueError, StopIteration)):
                audit.describe_parent(changed, '/tmp/run')

    def test_selection_date_not_quality_and_skip_missing_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stamp, quality in [('20260101', 40.), ('20260102', 24.)]:
                directory = root/'results/diag_ffhq_four_tasks_unit'/stamp
                tensors = directory/'image_000/super_resolution'; tensors.mkdir(parents=True)
                for name in ('measurement.pt', 'pre_last_state.pt', 'last_opt.pt'):
                    (tensors/name).touch()
                parent = parent_fixture(); parent['results'][0]['psnr'] = quality
                (directory/'run.json').write_text(json.dumps(parent))
            path, _, _, _ = audit.select_parent(root=root)
            self.assertEqual(path.parent.name, '20260102')
            (path.parent/'image_000/super_resolution/pre_last_state.pt').unlink()
            path, _, _, rejected = audit.select_parent(root=root)
            self.assertEqual(path.parent.name, '20260101'); self.assertEqual(len(rejected), 1)
            with self.assertRaises(ValueError):
                audit.select_parent(root/'results/diag_ffhq_four_tasks_unit/20260102')

    def test_historical_null_accepted_but_state_grid_rejected(self):
        new = parent_fixture(); grid = new['arms']['000_super_resolution']['grid']
        p = dict(status='complete', experiment='fixed30_state_measurement_domain_paired', dataset='ffhq',
                 image='/data/00000.png', seed=1000, config=new['arms']['000_super_resolution']['config'],
                 nominal_timesteps_ascending=grid, gates=dict(operator=dict(passed=True),
                     unroll_parity=dict(unit=dict(passed=True))),
                 scheduler_config=dict(unit_null=dict(response_strength=0.)),
                 arms=dict(unit_null=dict(noise_pairing_passed=True, changed_points_after_pilot=0,
                     grids=[list(reversed(grid))]*10, pre_snapshot_sha256='pre', output_sha256='out')),
                 measurement_sha256=dict(unit='y'), identity=dict(image_sha256='i', checkpoint_sha256='c', source_sha256={}),
                 results=[dict(domain='unit', policy='null', psnr=26., ssim=.8, lpips=.16)])
        self.assertEqual(audit.describe_parent(p, '/tmp/run')['transpose'], 'legacy_bicubic')
        p['arms']['unit_null']['changed_points_after_pilot'] = 1
        with self.assertRaises(ValueError):
            audit.describe_parent(p, '/tmp/run')

    def test_ties_missing_constants_and_lag_direction(self):
        self.assertEqual(audit.ranks([2., 1., 2., 3.]), [1.5, 0., 1.5, 3.])
        self.assertIsNone(audit.pearson([1.]*5, [1., 2., 3., 4., 5.]))
        rows = [dict(noise_u=k/5, c=float(k*k), q=float(k**3)) for k in range(6)]
        result = audit.association(rows, 'c', 'q', True)
        self.assertGreater(result['pearson'], .95)
        rows[2]['c'] = None
        self.assertEqual(audit.association(rows, 'c', 'q', True)['n'], 4)

    def test_falling_residual_does_not_certify_control(self):
        rows = [dict(step=k, t=999-999*k//29, noise_u=k/29, residual_noise_ratio=30-k,
                     residual_norm=30-k, cosine_clean=None if k<2 else .5,
                     cosine_latent=None if k==0 else -.1, psnr=20+k*.1, ssim=.7, lpips=.2+k*.01)
                for k in range(30)]
        summary = audit.summarize(rows)
        self.assertTrue(summary['lower_late_residual'])
        self.assertEqual(summary['clean_cosine_valid'], 28)
        self.assertEqual(summary['clean_cosine_span'], 0.)
        self.assertIn('NOT_ESTABLISHED', summary['controller_suitability'])
        with self.assertRaises(ValueError): audit.summarize(rows[:-1])


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'pywt')),
                     'PyTorch/PyWavelets unavailable; real-core observer/RNG tests NOT executed')
class TensorTests(unittest.TestCase):
    def test_real_core_observer_output_rng_parameters_and_cleanup(self):
        import torch
        from modules.zaps_algorithm import ZAPS
        class Score(torch.nn.Module):
            def forward(self, x, t): return .1*x + t.reshape(-1, 1, 1, 1)*.00001
        class Operator(torch.nn.Module):
            def H(self, x): return .5*(x+1)
            def transpose(self, r): return .5*r
        model = Score()
        dm = SimpleNamespace(device='cpu', num_steps=1000, model=model,
             alphas_cumprod=torch.cumprod(1-torch.linspace(.0001, .02, 1000), dim=0),
             _predict_eps=lambda x, t: model(x, t))
        z = ZAPS(dm, Operator(), num_steps=4, schedule=(2, 1, 1), img_size=8, channels=1,
                 wave='haar', level=1, use_learned_var=False, sampler_mode='ddpm')
        initial = torch.linspace(-.2, .2, 64).reshape(1, 1, 8, 8); y = torch.ones_like(initial)*.5
        start = torch.get_rng_state().clone(); before_zeta = z.zeta.detach().clone(); before_D = z.D.detach().clone()
        reference, _, _ = z.sample(y, init_noise=initial); end = torch.get_rng_state().clone()
        torch.set_rng_state(start); captures = []
        original = z._tweedie_estimate
        with audit.capture_clean(z, captures):
            observed, nfe, _ = z.sample(y, init_noise=initial)
        self.assertTrue(torch.equal(reference, observed)); self.assertTrue(torch.equal(end, torch.get_rng_state()))
        self.assertTrue(torch.equal(before_zeta, z.zeta)); self.assertTrue(torch.equal(before_D, z.D))
        self.assertEqual(nfe, 4); self.assertEqual([c['index'] for c in captures], [3, 2, 1, 0])
        self.assertEqual(z._tweedie_estimate, original)
        with self.assertRaisesRegex(RuntimeError, 'forced'):
            with audit.capture_clean(z, []): raise RuntimeError('forced')
        self.assertEqual(z._tweedie_estimate, original)

    def test_zero_directions_and_positive_speed_scaling(self):
        import torch
        a, b = torch.tensor([1., 2.]), torch.tensor([-2., 3.])
        self.assertIsNone(audit.cosine(a, torch.zeros(2)))
        self.assertIsNone(audit.cosine(a, None))
        self.assertAlmostEqual(audit.cosine(a, b), audit.cosine(3*a, .2*b), places=7)


if __name__ == '__main__':
    unittest.main()
