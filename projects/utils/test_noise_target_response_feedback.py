import math
import unittest
from utils.noise_target_response_feedback import select, stage_profile, profile_signature


class NoiseTargetTests(unittest.TestCase):
    def test_first_crossing_is_smaller_root_and_can_exceed_two(self):
        # r=10, a=1, sigma=1: reach noise norm 1 at gamma=9 rather than LS gamma=10.
        value = select(10., 100., 1., 1, target=1., sigma=1.)
        self.assertAlmostEqual(value['gamma'], 9.)
        self.assertAlmostEqual(value['predicted_R_after'], 1.)
        self.assertAlmostEqual(value['raw_gamma'], 10.)
        # Non-collinear r=(2,1), a=(1,0), noise target sqrt(2): first root is gamma=1.
        value = select(2., 5., 1., 1, target=math.sqrt(2.), sigma=1.)
        self.assertAlmostEqual(value['gamma'], 1.)

    def test_noise_satisfied_wrong_direction_and_unreachable(self):
        self.assertEqual(select(1., 1., 1., 1, sigma=1.)['gamma'], 0.)
        self.assertEqual(select(0., 0., 1., 1, sigma=1.)['gamma'], 0.)
        self.assertEqual(select(-1., 4., 1., 1, sigma=1.)['gamma'], 0.)
        value = select(1., 5., 1., 1, sigma=1.)
        self.assertEqual(value['reason'], 'TARGET_UNREACHABLE_USE_RAY_MINIMUM')
        self.assertAlmostEqual(value['gamma'], 1.)
        self.assertAlmostEqual(value['predicted_R_after'], 2.)
        self.assertTrue(select(0., 4., 0., 1, sigma=1.)['fallback'])

    def test_more_conservative_noise_target_requires_less_guidance(self):
        values = [select(10., 100., 1., 1, target=t, sigma=1.)['gamma'] for t in (.9, 1., 1.1)]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        # Scaling the entire observation model, including sigma, leaves the decision unchanged.
        self.assertAlmostEqual(select(40., 400., 4., 1, sigma=2.)['gamma'], values[1])

    def test_reference_exclusion_and_independent_signature(self):
        profiles = {k: dict(image_id=i, traces={'N100': [dict(gamma=g)]*30})
                    for k, i, g in [('a1','a',999.),('a2','a',999.),('b','b',.2),('c','c',.4)]}
        self.assertAlmostEqual(stage_profile(profiles, 'a', 'N100')[0], .3)
        data = dict(plan_sha256='a', aggregation='median', values={'a': {'TIME100': [.3]*30}},
                    sources={'a/N100': {'sha256': 'b'}})
        before = profile_signature(data, 'N100')
        data['values']['a']['TIME110'] = [.2]*30
        self.assertEqual(before, profile_signature(data, 'N100'))

    def test_torch_linear_sensor_hits_actual_measurement_target(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable on Mac; mandatory on server')
        from types import SimpleNamespace
        from utils.noise_target_response_feedback import sensor
        z = SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.ones(1)),
                            A=SimpleNamespace(H=lambda x: 2*x))
        q = torch.tensor([1., 2.], dtype=torch.float32)
        clean = torch.zeros_like(q)
        y = 3.*2*q
        value = sensor(z, y, clean, q, 0, 1.)
        achieved = float((y-z.A.H(clean+value['gamma']*q)).double().square().mean().sqrt()/.05)
        self.assertAlmostEqual(achieved, 1., places=4)
        self.assertGreater(value['gamma'], 2.)


if __name__ == '__main__':
    unittest.main()
