import unittest
from utils.continuous_response_feedback import select, stage_profile, profile_signature, POLICIES


class ContinuousTests(unittest.TestCase):
    def test_exact_quadratic_minimum_and_caps(self):
        # r=10, a=1: analytic optimum is gamma=10, well outside the old action set.
        for name, upper in POLICIES.items():
            value = select(10., 100., 1., upper)
            self.assertAlmostEqual(value['raw_gamma'], 10.)
            self.assertEqual(value['gamma'], 10. if upper is None else upper)
            self.assertEqual(value['cap_hit'], upper is not None)
        value = select(.31, .31**2, 1.)
        self.assertAlmostEqual(value['gamma'], .31)
        self.assertEqual(select(-1., 1., 1.)['gamma'], 0.)

    def test_deadband_and_unresolved_are_explicit(self):
        self.assertEqual(select(1., 1., 1.)['gamma'], 1.)
        self.assertTrue(select(1., 1., 1.)['fallback'])
        self.assertFalse(select(0., 0., 1.)['resolved'])
        self.assertEqual(select(0., 0., 1.)['gamma'], 1.)
        for args in ((float('nan'), 1., 1.), (1., -1., 1.)):
            with self.assertRaises(ValueError): select(*args)
        with self.assertRaises(ValueError): select(1., 1., 1., .5)

    def test_continuous_stage_excludes_all_held_seeds(self):
        def entry(image, gamma):
            return dict(image_id=image, traces={'I_free': [dict(gamma=gamma)]*30})
        profiles = {'a1': entry('a', 100.), 'a2': entry('a', 200.),
                    'b1': entry('b', .28), 'c1': entry('c', .34)}
        reference = stage_profile(profiles, 'a', 'I_free')
        self.assertAlmostEqual(reference[0], .31)
        profiles['a1'] = entry('a', 999.)
        self.assertEqual(reference, stage_profile(profiles, 'a', 'I_free'))

    def test_resuming_another_range_does_not_change_a_frozen_time_profile(self):
        data = dict(plan_sha256='frozen', aggregation='median', values={
            'case': {'STAGE_cont2': [.31]*30}}, sources={'case/I_cont2': {'sha256': 'a'}})
        signature = profile_signature(data, 'I_cont2')
        data['values']['case']['STAGE_free'] = [30.]*30
        data['sources']['case/I_free'] = {'sha256': 'b'}
        self.assertEqual(signature, profile_signature(data, 'I_cont2'))
        data['values']['case']['STAGE_cont2'][0] = .32
        self.assertNotEqual(signature, profile_signature(data, 'I_cont2'))

    def test_torch_linear_sensor_and_fractional_update(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable locally; mandatory on server')
        from types import SimpleNamespace
        from utils.continuous_response_feedback import sensor
        z = SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.ones(1)),
                            A=SimpleNamespace(H=lambda x: 2*x))
        q = torch.tensor([1., 2.], dtype=torch.float32)
        clean = torch.zeros_like(q)
        y = .31 * 2*q
        response = sensor(z, y, clean, q, 0, None)
        self.assertAlmostEqual(response['gamma'], .31, places=6)
        next_state = clean + response['gamma']*q
        self.assertLess(float((y-z.A.H(next_state)).abs().max()), 1e-6)
        self.assertAlmostEqual(response['cosine'], 1., places=6)


if __name__ == '__main__':
    unittest.main()
