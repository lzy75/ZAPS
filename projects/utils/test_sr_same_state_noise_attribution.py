import unittest

from utils.diag_sr_same_state_noise_attribution import ScalarTrajectoryProbe, measurement_deltas


class NumPyTests(unittest.TestCase):
    def test_noise_can_reverse_preference_at_a_fixed_state(self):
        row = measurement_deltas([0.], [0.], [1.], [1.], [0.], 1.)
        self.assertEqual(row['proxy'], -1.)
        self.assertEqual(row['clean'], 1.)
        self.assertEqual(row['noise_credit'], 2.)
        self.assertEqual(row['proxy_without_noise'], 1.)

    def test_anchor_credit_is_separate_from_noise(self):
        row = measurement_deltas([2.], [0.], [1.], [0.], [0.], .5)
        self.assertEqual(row['proxy'], -1.)
        self.assertEqual(row['clean'], 1.)
        self.assertEqual(row['noise_credit'], 0.)
        self.assertEqual(row['anchor_credit'], 2.)


class TorchTests(unittest.TestCase):
    def test_probe_advances_scalar_and_records_both_predictions(self):
        import torch

        class Identity:
            def H(self, x):
                return x

        def diagonal(zaps, operator, clean, q, residual, alpha, sigma, counts):
            return 2*q, {'scalar': {'gamma': .5}}, counts

        q = torch.tensor([2.], requires_grad=True)
        probe = ScalarTrajectoryProbe(diagonal)
        chosen, _, _ = probe.update(None, Identity(), torch.zeros(1), q, torch.zeros(1),
                                    torch.ones(1), .1, {})
        torch.testing.assert_close(chosen, torch.ones(1))
        self.assertTrue(chosen.requires_grad)
        self.assertEqual([x.item() for x in probe.measurements[0]], [0., 1., 4.])
