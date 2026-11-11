import unittest

from utils.diag_guarded_next_response import decision


class GuardTests(unittest.TestCase):
    def test_accepts_only_a_resolved_nonzero_descent(self):
        self.assertEqual(decision(1., .9, True), 'accepted')
        self.assertEqual(decision(1., 1.1, True), 'increased')
        self.assertEqual(decision(1., 1.-1e-8, True), 'within_tolerance')
        self.assertEqual(decision(1., .9, False), 'zero_step')

    def test_invalid_loss_is_a_numerical_failure(self):
        with self.assertRaises(ValueError):
            decision(1., float('nan'), True)
