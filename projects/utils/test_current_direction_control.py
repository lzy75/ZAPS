"""A55 direction/norm-law isolation and server Torch checks. No skip-on-missing-Torch."""
import unittest
import numpy as np
from utils import current_direction_control as rules


class NumpyTests(unittest.TestCase):
    def test_norm_rule_and_directions_separate(self):
        q, v = np.array([0., -3., 0.]), np.array([4., 0., 0.])
        a = rules.numpy_target(q, v, .25, 'V_PHYSICAL')
        b = rules.numpy_target(q, v, .25, 'Q_ORIGINAL')
        np.testing.assert_allclose(a, [1., 0., 0.])
        np.testing.assert_allclose(b, [0., -1., 0.])
        self.assertAlmostEqual(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
        np.testing.assert_allclose(b, rules.numpy_target(70*q, v, .25, 'Q_ORIGINAL'))

    def test_bad_direction_rejected(self):
        with self.assertRaises(ValueError):
            rules.numpy_target([0., 0.], [1., 0.], .2, 'Q_ORIGINAL')
        with self.assertRaises(ValueError):
            rules.numpy_target([1., 0.], [float('nan'), 0.], .2, 'V_PHYSICAL')

    def test_paid_case_contract_reuses_four_physical_arms(self):
        from utils import diag_current_direction_value as driver
        p = dict(physical_sources={str(i): 'source' for i in range(4)})
        names = [driver.new_arms(p, dict(key=str(i))) for i in range(10)]
        self.assertEqual(sum(30*len(n) for n in names), 480)
        self.assertEqual(sum('V_PHYSICAL' in n for n in names), 6)
        self.assertTrue(all('Q_ORIGINAL' in n for n in names))

    def test_executor_binding_restores_historical_functions(self):
        from utils import diag_current_direction_value as driver
        engine = driver.engine
        before = (engine.check, engine.item, engine.sample)
        with driver.executor_bindings():
            self.assertIs(engine.sample, driver.sample)
            self.assertIs(engine.item, driver.item)
        self.assertEqual(before, (engine.check, engine.item, engine.sample))


class TorchTests(unittest.TestCase):
    def test_target_matches_numpy_without_vjp_or_input_mutation(self):
        import torch
        from unittest.mock import patch
        q = torch.tensor([[1., -.7, .4]], dtype=torch.float64)
        v = torch.tensor([[.4, .2, -.8]], dtype=torch.float64)
        original = (q.clone(), v.clone())
        reading = rules.base.gram_readout(q, v, None, .2, 1.)
        with patch.object(rules.algebra, 'input_vjp', side_effect=AssertionError('Unexpected VJP')):
            for mode in rules.ARMS:
                result, row = rules.target(q, v, .2, mode, reading)
                np.testing.assert_allclose(result.numpy(), rules.numpy_target(q.numpy(), v.numpy(), .2, mode))
                self.assertLessEqual(row['norm_rule_relative_error'], 2e-6)
                self.assertEqual(row['backward_calls'], 0)
                self.assertFalse(row['current_Jacobian_used'])
        for a, b in zip((q, v), original):
            self.assertTrue(torch.equal(a, b))

    def test_physical_target_matches_a52_at_identical_state(self):
        import torch
        from utils import feedback_vector_constraint as old
        q = torch.tensor([[1., .2]], dtype=torch.float64)
        v = torch.tensor([[.8, -.1]], dtype=torch.float64)
        reading = rules.base.gram_readout(q, v, None, .3, 1.)
        expected, _ = old.target_from_readout(q, v, None, .3, 'FULL_PROX', reading)
        actual, _ = rules.target(q, v, .3, 'V_PHYSICAL', reading)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
