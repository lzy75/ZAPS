"""A56 frozen selection/profile/cost checks, without running a network."""
import unittest
from utils import frozen_direction_transfer as rules


class DesignTests(unittest.TestCase):
    def test_history_and_duplicate_exclusion_reproducible(self):
        pool = [dict(image_id=str(i), image_sha256=str(i//2)) for i in range(30)]
        a = rules.select(pool, {'2'}, {'3'}, 6, 56)
        self.assertEqual(a, rules.select(pool, {'2'}, {'3'}, 6, 56))
        self.assertEqual(len({c['image_sha256'] for c in a}), 6)
        self.assertNotIn('2', {c['image_id'] for c in a})
        self.assertNotIn('3', {c['image_sha256'] for c in a})

    def test_budget_counts_every_replay_and_probe(self):
        grid = list(range(30))
        grid[-1] = 999
        ordinary, probed = rules.expected_calls(grid, False), rules.expected_calls(grid, True)
        self.assertEqual(len(ordinary), 180)
        self.assertEqual(len(probed), 192)
        self.assertEqual(20*300 + 18*len(ordinary)+2*len(probed), 9624)

    def test_old_frozen_profile_has_no_target_arguments(self):
        cases = [dict(task=t, image_id=str(i), key=t+str(i)) for t in rules.TASKS for i in range(5)]
        traces = {c['key']: [float(int(c['image_id'])+1)]*30 for c in cases}
        profiles = rules.frozen_profiles(cases, traces)
        self.assertTrue(all(v == [3.]*30 for v in profiles.values()))
        with self.assertRaises(ValueError):
            rules.frozen_profiles(cases[:-1], traces)

    def test_baseline_injection_restores_historical_functions(self):
        from utils import diag_frozen_direction_transfer as driver
        before = driver.a40.done, driver.a40.attempt
        with driver.baseline_bindings():
            self.assertIs(driver.a40.done, driver.baseline_done)
            self.assertIs(driver.a40.attempt, driver.strict_attempt)
        self.assertEqual(before, (driver.a40.done, driver.a40.attempt))


if __name__ == '__main__':
    unittest.main()
