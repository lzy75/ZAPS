"""A57 budget/independent noise replay checks; missing Torch is not skipped."""
import unittest
from utils.no_learning_replay import burn_count


class DesignTests(unittest.TestCase):
    def test_original_noise_offset_and_budget(self):
        self.assertEqual(burn_count(), 261)
        self.assertEqual(1+burn_count()+29, 291)
        self.assertEqual(10*30, 300)
        with self.assertRaises(ValueError):
            burn_count(0)

    def test_executor_binding_restored(self):
        from utils import diag_no_learning_replay as d
        before = d.engine.check, d.engine.item, d.engine.sample
        with d.executor_bindings():
            self.assertIs(d.engine.sample, d.sample)
        self.assertEqual(before, (d.engine.check, d.engine.item, d.engine.sample))


class TorchTests(unittest.TestCase):
    def test_independent_seed_recipe_matches_original_ten_rounds(self):
        import torch
        from utils.no_learning_replay import noise_from_seed
        from utils.diag_learning_rate_ablation import set_seed
        with torch.random.fork_rng(devices=[]):
            set_seed(1000); expected_initial = torch.randn(1, 3, 4, 4)
            for _ in range(9):
                for _ in range(29):
                    torch.randn_like(expected_initial)
            expected_before = torch.get_rng_state()
            expected = [torch.randn_like(expected_initial) for _ in range(29)]
            actual_initial, actual, before = noise_from_seed(1000, 'cpu', (1, 3, 4, 4))
            self.assertTrue(torch.equal(actual_initial, expected_initial))
            self.assertTrue(torch.equal(before['cpu'], expected_before))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))


if __name__ == '__main__':
    unittest.main()
