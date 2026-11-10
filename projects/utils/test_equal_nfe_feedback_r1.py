"""A101R1 source and real-Torch repair tests."""
import unittest


class SourceTests(unittest.TestCase):
    def test_repair_has_separate_identity(self):
        from utils import equal_nfe_feedback_r1 as rules
        self.assertEqual(rules.EXPERIMENT, 'A101R1_equal_nfe_feedback')
        self.assertIn('A101R1_20261110', rules.ROOT)
        self.assertNotEqual(rules.BRANCH, 'results/zaps-equal-nfe-feedback-20261110')


class TorchTests(unittest.TestCase):
    def test_prefix_pairing_does_not_require_preexisting_rng_equality(self):
        import torch
        from utils.equal_nfe_feedback_runtime_r1 import independent_prefix_equal
        row = dict(zeta=torch.ones(2), D=torch.ones(2), tau=torch.arange(2),
                   initial=torch.ones(2), cpu_rng=torch.tensor([1], dtype=torch.uint8),
                   cuda_rng=[])
        other = {key: value.clone() if hasattr(value, 'clone') else value
                 for key, value in row.items()}
        other['cpu_rng'] = torch.tensor([2], dtype=torch.uint8)
        gate = independent_prefix_equal(row, other)
        self.assertTrue(gate['passed'])
        self.assertFalse(gate['rng_equal_before_pairing'])
        self.assertEqual(gate['copied_from_baseline'], ['cpu_rng', 'cuda_rng'])

    def test_diagonal_uses_true_base_adjoint_not_empirical_mapping(self):
        import torch
        from utils import equal_nfe_feedback_runtime_r1 as repair
        class DWT:
            def analysis(self, value): return value
            def synthesis(self, value): return value
        class Base:
            def transpose(self, value): return value
        class Operator:
            _mask_cache = None
            base = Base()
            def H(self, value): return value
            def transpose(self, value):
                raise AssertionError('legacy empirical mapping must not be used')
        class Zaps: dwt = DWT()
        counts = dict(module_dwt_analysis=0, module_dwt_synthesis=0,
                      module_adjoint_actions=0)
        delta, detail, _ = repair.exact_adjoint_diagonal(
            Zaps(), Operator(), torch.zeros(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.),
            torch.tensor(.8), .1, counts)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertLessEqual(detail['proxy_final'], detail['proxy_initial'] + 1e-7)


if __name__ == '__main__':
    unittest.main()
