import unittest


class TorchTests(unittest.TestCase):
    def test_prefix_tolerance_and_straight_through(self):
        import torch
        from utils import equal_nfe_feedback_runtime_r3 as runtime
        row = dict(zeta=torch.ones(2), D=torch.ones(2), tau=torch.arange(2),
                   initial=torch.ones(2), cpu_rng=torch.tensor([1], dtype=torch.uint8), cuda_rng=[])
        other = {key: value.clone() if hasattr(value, 'clone') else value for key, value in row.items()}
        other['zeta'][0] += 2e-6; other['cpu_rng'][0] = 2
        self.assertTrue(runtime.prefix_audit(row, other)['passed'])
        q = torch.ones(2, requires_grad=True); closed = torch.tensor([3., 4.])
        value = closed + q - q.detach()
        torch.testing.assert_close(value.detach(), closed)
        value.sum().backward(); torch.testing.assert_close(q.grad, torch.ones_like(q))


if __name__ == '__main__': unittest.main()
