import unittest

from utils.diag_300nfe_feedback_confirmation import ObservableProjection


class TorchTests(unittest.TestCase):
    def test_projected_increment_preserves_native_observation(self):
        import torch
        from utils.historical_ffhq_runtime import make_operator

        op = make_operator('cpu')
        projector = ObservableProjection(op, 'cpu')
        generator = torch.Generator().manual_seed(17)
        extra = torch.randn((1, 3, 256, 256), generator=generator)
        row = projector.row(extra)
        torch.testing.assert_close(op.H(row), op.H(extra), rtol=3e-6, atol=1e-6)
        torch.testing.assert_close(projector.row(row), row, rtol=3e-6, atol=1e-6)
        self.assertGreater(float((extra-row).norm()), 1.)
