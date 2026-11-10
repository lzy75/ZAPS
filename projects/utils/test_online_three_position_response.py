import unittest

from utils.diag_online_three_position_response import ARMS, TRUE_DIAG, TRUE_RAY, apply_length, summarize


class DesignTests(unittest.TestCase):
    def test_same_budget_direction_comparison_checks_actual_actions(self):
        rows = []
        for i in range(3):
            arms = {}
            for name in ARMS:
                gain = float(name == TRUE_DIAG)
                arms[name] = dict(quality=dict(psnr=25+gain, ssim=.8+.1*gain, lpips=.2-.1*gain),
                    errors=dict(clean_H_mse=2.-gain, observed_mse=2.-gain, missing_mse=3.-gain),
                    nfe=303 if name in (TRUE_DIAG, TRUE_RAY) else 300,
                    input_vjp=3 if name in (TRUE_DIAG, TRUE_RAY) else 0,
                    applied_steps=3, method_seconds=15)
            rows.append(dict(case=dict(task='inpainting', image_id=str(i)), arms=arms))
        self.assertTrue(summarize(rows)[0]['comparisons'][-1]['positive'])
        rows[0]['arms'][TRUE_RAY]['applied_steps'] = 2
        self.assertFalse(summarize(rows)[0]['comparisons'][-1]['positive'])


class TorchTests(unittest.TestCase):
    def test_fixed_length_and_zero_direction(self):
        import torch
        vector = torch.tensor([3., 4.])
        update, applied = apply_length(vector, .5)
        self.assertTrue(applied)
        torch.testing.assert_close(update, vector/10)
        self.assertFalse(apply_length(vector*0, .5)[1])
