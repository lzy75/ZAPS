import unittest

from utils.diag_sr_current_pullback import ANCHOR, ARMS, CURRENT, NEXT, PROXY, pullback, summarize


class DesignTests(unittest.TestCase):
    def test_image_level_gate_requires_all_equal_length_candidates(self):
        rows = []
        for i in range(3):
            error = {ANCHOR:2., PROXY:2.2, CURRENT:1.8, NEXT:1.6}
            point = dict(active={arm:True for arm in ARMS[1:]}, cosine_to_next={PROXY:0., CURRENT:.6},
                         quality={arm:dict(psnr=30.-e, full_mse=e, clean_H_mse=e, noisy_y_mse=e)
                                  for arm, e in error.items()}, actual_target_losses=error, virtual_clean_H_mse=error)
            rows.append(dict(case=dict(image_id=str(i)), positions=[dict(point) for _ in range(3)]))
        self.assertTrue(summarize(rows)['comparisons'][0]['positive'])
        rows[0]['positions'][0]['active'] = {PROXY:True, CURRENT:False, NEXT:True}
        self.assertFalse(summarize(rows)['comparisons'][0]['positive'])


class TorchTests(unittest.TestCase):
    def test_pullback_freezes_target_error_and_does_not_populate_input_grad(self):
        import torch

        inp = torch.tensor([1., 2.], requires_grad=True)
        error = 3.*inp
        count = {'input_vjp_attempts': 0}
        direction = pullback(inp.square(), inp, error, count, 'current_input_vjp')
        torch.testing.assert_close(direction, torch.tensor([-6., -24.]))
        self.assertIsNone(inp.grad)
        self.assertEqual(count, {'input_vjp_attempts': 1, 'current_input_vjp': 1})
