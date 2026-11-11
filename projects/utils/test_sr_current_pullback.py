import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils import diag_sr_current_pullback as driver
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

    def test_bicubic_probe_uses_one_scalar_anchor_and_reuses_current_forward(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D
        from modules.zaps_algorithm import ZAPS
        from utils import equal_nfe_feedback_runtime_r1 as geometry
        from utils.historical_ffhq_runtime import make_operator

        op = make_operator('cpu', size=32)
        model = SimpleNamespace(model=torch.nn.Identity(), device='cpu', alphas_cumprod=torch.tensor([.99, .9]))
        calls = []

        def predict_eps(x, t):
            calls.append(int(t[0]))
            return model.model(.1*x)

        model._predict_eps = predict_eps
        z = SimpleNamespace(tau=torch.tensor([0, 1]), dwt=OrthogonalDWT2D(), zeta=torch.tensor([.1, .1]),
                            D=torch.full((2, 3, 32, 32), .2), surrogate_score_jacobian=False)
        z._tweedie_estimate = ZAPS._tweedie_estimate.__get__(z)
        generator = torch.Generator().manual_seed(17)
        x = .2*torch.randn((1, 3, 32, 32), generator=generator)
        y = .2*torch.randn((1, 3, 8, 8), generator=generator)
        count = {'input_vjp_attempts': 0}
        module = dict(module_dwt_analysis=0, module_dwt_synthesis=0, module_adjoint_actions=0)
        with patch.object(driver, 'scalar_rule', side_effect=AssertionError('Independent scalar solve')), \
                patch.object(geometry, 'exact_adjoint_diagonal', wraps=geometry.exact_adjoint_diagonal) as solve:
            _, _, outputs, _ = driver.probe_position(model, z, op, x, y, 1, .1, count, module)
        solve.assert_called_once()
        self.assertEqual(calls, [1, 0, 0, 0, 0])
        self.assertEqual(set(outputs), set(ARMS))
        self.assertEqual(count, {'input_vjp_attempts': 2, 'current_input_vjp': 1, 'next_input_vjp': 1})
