import unittest

from utils.diag_equal_length_next_response import ARMS, summarize, tangent


class DesignTests(unittest.TestCase):
    def test_per_image_aggregation_and_missing_region_gate(self):
        image_rows = []
        for i in range(3):
            row = dict(full_mse=2., clean_H_mse=1., observed_mse=1., missing_mse=3.)
            better = dict(full_mse=1.5, clean_H_mse=.8, observed_mse=.8, missing_mse=2.2)
            item = dict(active=True, direction_cosine=.3, actual_target_losses=dict(zip(ARMS, (3., 2., 1.))),
                        quality=dict(zip(ARMS, (row, row, better))))
            image_rows.append(dict(case=dict(task='inpainting', image_id=str(i)), positions=[item]*(i+1)))
        result = summarize(image_rows)[0]
        self.assertEqual(result['active_positions'], 6)
        self.assertTrue(all(c['positive'] for c in result['comparisons']))
        self.assertEqual(result['comparisons'][1]['mean_delta']['actual_target_loss'], -1.)


class TorchTests(unittest.TestCase):
    def test_tangent_removes_ray_and_inactive_coordinates(self):
        import torch

        class Identity:
            def analysis(self, x):
                return x

            synthesis = analysis

        class Zaps:
            dwt = Identity()

        q = torch.tensor([1., 2., 0.])
        u = tangent(Zaps(), torch.tensor([4., 3., 9.]), q)
        torch.testing.assert_close(u, torch.tensor([2., -1., 0.]))
        self.assertEqual(float(u @ q), 0.)
