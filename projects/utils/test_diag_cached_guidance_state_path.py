"""State path geometry, zero-path retention and matched-t pairing."""
import math
import unittest

from utils import diag_cached_guidance_state_path as diagnostic

try:
    import torch
except ImportError:
    torch = None


class ScalarTests(unittest.TestCase):
    def test_factorization_separates_state_speed_from_noise_coordinate(self):
        base = diagnostic.path_decomposition(.6, .2, .1)
        faster = diagnostic.path_decomposition(.6, .4, .1)
        dense = diagnostic.path_decomposition(.6, .2, .05)
        self.assertAlmostEqual(base['omega3_recomputed'], 6.)
        self.assertAlmostEqual(base['direction_path_secant3'], 3.)
        self.assertAlmostEqual(faster['omega3_recomputed'], base['omega3_recomputed'])
        self.assertAlmostEqual(faster['direction_path_secant3'], base['direction_path_secant3']/2)
        self.assertAlmostEqual(dense['direction_path_secant3'], base['direction_path_secant3'])
        self.assertAlmostEqual(dense['state_speed3'], 2*base['state_speed3'])
        for result in (base, faster, dense):
            self.assertAlmostEqual(result['factorization_error'], 0.)

    def test_zero_state_path_retained_even_if_field_turns(self):
        result = diagnostic.path_decomposition(1., 0., .1)
        self.assertEqual(result['omega3_recomputed'], 10.)
        self.assertIsNone(result['direction_path_secant3'])
        self.assertIsNone(result['factorization_error'])
        still = diagnostic.path_decomposition(0., 1., .1)
        self.assertEqual(still['direction_path_secant3'], 0.)

    def test_invalid_arc_and_coordinate_rejected(self):
        for values in ((-1., 1., .1), (1., -1., .1), (1., 1., 0.), (math.nan, 1., .1)):
            with self.assertRaises(ValueError):
                diagnostic.path_decomposition(*values)

    def test_associations_keep_times_scales_and_three_clusters(self):
        rows = [dict(case=str(i), image_id=str(i//2), t=t, omega3_cached=i,
                     direction_path_secant3=i, state_speed3=6-i, incoming_q_abs_cosine=None,
                     kappa_dir_01=i, kappa_dir_02=2*i)
                for t in diagnostic.TIMES for i in range(6)]
        links = diagnostic.associations(rows)
        self.assertEqual(len(links), 40)
        self.assertEqual(diagnostic.hypothesis_readout(links), 'EXPLORATORY_POSITIVE_ORDERING_ONLY')
        for item in links:
            if item['predictor'] == 'incoming_q_abs_cosine':
                self.assertEqual(item['n'], 0)
                self.assertIsNone(item['spearman'])
            else:
                self.assertEqual(item['image_clusters'], 3)
                self.assertAlmostEqual(item['spearman'], -1. if item['predictor'] == 'state_speed3' else 1.)
        links[2]['spearman'] = -.5
        self.assertEqual(diagnostic.hypothesis_readout(links), 'UNIFORM_POSITIVE_PROXY_NOT_SUPPORTED')
        links[2]['n'] = 5
        self.assertEqual(diagnostic.hypothesis_readout(links), 'INCONCLUSIVE_ORDERING')
        with self.assertRaises(ValueError):
            diagnostic.associations(rows[:-1])


@unittest.skipIf(torch is None, 'PyTorch unavailable; tensor geometry must run on server')
class TensorTests(unittest.TestCase):
    def fixture(self):
        q = torch.eye(3)
        x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        rows = [dict(t=t, noise_u=i*.1) for i, t in enumerate((999, 641, 319))]
        rows[2].update(guidance_variation3_per_u=2*math.sqrt(2)/.2,
                       residual_noise_ratio=1.3, raw_x0_clip_fraction=0.)
        probe = dict(step=2, t=319, q_rms=1/math.sqrt(3),
                     pairs=[dict(eta=e, s_dir=.1, s_dir_per_state_rms=.1*math.sqrt(3)) for e in (.01, .02)])
        return q, x, rows, probe

    def test_actual_axis_and_state_rms_not_l2_distance(self):
        q, x, rows, probe = self.fixture()
        result = diagnostic.point_readout(q, x, rows, probe)
        self.assertAlmostEqual(result['incoming_q_abs_cosine'], 0.)
        self.assertAlmostEqual(result['state_arc3_rms'], 2/math.sqrt(3))
        self.assertAlmostEqual(result['direction_path_secant3'], math.sqrt(6))
        self.assertAlmostEqual(result['factorization_error'], 0.)

    def test_zero_path_not_dropped_and_source_omega_mismatch_rejected(self):
        q, x, rows, probe = self.fixture()
        result = diagnostic.point_readout(q, x*0, rows, probe)
        self.assertIsNone(result['direction_path_secant3'])
        self.assertIsNone(result['incoming_q_cosine'])
        rows[2]['guidance_variation3_per_u'] = 1.
        with self.assertRaisesRegex(ValueError, 'Omega3'):
            diagnostic.point_readout(q, x, rows, probe)


if __name__ == '__main__':
    unittest.main()
