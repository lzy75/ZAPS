import unittest
import numpy as np
from utils.trajectory_state_geometry import (path_geometry, high_part, History, cross_validate,
                                            action_audit, FEATURES, POINTS, FACTORS, CALLS)


class GeometryTests(unittest.TestCase):
    def test_straight_path_and_reversal(self):
        line = [np.array([v, 0.]) for v in (0., 1., 2.)]
        straight = path_geometry(line, .2)
        self.assertEqual(straight['curvature'], 0.)
        self.assertEqual(straight['backtrack'], 0.)
        bent = path_geometry([line[0], line[1], line[0]], .2)
        self.assertGreater(bent['curvature'], 0.)
        self.assertEqual(bent['backtrack'], 1.)
        self.assertFalse(path_geometry([line[0]]*3, .2)['resolved'])

    def test_block_projection_is_orthogonal_not_bicubic_nullspace(self):
        x = np.arange(192, dtype=float).reshape(3, 8, 8)
        high = high_part(x); low = x-high
        self.assertAlmostEqual(float((high*low).sum()), 0.)
        np.testing.assert_allclose(high_part(high), high)
        np.testing.assert_allclose(high_part(low), 0.)

    def test_history_is_causal_and_reports_small_motion_missing(self):
        h = History(); q = np.ones((3, 8, 8), dtype=np.float32)
        readout = dict(R=2., cosine=.9, relative_response=.5)
        for k in range(3):
            result = h.append(.1*k, k*q, q, -q, 1., readout, 0.)
        self.assertEqual(result['features']['prior_guide_cos'], -1.)
        self.assertEqual(result['features']['log_curvature'], 0.)
        self.assertIsNone(result['features']['log_high_curvature'])
        frozen = dict(result['features'])
        h.append(.3, 999.*q, q, q, 1., readout, 0.)
        self.assertEqual(result['features'], frozen)
        self.assertEqual(CALLS, 108)

    def test_held_image_labels_cannot_affect_its_own_prediction(self):
        rows = [dict(image_id=str(i), seed=s, step=k, features={'x': float(i+k)},
                     labels={'psnr': .1*i+.2*k}) for i in range(4) for s in (1, 2) for k in (8, 19)]
        first = cross_validate(rows, ('x',), 'psnr')['predictions']
        for r in rows:
            if r['image_id'] == '0':r['labels']['psnr'] += 100.
        second = cross_validate(rows, ('x',), 'psnr')['predictions']
        self.assertEqual([r['prediction'] for r in first if r['image_id'] == '0'],
                         [r['prediction'] for r in second if r['image_id'] == '0'])

    def test_action_choice_uses_predictions_not_held_outcome_and_keep_is_available(self):
        rows, predictions = {}, {}
        for f in FACTORS:
            rows[f] = [dict(image_id=str(i), seed=s, step=k,
                            labels=dict(psnr=-1. if f == .5 else 1., ssim=0., lpips=0., loss=0.))
                       for i in range(3) for s in (1, 2) for k, _ in POINTS]
            predictions[f] = {name: [dict(image_id=r['image_id'], seed=r['seed'], step=r['step'],
                                         prediction=-.1 if r['image_id'] == '1' else .2)
                                     for r in rows[f]] for name in FEATURES}
        result = action_audit(rows, predictions)['models']['CURVE']
        self.assertEqual(result['counts'], {'1.0': 10, '0.5': 20, '1.5': 0})
        self.assertLess(result['mean_delta']['psnr'], 0.)  # Does not pick the GT-winning arm.
        for items in rows.values():
            for r in items:r['labels']['psnr'] *= -1.
        updated = action_audit(rows, predictions)['models']['CURVE']
        self.assertEqual([r['factor'] for r in result['choices']], [r['factor'] for r in updated['choices']])


if __name__ == '__main__':
    unittest.main()
