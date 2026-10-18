"""Geometric semantics, source rejection, and actual-H gradient isolation."""
import copy
import importlib.util
import math
import tempfile
from pathlib import Path
import unittest

from utils import diag_cached_guidance_geometry as audit
from utils.test_diag_cached_direction import trace_fixture


class GeometryContractTests(unittest.TestCase):
    def test_round_trip_and_constant_directions_have_distinct_trends(self):
        r = audit.trend_geometry(2., 2., 0., .4)
        self.assertEqual(r['backtrack3'], 1.)
        self.assertEqual(r['variation3_per_u'], 10.)
        constant = audit.trend_geometry(0., 0., 0., .4)
        self.assertIsNone(constant['backtrack3'])
        self.assertTrue(constant['no_resolved_turn3'])

    def test_small_consistent_rotation_has_little_backtracking(self):
        theta = .05
        chord = 2*math.sin(theta/2)
        r = audit.trend_geometry(chord, chord, 2*math.sin(theta), .5)
        self.assertLess(r['backtrack3'], .001)
        scaled_time = audit.trend_geometry(chord, chord, 2*math.sin(theta), 1.)
        self.assertEqual(scaled_time['backtrack3'], r['backtrack3'])
        self.assertAlmostEqual(scaled_time['variation3_per_u'], r['variation3_per_u']/2)

    def test_invalid_chords_and_time_are_not_silently_clamped(self):
        for values in [(1., 1., 3., .1), (-1., 1., 0., .1),
                       (1., 1., 1., 0.), (float('nan'), 1., 0., .1)]:
            with self.assertRaises(ValueError): audit.trend_geometry(*values)

    def test_source_protocol_does_not_accept_failed_or_reindexed_trace(self):
        from utils.diag_cached_direction import validate_trace
        source = trace_fixture(); validate_trace(source)
        for change in ('failed', 'reindexed'):
            altered = copy.deepcopy(source)
            if change == 'failed': altered['status'] = 'failed'
            else: altered['rows'][4]['parameter_index'] = 4
            with self.assertRaises(ValueError): validate_trace(altered)

    def test_resolve_source_rejects_changed_parent_before_tensor_or_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)/'run.json'; parent.write_text('{}')
            trace = {'source': str(parent), 'source_sha256': '0'*64}
            with self.assertRaisesRegex(RuntimeError, 'parent run changed'):
                audit.resolve_source(Path(folder)/'trace.json', trace)

    def test_summary_keeps_missing_neighbors_and_unequal_response_intervals(self):
        rows = trace_fixture()['rows']
        for r in rows:
            for mode in ('guidance', 'physical'):
                r.update({mode+'_valid': True, mode+'_cosine': None if r['step'] == 0 else .2,
                          mode+'_turn': None if r['step'] == 0 else 1., mode+'_turn_per_u': 29.,
                          mode+'_variation3_per_u': None if r['step'] < 2 else 29.,
                          mode+'_backtrack3': None if r['step'] < 2 else .1, mode+'_rms': .5})
        rows[7]['guidance_cosine'] = None
        summary = audit.summarize(rows, 0.)
        self.assertEqual(summary['guidance_valid'], 30)
        self.assertEqual(summary['associations']['guidance_cosine_vs_next_psnr_per_u']['n'], 27)
        self.assertEqual(summary['associations']['guidance_backtrack3_vs_next_psnr_per_u']['n'], 27)
        self.assertFalse(summary['local_input_sensitivity_measured'])
        self.assertEqual(summary['controller_suitability'], 'NOT_ESTABLISHED')
        with self.assertRaises(ValueError): audit.summarize(rows[:-1], 0.)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable: tensor checks await server')
class TensorGeometryTests(unittest.TestCase):
    def test_forward_operator_gradient_ignores_deliberately_wrong_transpose(self):
        import torch
        class Operator:
            def H(self, z): return 2*z+1
            def transpose(self, r): raise AssertionError('Empirical transpose is NOT the true gradient')
        clean = torch.tensor([[[[.1, -.2]]]], dtype=torch.float32)
        y = torch.tensor([[[[.5, 1.5]]]], dtype=torch.float32)
        original = clean.clone(); rng = torch.get_rng_state().clone()
        with torch.no_grad(): descent, loss = audit.physical_descent(Operator(), clean, y)
        expected = 2*(y-(2*clean+1))/(clean.numel()*.05**2)
        torch.testing.assert_close(descent, expected)
        self.assertAlmostEqual(loss, float(((2*clean+1-y).double().square().mean()/(2*.05**2))), places=9)
        self.assertTrue(torch.equal(clean, original)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(descent.requires_grad)

    def test_amplitude_rescaling_preserves_direction_but_not_step_magnitude(self):
        import torch
        a = torch.tensor([1., 0.], dtype=torch.float64)
        b = torch.tensor([0., 1.], dtype=torch.float64)
        one = audit.vector_readouts(a, b, a, .5, .1, 0.)
        tiny = audit.vector_readouts(a*.001, b*.001, a*.001, .5, .1, 0.)
        for key in ('cosine', 'turn', 'variation3_per_u', 'backtrack3'):
            self.assertAlmostEqual(one[key], tiny[key])
        self.assertAlmostEqual(one['rms']*.001, tiny['rms'])
        self.assertEqual(one['backtrack3'], 1.)

    def test_zero_direction_does_not_bridge_original_neighbors(self):
        import torch
        a, zero = torch.tensor([1., 0.]), torch.tensor([0., 0.])
        result = audit.vector_readouts(a, zero, a, .5, .1, 0.)
        self.assertTrue(result['valid']); self.assertIsNone(result['cosine'])
        self.assertIsNone(result['backtrack3'])
        self.assertFalse(audit.vector_readouts(zero, a, a, .5, .1, 0.)['valid'])

    def test_unordered_time_and_nonfinite_vectors_fail(self):
        import torch
        a = torch.tensor([1., 0.])
        with self.assertRaises(ValueError): audit.vector_readouts(a, a, a, .1, .1, 0.)
        with self.assertRaises(ValueError): audit.unit_direction(torch.tensor([float('nan')]))


if __name__ == '__main__': unittest.main()
