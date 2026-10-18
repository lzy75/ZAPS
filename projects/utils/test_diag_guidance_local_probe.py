"""No GT selection, fixed budget/probes, and analytic finite-difference checks."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from utils import diag_guidance_local_probe as audit
from utils.test_diag_cached_direction import trace_fixture


def pair_fixture(eta, phi=-2., s_dir=3.):
    return dict(eta=eta, input_resolved=True, direction_chord=2*eta*s_dir,
                s_dir=s_dir, loss_difference=2*eta*phi, phi=phi)


class ProbeContractTests(unittest.TestCase):
    def test_snapshot_selection_never_uses_quality(self):
        rows = trace_fixture()['rows']
        first = audit.select_snapshots(rows)
        self.assertEqual([p['target_u'] for p in first], list(audit.TARGETS))
        self.assertEqual(len({p['step'] for p in first}), 5)
        changed = copy.deepcopy(rows)
        for row in changed: row.update(psnr=100-row['step'], lpips=row['step'])
        self.assertEqual(audit.select_snapshots(changed), first)

    def test_duplicate_or_missing_time_points_are_rejected(self):
        rows = trace_fixture()['rows']
        for k, row in enumerate(rows): row['noise_u'] = k*.00001
        with self.assertRaisesRegex(ValueError, 'duplicate'): audit.select_snapshots(rows)
        rows[3]['noise_u'] = rows[2]['noise_u']
        with self.assertRaises(ValueError): audit.select_snapshots(rows)
        with self.assertRaises(ValueError): audit.select_snapshots(rows[:-1])

    def test_direction_readability_does_not_imply_descent(self):
        floors = dict(direction_chord=0., loss_abs=0.)
        down = audit.classify_snapshot([pair_fixture(e) for e in audit.ETAS], floors)
        up = audit.classify_snapshot([pair_fixture(e, phi=2.) for e in audit.ETAS], floors)
        self.assertEqual(down['loss_status'], 'LOCAL_FINITE_SCALE_DESCENT')
        self.assertEqual(up['loss_status'], 'LOCAL_FINITE_SCALE_ASCENT')
        self.assertTrue(up['usable'])  # Measurable ascent is a result, not silently discarded.

    def test_scale_dependence_and_unresolved_signal_not_cherry_picked(self):
        pairs = [pair_fixture(.01), pair_fixture(.02, phi=2., s_dir=9.)]
        result = audit.classify_snapshot(pairs, dict(direction_chord=0., loss_abs=0.))
        self.assertFalse(result['usable'])
        self.assertEqual(result['direction_status'], 'INCONCLUSIVE_SCALE_DEPENDENT')
        self.assertEqual(result['loss_status'], 'INCONCLUSIVE_SCALE_DEPENDENT')
        small = audit.classify_snapshot([pair_fixture(e) for e in audit.ETAS],
                                        dict(direction_chord=1., loss_abs=1.))
        self.assertFalse(small['usable']); self.assertEqual(small['loss_status'], 'INCONCLUSIVE_SMALL_SIGNAL')
        with self.assertRaises(ValueError): audit.classify_snapshot(pairs[:1], dict(direction_chord=0., loss_abs=0.))

    def test_input_rounding_failure_is_not_treated_as_low_sensitivity(self):
        pairs = [pair_fixture(e) for e in audit.ETAS]; pairs[0]['input_resolved'] = False
        result = audit.classify_snapshot(pairs, dict(direction_chord=0., loss_abs=0.))
        self.assertEqual(result['direction_status'], 'INCONCLUSIVE_INPUT_REPRESENTATION')

    def test_failed_geometry_cannot_unlock_probe(self):
        fixture = dict(experiment='cached_guidance_geometry', status='failed', method_nfe=0,
                       diagnostic_nfe=0, model_loaded=False, optimizer_steps=0)
        with self.assertRaises(ValueError): audit.validate_geometry(fixture)

    def test_missing_direction_tensor_is_rejected_before_torch_or_checkpoint_load(self):
        required = ('identity_files_unchanged', 'cached_steps', 'correction_repeat',
                    'noise_draw_suffix', 'residual_clean_and_loss')
        fixture = dict(experiment='cached_guidance_geometry', status='complete', method_nfe=0,
                       diagnostic_nfe=0, model_loaded=False, optimizer_steps=0,
                       gates={k: dict(passed=True) for k in required}, rows=trace_fixture()['rows'],
                       summary=dict(classification='READABLE_GUIDANCE_DIRECTION_CHANGE', guidance_valid=30),
                       geometry_sha256='0'*64)
        audit.validate_geometry(fixture)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'run.json').write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, 'No completed compatible'):
                audit.select_geometry(folder)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; numerical probes await server')
class InputTensorTests(unittest.TestCase):
    def test_fp32_unrepresentable_perturbation_and_input_immutability(self):
        import torch
        x, q = torch.tensor([1e8], dtype=torch.float32), torch.tensor([.1], dtype=torch.float32)
        original = x.clone(); xp, xm, record = audit.perturb_inputs(x, q, .01)
        self.assertFalse(record['input_resolved']); self.assertEqual(record['effective_span'], 0.)
        self.assertTrue(torch.equal(x, original)); self.assertTrue(torch.equal(xp, xm))


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'pywt', 'numpy')),
                     'Real ZAPS numeric check requires PyTorch/PyWavelets; await server')
class RealFieldTensorTests(unittest.TestCase):
    def test_anisotropic_quadratic_has_known_descent_and_six_forwards(self):
        import torch
        from modules.zaps_algorithm import ZAPS
        class Operator(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer('a', torch.tensor([.5]*4+[2.]*4).reshape(1, 1, 1, 8))
            def H(self, x): return self.a*x
            def transpose(self, r): return self.a*r
        calls = []
        def predict(x, tb): calls.append(int(tb[0])); return torch.zeros_like(x)
        dm = SimpleNamespace(device='cpu', num_steps=1000, alphas_cumprod=torch.ones(1000), _predict_eps=predict)
        z = ZAPS(dm, Operator(), num_steps=4, schedule=(2, 1, 1), channels=1, img_size=8,
                 wave='haar', level=1, use_learned_var=False, sampler_mode='ddpm', zeta_init=.1)
        x = torch.linspace(-.3, .3, 64).reshape(1, 1, 8, 8); y = torch.zeros_like(x)
        rng = torch.get_rng_state().clone(); parameters = z.zeta.detach().clone()
        first = audit.evaluate_field(z, y, x, 999, 3)
        second = audit.evaluate_field(z, y, x, 999, 3)
        q = first['q']; pairs = []
        for eta in audit.ETAS:
            xp, xm, inp = audit.perturb_inputs(x, q, eta)
            plus = audit.evaluate_field(z, y, xp, 999, 3)
            minus = audit.evaluate_field(z, y, xm, 999, 3)
            pair = audit.pair_readout(plus, minus, inp, q, eta, first); pairs.append(pair)
            exact_phi = -float(q.double().square().sum())/(float(z.zeta[3])*x.numel()*.05**2)
            self.assertAlmostEqual(pair['phi'], exact_phi, delta=.001*abs(exact_phi))
            self.assertGreater(pair['s_dir'], 0.); self.assertEqual(pair['plus_clip_state_changed'], 0.)
        result = audit.classify_snapshot(pairs, dict(direction_chord=0., loss_abs=abs(first['loss']-second['loss'])))
        self.assertTrue(result['usable']); self.assertEqual(result['loss_status'], 'LOCAL_FINITE_SCALE_DESCENT')
        self.assertEqual(calls, [999]*6)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(parameters, z.zeta.detach()))


if __name__ == '__main__': unittest.main()
