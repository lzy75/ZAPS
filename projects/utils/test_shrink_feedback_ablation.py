"""A72 necessary pure protocol/decision and real worker-CLI checks."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from utils import shrink_feedback_ablation as rules
from utils import noise_signal_shrink as original


class DesignTests(unittest.TestCase):
    def fixture(self):
        cases = [dict(key=str(i), image_id=str(i), task='inpainting', sigma=.1) for i in range(3)]
        curves = {str(i): [float(i+k)/10 for k in range(30)] for i in range(3)}
        return cases, curves

    def test_donors_exclude_target_and_keep_cyclic_identity(self):
        cases, curves = self.fixture(); a = rules.calibrate(cases, curves)
        self.assertEqual(a['0']['donor_keys'], ['1', '2'])
        self.assertEqual(a['0']['mismatch_donor'], '1')
        self.assertEqual(a['2']['mismatch_donor'], '0')
        self.assertFalse(a['0']['GT_used'])
        changed = dict(curves, **{'0': [999.]*30})
        self.assertEqual(a['0'], rules.calibrate(cases, changed)['0'])
        self.assertAlmostEqual(a['0']['fixed_profile'][0], .15)

    def test_unshrunk_ray_and_patch_do_not_recurse(self):
        choose, state = rules.selector('NO_NOISE', {})
        with patch.object(original, 'select', side_effect=choose):
            row = original.select(3., 4., 9., 100, .1)
        self.assertAlmostEqual(row['gamma'], 1/3)
        self.assertEqual(state['calls'], 1)
        self.assertAlmostEqual(rules.SELECT(3., 4., 9., 100, .1)['gamma'], .25)

    def test_frozen_actions_ignore_target_sensed_values_and_are_unbounded(self):
        cfg = dict(fixed_scalar=3.4, fixed_profile=[2.+i for i in range(30)], mismatch=[7.]*30)
        for arm, expected in (('FIXED_SCALAR', 3.4), ('FIXED_PROFILE', 2.), ('MISMATCH', 7.)):
            one, _ = rules.selector(arm, cfg); two, _ = rules.selector(arm, cfg)
            self.assertEqual(one(1., 2., 3., 10, .1)['gamma'], expected)
            self.assertEqual(two(-.5, 4., 5., 20, .3)['gamma'], expected)
        choose, _ = rules.selector('FIXED_PROFILE', cfg)
        self.assertEqual([choose(1., 2., 3., 10, .1)['gamma'] for _ in range(30)], cfg['fixed_profile'])
        with self.assertRaises(ValueError):
            choose(1., 2., 3., 10, .1)

    def test_worker_cli_reaches_explicit_uuid_without_torch_or_gpu(self):
        root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, 'utils/diag_shrink_feedback_ablation.py', '--worker', '--root', '/tmp/a72-cli-only',
               '--shard', '1', '--device-uuid', 'GPU-test', '--check-worker-cli']
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), dict(shard=1, device_uuid='GPU-test', model_calls=0))


if __name__ == '__main__':
    unittest.main()
