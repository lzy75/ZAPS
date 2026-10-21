"""A29 zero-call audit checks: GT isolation, whole-image holdout and signed inputs."""
import copy
from pathlib import Path
import tempfile
import unittest
from utils import diag_cached_stage_action_audit as d


class AuditTests(unittest.TestCase):
    def test_state_physical_choice_never_reads_quality(self):
        actions = [dict(gamma=g, mean=dict(loss=loss), stochastic=dict(psnr=psnr))
                   for g, loss, psnr in ((.5, 3., 50.), (1., 2., 40.), (2., 1., 10.))]
        self.assertEqual(d.choose(actions, ('mean', 'loss')), 2.)
        for a in actions:
            a['stochastic']['psnr'] *= -100
        self.assertEqual(d.choose(actions, ('mean', 'loss')), 2.)
        self.assertEqual(d.choose([dict(gamma=g, mean=dict(loss=1.)) for g in (.5, 1., 2.)], ('mean', 'loss')), 1.)

    def test_stage_calibration_holds_out_both_seeds_and_conditions_on_span(self):
        points = []
        for image in ('a', 'b', 'c'):
            for seed in (1000, 2000):
                actions = []
                for factor in (.5, 1., 1.5):
                    for gamma in (.5, 1., 2.):
                        good = gamma == (1. if factor == 1. else .5)
                        psnr = 100. if (image == 'c' and gamma == 2.) else 20.+good
                        actions.append(dict(factor=factor, gamma=gamma, mean=dict(loss=1.-.1*good), stochastic=dict(psnr=psnr)))
                points.append(dict(image_id=image, seed=seed, step=10, actions=actions))
        self.assertEqual(d.stage_choice(points, 'c', 10, 1., quality=True), 1.)
        self.assertEqual(d.stage_choice(points, 'c', 10, .5, quality=True), .5)
        changed = copy.deepcopy(points)
        for p in changed:
            if p['image_id'] == 'c':
                for a in p['actions']:
                    a['stochastic']['psnr'] = -999*a['gamma']
        self.assertEqual(d.stage_choice(changed, 'c', 10, 1., quality=True), 1.)

    def test_changed_source_bytes_fail_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'; path.write_text('{}\n')
            path.with_suffix('.sha256').write_text(d.sha(path)+'\n')
            self.assertEqual(d.read_signed(path), {})
            path.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                d.read_signed(path)


if __name__ == '__main__':
    unittest.main()
