import unittest
from pathlib import Path

from utils import diag_imagenet_closed_blur as transfer
from utils.paper_clock_alignment import read


class DesignTests(unittest.TestCase):
    def test_gaussian_config_matches_recorded_local_reference(self):
        reference = read(transfer.MANIFEST)['archived_reference']
        config = transfer.config('gaussian_deblur')
        for name in ('num_steps', 'num_epochs', 'schedule', 'lr', 'd_init', 'eta',
                     'surrogate_score_jacobian', 'use_learned_var'):
            self.assertEqual(config[name], reference[name])
        self.assertEqual(config['zeta_init'], reference['gaussian_deblur']['zeta_init'])
        self.assertEqual(config['wave'], reference['wavelet'])
        self.assertEqual(config['level'], reference['wavelet_level'])
        self.assertEqual(transfer.protocol.grid(), reference['actual_timesteps_ascending'])

    def test_panel_and_shared_prefix_budget(self):
        cases = transfer.cases(Path('/imagenet'))
        self.assertEqual(len({c['image_id'] for c in cases}), 10)
        self.assertEqual(len({c['measurement_seed'] for c in cases}), 20)
        self.assertEqual(len(cases) * (270+30*len(transfer.ARMS)), 7200)
        self.assertTrue(all(c['seed'] == 1000+int(c['image_id']) and c['sigma'] == .05 for c in cases))
