import unittest
from utils import equal_nfe_feedback as rules


class Tests(unittest.TestCase):
    def test_frozen_panel_and_budget_shape(self):
        rows = rules.cases('/tmp/ffhq')
        self.assertEqual(len(rows), 10 * 4 * 3)
        self.assertEqual(len({r['key'] for r in rows}), len(rows))
        self.assertEqual(rules.IMAGES, tuple(f'{i:05d}' for i in range(10)))
        self.assertEqual(rules.ARMS,
                         ('ZAPS_LOCAL_300', 'CLOSED_SCALAR_300', 'CLOSED_DIAG_300'))
        for row in rows:
            image_index = int(row['image_id'])
            self.assertEqual(row['seed'], 1000 + image_index)
            self.assertEqual(row['measurement_seed'], row['seed'])
            self.assertEqual(row['mask_seed'], 2001000 + image_index)

    def test_bootstrap_is_deterministic(self):
        a = rules.bootstrap_mean_ci([1., 2., 3.], 7, draws=1000)
        b = rules.bootstrap_mean_ci([1., 2., 3.], 7, draws=1000)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], 2.)
        self.assertGreaterEqual(a[1], 2.)


if __name__ == '__main__':
    unittest.main()
