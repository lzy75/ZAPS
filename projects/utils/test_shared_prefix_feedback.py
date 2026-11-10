import unittest
from utils import shared_prefix_feedback as rules


class Tests(unittest.TestCase):
    def test_panel_and_exact_accounting(self):
        rows = rules.cases('/tmp/ffhq')
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({r['key'] for r in rows}), 40)
        self.assertEqual(270 + 3*30, 360)
        for row in rows:
            self.assertEqual(row['measurement_seed'], row['seed'])


if __name__ == '__main__': unittest.main()
