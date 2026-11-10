import unittest
from utils import shared_prefix_feedback_multinoise as rules
class Tests(unittest.TestCase):
 def test_fixed_panel(self):
  rows=rules.cases('/tmp/ffhq');self.assertEqual(len(rows),80);self.assertEqual({r['sigma'] for r in rows},{.025,.10})
  self.assertEqual(len({r['key'] for r in rows}),80)
if __name__=='__main__':unittest.main()
