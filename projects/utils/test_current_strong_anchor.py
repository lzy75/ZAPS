"""A66 metadata/cost provenance tests, no model calls."""
from copy import deepcopy
from pathlib import Path
import unittest
from utils import current_strong_anchor as rules
from utils import historical_ffhq_gain as old


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        root=Path(__file__).resolve().parents[1]/'reports/20261028_a65_received/A65'
        plan=rules.read(root/'plan.json');summary=rules.read(root/'summary.json');null=rules.read(root/'null_diagnostic/run.json')
        runs=[rules.read(root/'cases'/c['key']/'HISTORICAL_NATIVE/repeat_0/run.json') for c in plan['cases']]
        return plan,summary,runs,null

    def test_current_anchor_accepted_without_relabelling_failed_historical_gate(self):
        p,s,rows,n=self.fixture();before=deepcopy(s)
        receipt=rules.source_contract(p,s,rows,n)
        self.assertFalse(receipt['original_historical_gate_passed'])
        self.assertFalse(receipt['source_records_relabelled'])
        self.assertFalse(receipt['algorithm_or_tolerances_changed'])
        self.assertEqual(s,before)
        self.assertAlmostEqual(receipt['current_anchor_quality']['psnr'],28.226237487033025)

    def test_budget_counts_inherited_and_new_once(self):
        b=rules.budget()
        self.assertEqual(b['new_method_forwards'],33900)
        self.assertEqual(b['new_method_forwards']+b['inherited_method_forwards'],old.budget()['max_method_forwards'])
        self.assertEqual(b['new_optimizer_backwards']+b['inherited_optimizer_backwards'],1170)
        self.assertEqual(len(rules.queue()),14)
        self.assertNotIn(('HISTORICAL_NATIVE',0),rules.queue())
        self.assertEqual(rules.queue(),old.queue()[1:])

    def test_source_failure_or_duplicate_is_not_accepted(self):
        p,s,rows,n=self.fixture()
        bad=deepcopy(rows);bad[0]['status']='failed'
        with self.assertRaises(ValueError):rules.source_contract(p,s,bad,n)
        with self.assertRaises(ValueError):rules.source_contract(p,s,rows[:-1],n)
        bad=deepcopy(rows);bad[-1]=bad[0]
        with self.assertRaises(ValueError):rules.source_contract(p,s,bad,n)
        with self.assertRaises(ValueError):rules.source_contract(p,s,rows,dict(n,passed=False))

    def test_reference_algorithm_settings_are_identical(self):
        p,_,_,_=self.fixture()
        self.assertEqual(p['config'],old.config())
        self.assertEqual(p['grid'],old.grid())
        self.assertEqual(p['arms'],list(old.ARMS))
        self.assertEqual([c['seed'] for c in p['cases']],list(range(1000,1010)))


if __name__=='__main__':unittest.main()
