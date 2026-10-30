"""A73 scope/budget and actual worker CLI checks, no Torch requirement."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from utils import all_unroll_shrink as rule
from utils import shrink_multitask as parent


class DesignTests(unittest.TestCase):
    def test_fixed_scope_and_balanced_shards_without_mutating_parent(self):
        before=parent.cases('/not-loaded');old=[dict(c) for c in before]
        cases=rule.cases(before)
        self.assertEqual(before,old)
        self.assertEqual(len(cases),12)
        self.assertEqual({c['sigma'] for c in cases},{.05})
        self.assertEqual([sum(c['shard']==i for c in cases) for i in (0,1)],[6,6])
        self.assertEqual(len(cases)*len(rule.ARMS)*300,7200)
        self.assertEqual(len(cases)*len(rule.ARMS)*10,240)
    def test_no_new_or_missing_case_substitution(self):
        with self.assertRaises(ValueError):rule.cases(parent.cases('/not-loaded')[:-3])
    def test_actual_worker_cli(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'utils/diag_all_unroll_shrink.py','--worker',
            '--shard','1','--device-uuid','GPU-test','--check-worker-cli'],cwd=root,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),dict(shard=1,device_uuid='GPU-test',model_calls=0))


if __name__=='__main__':unittest.main()
