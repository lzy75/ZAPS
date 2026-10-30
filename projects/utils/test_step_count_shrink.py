"""A74 fixed-grid, explicit reuse and real CLI protocol checks."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from utils import step_count_shrink as rules
from utils import shrink_multitask as parent
from utils.paper_clock_alignment import grid as original_grid


class DesignTests(unittest.TestCase):
    def test_grid_counts_endpoints_and_exact_original30(self):
        self.assertEqual(rules.grid(30),original_grid())
        for s in rules.COUNTS:
            g=rules.grid(s)
            self.assertEqual(len(g),s);self.assertEqual(len(set(g)),s)
            self.assertEqual(g,sorted(g));self.assertEqual((g[0],g[-1]),(0,999))
            self.assertEqual(sum(rules.COUNTS[s]),s)
    def test_budget_and_shared_observation_parent_keys(self):
        source=parent.cases('/not-loaded');cases=rules.cases(source)
        self.assertEqual(len(cases),36)
        self.assertEqual(len({c['parent_key'] for c in cases}),12)
        fresh=[c for c in cases if c['steps']!=30]
        self.assertEqual(len(fresh)*len(rules.ARMS),48)
        self.assertEqual(sum(10*c['steps']*len(rules.ARMS) for c in fresh),6000)
        self.assertEqual(len(cases)-len(fresh),12)
    def test_actual_worker_cli_has_no_model_work(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'utils/diag_step_count_shrink.py','--worker',
            '--shard','1','--device-uuid','GPU-test','--check-worker-cli'],cwd=root,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),dict(shard=1,device_uuid='GPU-test',model_calls=0))


if __name__=='__main__':unittest.main()
