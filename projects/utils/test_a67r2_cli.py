"""Exercise the REAL child entry with the exact queue flags, without a model."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class ChildCLITests(unittest.TestCase):
    def test_queue_worker_arguments_feedback_and_time(self):
        entry=Path(__file__).with_name('a67r2_session.py')
        for shard,phase in ((0,'feedback'),(1,'time')):
            result=subprocess.run([sys.executable,str(entry),'--worker','--root','/not-created/a67',
                '--shard',str(shard),'--phase',phase,'--device-uuid','GPU-test-uuid','--check-worker-cli'],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            row=json.loads(result.stdout)
            self.assertEqual(row['device_uuid'],'GPU-test-uuid')
            self.assertEqual((row['shard'],row['phase'],row['model_calls']),(shard,phase,0))


if __name__=='__main__':unittest.main()
