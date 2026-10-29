import json
from pathlib import Path
import subprocess
import sys
import unittest
from utils.a71_session import worker_command,PROJECTS


class WorkerCLITests(unittest.TestCase):
    def test_actual_queue_command_parses_without_gpu(self):
        for shard in (0,1):
            command=worker_command(Path('/not-created/a71'),shard,'GPU-test')+['--check-worker-cli']
            r=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(r.stdout),dict(shard=shard,device_uuid='GPU-test',model_calls=0))


if __name__=='__main__':unittest.main()
