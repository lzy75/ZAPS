"""Local stage-contract tests; Torch numerical replay tested separately."""
import subprocess,sys,unittest
from pathlib import Path
from utils.mask_stage_response import ARMS,PROBES,applied_gamma
class Tests(unittest.TestCase):
    def test_only_one_frozen_block_is_intervened(self):
        for arm,block in [('BASE_EARLY',range(10)),('BASE_MIDDLE',range(10,20)),('BASE_LATE',range(20,30))]:
            self.assertEqual([i for i in range(30) if applied_gamma(arm,i,.3)==1],list(block))
        self.assertTrue(all(applied_gamma('BASE_CHECK',i,.3)==1 for i in range(30)))
        self.assertTrue(all(applied_gamma('SHRINK_CHECK',i,.3)==.3 for i in range(30)))
    def test_budget_is_bounded(self):
        self.assertEqual(3*(len(ARMS)*30+len(PROBES)),459)
        self.assertTrue(all(0<=i<29 for i in PROBES))
    def test_real_cli(self):
        f=Path(__file__).with_name('diag_mask_stage_response.py')
        r=subprocess.run([sys.executable,str(f),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--gpus',r.stdout)
if __name__=='__main__':unittest.main()
