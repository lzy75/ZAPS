"""Counter contract and cost-preserving recovery, with no model dependency."""
import json
from pathlib import Path
import tempfile
import unittest
from utils import paper_clock_alignment as files
from utils import transfer_diagonal_runtime as control
from utils.transfer_counter_recovery import preserve_failure
from utils.diag_complete_diagonal_transfer import summarize


class Tests(unittest.TestCase):
    def fixture(self,root):
        files.write(root/'plan.json',dict(host='pigdm',dataset='ffhq',nfe=100,sources=[],files={'old':'hash'}))
        files.write(root/'session.json',dict(status='failed',error="KeyError('corrections')"))
        files.write(root/'load.json',dict(probe_counts=dict(forwards=1,input_vjp_attempts=1)))
        files.write(root/'summary.json',dict(completed_runs=1))
        (root/'worker.log').write_text("KeyError: 'corrections'\n")
        case=dict(key='sr_1',task='super_resolution')
        files.write(root/'cases/sr_1/BASE/run.json',dict(status='complete',case=case,arm='BASE',
                    reused=True,counts={},metrics=dict(psnr=20,ssim=.5,lpips=.2),full_seconds=1))
        files.write(root/'cases/sr_1/SHRINK/run.json',dict(status='failed',case=case,arm='SHRINK',
                    error="KeyError('corrections')",traceback='pigdm_noisy_r1_runtime.py',artifacts={},
                    counts=dict(forward_attempts=1,forwards=1,input_vjp_attempts=1,optimizer_backward_attempts=0)))

    def test_counter_contract_preserves_existing_forward_accounting(self):
        count=dict(forwards=12,input_vjp_attempts=12)
        control.initialize_counts(count)
        count['corrections']+=1
        self.assertEqual(count['corrections'],1)
        self.assertEqual(count['forwards'],12)

    def test_failed_attempt_preserved_completed_baseline_retained_and_cost_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            old=(root/'cases/sr_1/SHRINK/run.json').read_bytes()
            base=(root/'cases/sr_1/BASE/run.json').read_bytes()
            backup=preserve_failure(root)
            self.assertEqual((backup/'cases/sr_1/SHRINK/run.json').read_bytes(),old)
            self.assertEqual((root/'cases/sr_1/BASE/run.json').read_bytes(),base)
            self.assertFalse((root/'cases/sr_1/SHRINK').exists())
            files.write(root/'plan.json',files.read(backup/'plan.json'))
            result=summarize(root)
            self.assertEqual(result['actual_new_cost']['forwards'],1)
            self.assertEqual(result['actual_new_cost']['input_vjp_attempts'],1)
            self.assertEqual(result['preserved_model_gate']['probe_counts']['forwards'],1)

    def test_other_error_is_retained_without_reopening(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            files.write(root/'session.json',dict(status='failed',error='Another failure'))
            with self.assertRaises(ValueError):preserve_failure(root)
            self.assertTrue((root/'cases/sr_1/SHRINK/run.json').exists())

    def test_later_failure_is_retained_without_reopening(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            path=root/'cases/sr_1/SHRINK/run.json'
            row=files.read(path);row['counts']['forwards']=2;files.write(path,row)
            with self.assertRaises(ValueError):preserve_failure(root)
            self.assertTrue(path.exists())


if __name__=='__main__':
    unittest.main()
