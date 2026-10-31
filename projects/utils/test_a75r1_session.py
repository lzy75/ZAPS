"""A75R1 regression: environment before imports, safe recovery, no result deletion."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from utils import a75r1_session as repair


class RepairTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder)/'experiment';state=Path(folder)/'experiment_session';state.mkdir()
        old=dict(status='failed',error="CalledProcessError: utils.test_dps_multiplier_transfer")
        (state/'session.json').write_text(json.dumps(old))
        (state/'coordinator.log').write_text('MKL_THREADING_LAYER=INTEL is incompatible with libgomp')
        return root,state,old
    def test_gnu_environment_set_before_original_imports(self):
        self.assertEqual(os.environ['MKL_THREADING_LAYER'],'GNU')
        self.assertNotIn('MKL_SERVICE_FORCE_INTEL',os.environ)
    def test_zero_model_failure_is_preserved_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as folder:
            root,state,old=self.fixture(folder);backup=repair.preserve_mkl_failure(root,state)
            self.assertEqual(json.loads((backup/'session.json').read_text()),old)
            self.assertIn('libgomp',(backup/'coordinator.log').read_text())
            self.assertFalse(root.exists());self.assertIsNone(repair.preserve_mkl_failure(root,state))
    def test_existing_model_or_plan_files_forbid_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root,state,old=self.fixture(folder);root.mkdir();(root/'plan.json').write_text('{}')
            with self.assertRaises(ValueError):repair.preserve_mkl_failure(root,state)
            self.assertEqual(json.loads((state/'session.json').read_text()),old)
            self.assertTrue((root/'plan.json').exists())
    def test_different_failure_is_not_reopened(self):
        with tempfile.TemporaryDirectory() as folder:
            root,state,old=self.fixture(folder);(state/'coordinator.log').write_text('some other error')
            with self.assertRaises(ValueError):repair.preserve_mkl_failure(root,state)
            self.assertEqual(json.loads((state/'session.json').read_text()),old)


if __name__=='__main__':unittest.main()
