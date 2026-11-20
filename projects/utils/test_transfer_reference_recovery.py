"""Regress shared DPS import and lossless recovery without loading a model."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import dps_multiplier_transfer_runtime as original
from utils import dps_missing_tasks_runtime as missing
from utils import transfer_diagonal_runtime as control
from utils.transfer_reference_recovery import preserve_failure


class Tests(unittest.TestCase):
    def test_both_adapters_use_one_loader_and_restore_after_scope(self):
        original.reference.cache_clear()
        prior = missing.reference
        reference = SimpleNamespace(__package__='_isolated_dps_test')
        calls = []
        def load(path):
            calls.append(path)
            if len(calls)>1:
                raise RuntimeError('Reference package already loaded')
            return reference, {'root':str(path)}
        try:
            with patch.object(original,'load_dps_reference',load), \
                    patch.object(original.importlib,'import_module',return_value=object()), \
                    control.dps_reference_scope():
                first = original.reference('/tmp/dps-reference-regression')
                second = missing.reference('/tmp/dps-reference-regression')
                self.assertIs(first,second)
                self.assertEqual(len(calls),1)
            self.assertIs(missing.reference,prior)
        finally:
            original.reference.cache_clear()

    def fixture(self, root):
        data = dict(status='failed',error="CalledProcessError: utils.test_transfer_diagonal_runtime")
        (root/'session.json').write_text(json.dumps(data))
        (root/'worker.log').write_text('RuntimeError: Reference package already loaded\n')
        (root/'plan.json').write_text(json.dumps(dict(host='dps',sources=['frozen'],files={'old':'hash'})))
        (root/'summary.json').write_text(json.dumps(dict(completed_runs=0,actual_new_cost={'forwards':0})))

    def test_failed_log_and_plan_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            backup=preserve_failure(root)
            for name,value in before.items():
                self.assertEqual((backup/name).read_bytes(),value)
            self.assertFalse((root/'plan.json').exists())
            self.assertEqual((root/'worker.log').read_bytes(),before['worker.log'])

    def test_model_attempts_block_this_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            (root/'load.json').write_text('{}')
            with self.assertRaises(ValueError):preserve_failure(root)
            self.assertTrue((root/'plan.json').exists())

    def test_other_failure_cannot_be_reopened(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            (root/'worker.log').write_text('Another error\n')
            with self.assertRaises(ValueError):preserve_failure(root)


if __name__=='__main__':
    unittest.main()
