"""Meaningful preservation/locking tests, no Torch or GPU calls."""
import json
from pathlib import Path
import tempfile
import unittest
from utils import a54_session as session


class RecoveryTests(unittest.TestCase):
    def make(self, root, status='running', error=None):
        root.mkdir()
        plan = dict(experiment='A54_current_norm_gain', cases=[dict(key='example')])
        session.write(root / 'plan.json', plan)
        folder = root / 'example/control_000'
        folder.mkdir(parents=True)
        session.write(folder / 'run.json', dict(case=plan['cases'][0], plan_sha256=session.digest(root / 'plan.json'),
                      status=status, error=error, nfe=17, backward_attempts=16))
        (folder / 'partial.pt').write_bytes(b'preserve these incomplete bytes')
        return folder

    def test_preserve_interrupted_bytes_and_cost_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            old = self.make(root)
            before = (old / 'run.json').read_bytes()
            plan_before = (root / 'plan.json').read_bytes()
            ledger = session.recover_stale(root)
            saved = root / ledger['entries'][0]['preserved']
            self.assertFalse(old.exists())
            self.assertEqual(before, (saved / 'run.json').read_bytes())
            self.assertEqual(plan_before, (root / 'plan.json').read_bytes())
            self.assertEqual(17, ledger['recorded_interrupted_forwards'])
            self.assertEqual(16, ledger['recorded_interrupted_backwards'])
            self.assertEqual(ledger, session.recover_stale(root))
            self.make_again(root)
            with self.assertRaisesRegex(ValueError, 'already used'):
                session.recover_stale(root)

    @staticmethod
    def make_again(root):
        folder = root / 'example/control_000'
        folder.mkdir()
        session.write(folder / 'run.json', dict(case=dict(key='example'), plan_sha256=session.digest(root / 'plan.json'),
                      status='running', nfe=3, backward_attempts=2))

    def test_complete_and_scientific_failure_untouched(self):
        for status, error in [('complete', None), ('failed', "RuntimeError('Nonfinite trajectory')")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'run'
                folder = self.make(root, status, error)
                before = (folder / 'run.json').read_bytes()
                if status == 'complete':
                    self.assertEqual([], session.recover_stale(root)['entries'])
                else:
                    with self.assertRaisesRegex(ValueError, 'not auto-retried'):
                        session.recover_stale(root)
                self.assertEqual(before, (folder / 'run.json').read_bytes())

    def test_journaled_rename_recovers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            old = self.make(root)
            saved = root / 'interrupted_attempts/example/control_saved'
            entry = dict(case='example', original=str(old.relative_to(root)), preserved=str(saved.relative_to(root)),
                         run_sha256=session.digest(old / 'run.json'), state='journaled',
                         recorded_attempted_forwards=17, recorded_backward_attempts=16)
            session.write(root / 'interruption_recovery.json', dict(plan_sha256=session.digest(root / 'plan.json'), entries=[entry]))
            session.recover_stale(root)
            self.assertTrue((saved / 'run.json').is_file())
            self.assertFalse(old.exists())

    def test_real_lock_detects_active_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lock'
            self.assertFalse(session.busy(path))
            with session.exclusive(path):
                self.assertTrue(session.busy(path))
                with self.assertRaises(BlockingIOError):
                    with session.exclusive(path):
                        pass
            self.assertFalse(session.busy(path))


if __name__ == '__main__':
    unittest.main()
