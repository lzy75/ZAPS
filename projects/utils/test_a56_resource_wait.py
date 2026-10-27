"""Resource/zero-call recovery tests; no Torch/model invocation."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import unittest

from utils import a56_resource_wait_session as r


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(index='0', uuid='GPU-a', free=11000), dict(index='1', uuid='GPU-b', free=13000)]

    def test_one_card_is_serial_two_cards_parallel(self):
        self.assertEqual(r.select_devices(self.rows), ['1', '1'])
        self.rows[0]['free'] = 15000
        self.assertEqual(r.select_devices(self.rows), ['0', '1'])

    def test_insufficient_waits_without_lowering_threshold(self):
        self.rows[1]['free'] = 12287
        self.assertIsNone(r.select_devices(self.rows))
        self.rows[1]['free'] = 12288
        self.assertEqual(r.select_devices(self.rows), ['1', '1'])

    def test_visible_uuid_aliases_never_escape_allocation(self):
        self.assertIsNone(r.select_devices(self.rows, '0'))
        self.assertEqual(r.select_devices(self.rows, 'GPU-b,1'), ['GPU-b', 'GPU-b'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            r.select_devices(self.rows, 'GPU-b', first='0')
        for visible in ('', '-1', 'unknown'):
            with self.assertRaises(ValueError):
                r.select_devices(self.rows, visible)

    def test_explicit_low_card_waits_instead_of_switching(self):
        self.assertIsNone(r.select_devices(self.rows, first='0'))
        self.assertEqual(r.select_devices(self.rows, first='1'), ['1', '1'])

    def fixture(self, folder, stack):
        prior = Path(folder)/'prior'; prior.mkdir()
        root = prior/'frozen'; root.mkdir()
        r.write(root/'plan.json', dict(implementation_revision='A56_SELECTION_R2', cases=[{}]*20))
        r.write(prior/'session.json', dict(status='failed'))
        (prior/'session.log').write_text(r.RESOURCE_ERROR+'\n')
        for name, value in dict(PRIOR=prior, ROOT=root, EVIDENCE=root/'resource_resume_20261027').items():
            stack.enter_context(patch.object(r, name, value))
        stack.enter_context(patch.object(r.frozen, 'proof'))
        stack.enter_context(patch.object(r.frozen.original, 'check', side_effect=lambda p:r.read(p/'plan.json')))
        return prior, root

    def test_frozen_plan_and_failure_bytes_unchanged_idempotent(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            prior, root = self.fixture(folder, stack)
            before = {str(p):r.digest(p) for p in (root/'plan.json', prior/'session.json', prior/'session.log')}
            first = r.admit_frozen()
            self.assertEqual(first, r.admit_frozen())
            self.assertEqual(before, {p:r.digest(p) for p in before})
            self.assertTrue(all(r.digest(p) == sha for p,sha in first['snapshots'].items()))
            (root/'plan.json').write_text('{}')
            with self.assertRaises(ValueError):
                r.admit_frozen()

    def test_any_started_attempt_or_unknown_failure_refuses_retry(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            prior, root = self.fixture(folder, stack)
            attempt = root/'case'/'BASELINE'/'attempt_000'; attempt.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'attempt'):
                r.admit_frozen()
            attempt.rmdir()
            (prior/'session.log').write_text('CUDA out of memory during model forward')
            with self.assertRaisesRegex(ValueError, 'resource failure'):
                r.admit_frozen()

    def coordinator_fixture(self, folder, stack):
        prior, root = self.fixture(folder, stack)
        old = Path(folder)/'original'; old.mkdir()
        stack.enter_context(patch.object(r, 'BASE', prior/'resource_wait'))
        stack.enter_context(patch.object(r.frozen, 'OLD_ROOT', old/'A56'))
        stack.enter_context(patch.dict(r.os.environ, {}, clear=True))
        return prior, root

    def test_wait_then_invoke_original_launcher_once_without_reselection(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            prior, root = self.coordinator_fixture(folder, stack)
            plan_sha = r.digest(root/'plan.json')
            stack.enter_context(patch.object(r, 'inventory', side_effect=[
                [dict(index='0', uuid='GPU-a', free=11000)],
                [dict(index='0', uuid='GPU-a', free=13000)]]))
            sleep = stack.enter_context(patch.object(r.time, 'sleep'))
            run = stack.enter_context(patch.object(r.subprocess, 'run', return_value=SimpleNamespace(returncode=0)))
            self.assertEqual(r.coordinator(), 0)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ['bash', str(r.PROJECTS/'run_a56_selection_r2_20261027.sh')])
            self.assertEqual(run.call_args.kwargs['env']['ZAPS_A56_GPU0'], '0')
            self.assertEqual(run.call_args.kwargs['env']['ZAPS_A56_GPU1'], '0')
            sleep.assert_called_once_with(30)
            self.assertEqual(plan_sha, r.digest(root/'plan.json'))
            self.assertEqual(r.read(r.BASE/'session.json')['status'], 'complete')

    def test_unknown_launcher_failure_stops_without_automatic_retry(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            prior, root = self.coordinator_fixture(folder, stack)
            stack.enter_context(patch.object(r, 'inventory', return_value=self.rows))
            run = stack.enter_context(patch.object(r.subprocess, 'run', return_value=SimpleNamespace(returncode=1)))
            self.assertEqual(r.coordinator(), 1)
            run.assert_called_once()
            self.assertEqual(r.read(r.BASE/'session.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
