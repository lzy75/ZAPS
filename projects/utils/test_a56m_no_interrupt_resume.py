"""No model calls: only the new explicit-allocation/no-interruption contract."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from utils import a56m_no_interrupt_resume as r


class SafetyTests(unittest.TestCase):
    def test_one_gpu_is_the_complete_allocation(self):
        env = r.pinned_environment('GPU-example', {'ZAPS_A56_GPU0':'0', 'ZAPS_A56_GPU1':'1', 'KEEP':'yes'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-example')
        self.assertNotIn('ZAPS_A56_GPU0', env)
        self.assertNotIn('ZAPS_A56_GPU1', env)
        self.assertEqual(env['KEEP'], 'yes')
        with self.assertRaises(ValueError): r.pinned_environment('0,1')

    def test_active_queue_refused_without_signal(self):
        with TemporaryDirectory() as folder, patch.object(r.legacy.run.queue, 'BASE', Path(folder)), \
             patch.object(r, 'busy', return_value=True), patch.object(r.os, 'kill') as kill, \
             patch.object(r.os, 'killpg') as killpg:
            with self.assertRaisesRegex(RuntimeError, 'no signal sent'): r.refuse_active_queue(None)
            kill.assert_not_called(); killpg.assert_not_called()

    def test_inactive_queue_does_not_signal(self):
        with TemporaryDirectory() as folder, patch.object(r.legacy.run.queue, 'BASE', Path(folder)), \
             patch.object(r, 'busy', return_value=False), patch.object(r.os, 'kill') as kill:
            r.refuse_active_queue(None)
            kill.assert_not_called()


if __name__ == '__main__': unittest.main()
