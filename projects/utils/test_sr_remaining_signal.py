import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils import diag_sr_remaining_signal as driver
from utils.diag_sr_remaining_signal import corrected_update, remaining_signal
from utils.noise_signal_shrink import select
from utils.paper_clock_alignment import write


class RuleTests(unittest.TestCase):
    def test_original_ray_is_kept_when_remaining_energy_is_below_noise(self):
        scalar = select(2., 5., 1., 4, 1.)
        self.assertGreater(scalar['weight'], 0.)
        result = remaining_signal(scalar, 4, 1.)
        self.assertEqual(result['residual_energy'], 1.)
        self.assertEqual(result['noise_energy'], 3.)
        self.assertEqual(result['beta'], 0.)

    def test_surplus_is_rescaled_relative_to_original_target(self):
        scalar = select(2., 10., 1., 4, 1.)
        result = remaining_signal(scalar, 4, 1.)
        self.assertAlmostEqual(result['beta'] * scalar['weight'], .5)
        self.assertEqual(remaining_signal(select(2., 20., 1., 4, 1.), 4, 1.)['beta'], 1.)


class ResultDirectoryTests(unittest.TestCase):
    def test_completed_directory_is_reused_without_a_gpu_run(self):
        with TemporaryDirectory() as tmp:
            root, parent, audit = (Path(tmp)/name for name in ('A113', 'A103', 'A112'))
            report = dict(experiment='A113', status='complete', completed_cases=20, planned_cases=20)
            write(root/'summary.json', report)
            write(root/'plan.json', dict(experiment='A113', parent=str(parent), audit=str(audit)))
            before = (root/'summary.json').read_bytes()
            with patch.object(driver, 'execute') as execute, patch.object(driver, 'print_summary') as emit:
                driver.run_or_reuse(parent, audit, root)
            execute.assert_not_called()
            emit.assert_called_once_with(report)
            self.assertEqual((root/'summary.json').read_bytes(), before)

    def test_interrupted_case_is_preserved_beside_completed_case(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete, running = ({'key': name} for name in ('complete', 'interrupted'))
            descending = list(range(29, -1, -1))

            def count(repetitions, backwards):
                return dict(forwards=30*repetitions, forward_attempts=30*repetitions,
                            calls=descending*repetitions, optimizer_backward_attempts=backwards,
                            optimizer_backward_completed=backwards)

            counts = dict(prefix=count(9, 9), **{arm:count(1, 1) for arm in driver.ARMS})
            saved = root/'cases'/complete['key']
            write(saved/'run.json', dict(status='complete', case=complete, counts=counts,
                                        arms={arm:{} for arm in driver.ARMS}, updates=[{}]*30))
            for arm in driver.ARMS:
                (saved/(arm+'.pt')).write_bytes(b'fixture')
            previous = root/'cases'/running['key']/'run.json'
            write(previous, dict(status='running', case=running, counts={}))
            before = previous.read_bytes()
            rows, interrupted = driver.read_completed_cases(root, [{'case': complete}, {'case': running}], descending)
            self.assertEqual([row['case'] for row in rows], [complete])
            driver.preserve_interrupted(root, interrupted)
            self.assertTrue((saved/'run.json').is_file())
            self.assertFalse(previous.exists())
            archived, = (root/'interrupted').glob('*/interrupted/run.json')
            self.assertEqual(archived.read_bytes(), before)


class TorchTests(unittest.TestCase):
    def test_zero_remaining_signal_returns_exact_scalar_with_graph(self):
        import torch

        scalar = select(2., 5., 1., 4, 1.)

        def original(zaps, operator, clean, q, residual, alpha, sigma, counts):
            return q + 100, {'scalar': scalar}, counts

        q = torch.ones(4, requires_grad=True)
        chosen, detail, _ = corrected_update(original)(None, None, q, q, q, None, 1., {})
        torch.testing.assert_close(chosen, scalar['gamma']*q, rtol=0, atol=0)
        chosen.sum().backward()
        torch.testing.assert_close(q.grad, torch.full_like(q, scalar['gamma']))
        self.assertEqual(detail['remaining_signal']['applied_increment_norm'], 0.)
