"""A65 CPU protocol tests, not model results."""
from copy import deepcopy
import unittest
from utils import historical_ffhq_gain as rules


class ContractTests(unittest.TestCase):
    def test_original_rows_seeds_and_learning_not_a64_units(self):
        cases=rules.cases('/ffhq')
        self.assertEqual([c['seed'] for c in cases],list(range(1000,1010)))
        self.assertEqual([c['image_id'] for c in cases],[f'{i:05d}' for i in range(10)])
        self.assertEqual(rules.config()['lr'],.01)
        self.assertEqual(rules.config()['schedule'],[15,10,5])
        self.assertAlmostEqual(sum(c['reference']['psnr'] for c in cases)/10,28.19048207925228)

    def test_full_budget_queue_and_no_hidden_prefix(self):
        self.assertEqual(rules.budget()['max_method_forwards'],36900)
        self.assertEqual(rules.budget()['max_optimizer_backwards'],1170)
        self.assertEqual(rules.budget()['max_method_input_vjp'],1800)
        q=rules.queue();self.assertEqual(len(q),15);self.assertEqual(len(set(q)),15)
        self.assertEqual(q[0],('HISTORICAL_NATIVE',0))
        self.assertLess(q.index(('LEGACY_IEEE',0)),q.index(('RESPONSE_300',0)))

    def test_gate_catches_wrong_observation_even_if_reconstruction_matches(self):
        rows=[]
        for case in rules.cases('/ffhq'):
            ref=case['reference']
            rows.append(dict(case=case,arm='HISTORICAL_NATIVE',repeat=0,metrics=ref,
                observation=dict(observed_psnr=ref['observed_psnr']),loss_history=[ref['final_mse']]))
        self.assertTrue(rules.replay_gate(rows)['passed'])
        bad=deepcopy(rows);bad[0]['observation']['observed_psnr']+=.01
        self.assertFalse(rules.replay_gate(bad)['passed'])
        with self.assertRaises(ValueError):rules.replay_gate(rows[:-1])

    def test_gate_catches_quality_and_training_drift_without_selecting_images(self):
        rows=[]
        for case in rules.cases('/ffhq'):
            ref=case['reference']
            rows.append(dict(case=case,arm='HISTORICAL_NATIVE',repeat=0,metrics=dict(ref),
                observation=dict(observed_psnr=ref['observed_psnr']),loss_history=[ref['final_mse']]))
        rows[0]['metrics']['psnr']-=.2
        self.assertFalse(rules.replay_gate(rows)['passed'])
        rows[0]['metrics']['psnr']+=.2;rows[0]['loss_history'][-1]*=2
        self.assertFalse(rules.replay_gate(rows)['passed'])

    def test_report_never_fabricates_missing_quality_or_three_repeat_time(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from utils.diag_historical_ffhq_gain import render_table
        groups=[dict(arm=a,name=rules.NAMES[a],images=0,complete_runs=0,forward_nfe=rules.NFE[a],
                     optimizer_backwards=rules.UPDATES[a]) for a in rules.ARMS]
        summary=dict(groups=groups,primary=dict(classification='PRIMARY_COMPARISON_NOT_COMPLETE'),limits='test')
        with TemporaryDirectory() as folder:
            render_table(Path(folder),summary)
            report=(Path(folder)/'comparison.md').read_text()
            self.assertIn('未运行',report);self.assertIn('未完成三次',report)
            self.assertIn('28.1905',report);self.assertIn('26.6300',report)
            self.assertNotIn('**28.1905**',report)

    def test_zero_model_preflight_retry_never_retries_paid_or_warmup_attempt(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from utils.a65_session import no_model_attempts
        with TemporaryDirectory() as folder:
            root=Path(folder);self.assertTrue(no_model_attempts(root))
            path=root/'workers/shard0/load_000.json'
            rules.write(path,dict(warmup=dict(forward_attempts=0)))
            self.assertTrue(no_model_attempts(root))
            rules.write(path,dict(warmup=dict(forward_attempts=1)))
            self.assertFalse(no_model_attempts(root))


if __name__=='__main__':unittest.main()
