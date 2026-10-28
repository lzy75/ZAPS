"""A64 pure contract checks runnable on Mac; no model quality claims."""
from pathlib import Path
import unittest
from utils import paper_clock_alignment as rules
from utils import paper_clock_runtime as runtime


class ContractTests(unittest.TestCase):
    def test_actual_grid_and_public_settings_not_old_labels(self):
        expected=[0,24,48,71,95,119,143,166,190,214,238,262,285,309,333,
                  334,371,408,445,482,518,555,592,629,666,667,750,833,916,999]
        self.assertEqual(rules.grid(),expected)
        self.assertEqual([sum(lo<=t<=hi for t in rules.grid()) for lo,hi in ((0,333),(334,666),(667,999))],
                         [15,10,5])
        for task in rules.TASKS:
            cfg=rules.config(task)
            self.assertEqual((cfg['lr'],cfg['d_init'],cfg['num_epochs']),(.001,.2,10))
            self.assertEqual(cfg['zeta_init'],.2 if 'deblur' in task else .1)

    def test_fixed_cohort_hashes_match_existing_identity_not_recursive_selection(self):
        source=rules.read(Path(__file__).resolve().parents[1]/'reports/20261027_a56_ffhq_received/A56_plan.json')
        cohort=rules.cases(dict(ffhq='/ffhq',imagenet='/imagenet'))
        self.assertEqual(len(cohort),16)
        self.assertEqual(len({c['key'] for c in cohort}),16)
        for c in cohort:
            originals=[v for v in source['cases'] if v['dataset']==c['dataset'] and v['image_id']==c['image_id']]
            self.assertTrue(originals)
            self.assertTrue(all(v['image_sha256']==c['image_sha256'] for v in originals))
            self.assertIn('mask_seed',c)  # Regression for A56/A57 missing-key failure.

    def test_budget_separates_forwards_backwards_and_repeat_queue(self):
        self.assertEqual(rules.budget()['max_method_forwards'],30240)
        self.assertEqual(rules.budget()['max_input_vjp'],2880)
        queue=rules.queue()
        self.assertEqual(queue[0],('PAPER_ZAPS',0))
        self.assertEqual(len(queue),9)
        self.assertEqual(len(set(queue)),9)
        from utils.no_learning_replay import burn_count
        self.assertEqual(burn_count(),261)

    def test_gate_requires_complete_baseline_and_cannot_certify_reproduction(self):
        def rows():
            cases=[c for c in rules.cases(dict(ffhq='/f',imagenet='/i')) if c['dataset']=='ffhq']
            return [dict(case=c,arm='PAPER_ZAPS',repeat=0,metrics=dict(rules.PAPER['ffhq'][c['task']])) for c in cases]
        records=rows();gate=rules.baseline_gate('ffhq',records)
        self.assertTrue(gate['proceed'])
        self.assertIn('not same-sample',gate['limits'])
        records[0]['metrics']['psnr']-=7
        self.assertFalse(rules.baseline_gate('ffhq',records)['proceed'])
        with self.assertRaises(ValueError):rules.baseline_gate('ffhq',records[:-1])
        self.assertNotIn('inpainting',rules.PAPER['imagenet'])

    def test_timing_covers_input_and_output_sync_in_order(self):
        events=[];times=iter((10.,14.))
        def sync():events.append('sync')
        def operation():events.extend(('load_observation','make_operator','all_learning','CPU_output'));return 'cpu'
        result,seconds=runtime.clocks(sync,operation,clock=lambda:next(times))
        self.assertEqual((result,seconds),('cpu',4.))
        self.assertEqual(events,['sync','load_observation','make_operator','all_learning','CPU_output','sync'])

    def test_diagnostic_nonfinite_and_numpy_scalar_are_json_safe(self):
        from utils.diag_paper_clock_alignment import finite
        class BooleanScalar:
            def item(self): return True
        self.assertEqual(finite(dict(value=BooleanScalar(),bad=float('nan'))),dict(value=True,bad=None))

    def test_allocations_never_escape_parent_and_one_gpu_is_serial(self):
        rows=[dict(index='0',uuid='GPU-a',free_bytes=10*rules.GIB),
              dict(index='1',uuid='GPU-b',free_bytes=8*rules.GIB)]
        self.assertEqual([r['index'] for r in rules.available_gpus(rows,'0,1','1')],['1'])
        with self.assertRaises(ValueError):rules.available_gpus(rows,'0','1')
        from tempfile import TemporaryDirectory
        from utils.a64_session import choose_assignment
        with TemporaryDirectory() as folder:
            assigned=choose_assignment(Path(folder),rows[:1],rules.DATASETS)
            self.assertEqual({v['uuid'] for v in assigned.values()},{'GPU-a'})
            self.assertEqual(set(assigned),set(rules.DATASETS))


if __name__=='__main__':unittest.main()
