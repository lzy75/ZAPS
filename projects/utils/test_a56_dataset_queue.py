"""Partition/resource/handoff regression tests; no Torch/model execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from utils import a56_dataset_queue_session as q


class QueueTests(unittest.TestCase):
    def plan(self):
        cases=[]
        for d,n in (('ffhq',6),('imagenet',4)):
            for t in ('super_resolution','inpainting'):
                for i in range(n):cases.append(dict(key=f'{d}_{t}_{i}',dataset=d,task=t))
        return dict(cases=cases)

    def rows(self):
        return [dict(index='0',uuid='GPU-a',free=7000),dict(index='1',uuid='GPU-b',free=13000)]

    def test_four_groups_disjoint_exact_budget_and_original_shard_assignment(self):
        p=self.plan();jobs=q.groups(p)
        self.assertEqual([len(g['case_keys']) for g in jobs],[6,6,4,4])
        self.assertEqual(len({k for g in jobs for k in g['case_keys']}),20)
        for g in jobs:
            self.assertEqual(g['case_keys'],[c['key'] for i,c in enumerate(p['cases']) if i%2==g['shard'] and c['dataset']==g['dataset']])
        metrology={next(c['key'] for c in p['cases'] if c['dataset']==d and c['task']=='super_resolution') for d in ('ffhq','imagenet')}
        grid=list(range(30));grid[-1]=999
        cost=sum(300+len(q.frozen.original.rules.expected_calls(grid,c['key'] in metrology)) for c in p['cases'])
        self.assertEqual(cost,9624)
        with self.assertRaises(ValueError):q.groups(dict(cases=p['cases'][:-1]))

    def test_large_and_small_gpu_run_different_datasets_at_once(self):
        pairs=q.assignments(q.groups(self.plan()),q.allocation(self.rows()))
        self.assertEqual(len(pairs),2)
        self.assertEqual({g['dataset']:r['index'] for g,r in pairs},{'imagenet':'1','ffhq':'0'})
        self.assertEqual(len({r['uuid'] for g,r in pairs}),2)

    def test_one_small_gpu_runs_ffhq_only_and_never_double_books(self):
        rows=q.allocation(self.rows(),'0');jobs=q.groups(self.plan())
        pairs=q.assignments(jobs,rows)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0][0]['dataset'],'ffhq')
        self.assertEqual(q.assignments(jobs,rows,occupied=['GPU-a']),[])
        self.assertEqual(q.assignments([g for g in jobs if g['dataset']=='imagenet'],rows),[])
        self.assertEqual(q.assignments(jobs,rows,slots=0),[])

    def test_allocation_and_explicit_aliases_are_hard_boundaries(self):
        rows=q.allocation(self.rows(),'GPU-a,0,GPU-b',first='0',second='GPU-a')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['uuid'],'GPU-a')
        for visible in ('','-1','unknown'):
            with self.assertRaises(ValueError):q.allocation(self.rows(),visible)
        with self.assertRaises(ValueError):q.allocation(self.rows(),'0',second='1')

    def test_waiter_identity_forbids_running_or_unrelated_process_signals(self):
        command=['python',str(q.PROJECTS/'utils/a56_resource_wait_session.py'),'--coordinator']
        state=dict(status='waiting_for_gpu',pid=100)
        self.assertTrue(q.waiter_identity(state,command,1000,1000,100))
        for state2,command2,owner,pgid in (
            (dict(status='running_frozen_launcher',pid=100),command,1000,100),
            (state,['python','other.py','--coordinator'],1000,100),
            (state,command,1001,100),(state,command,1000,200)):
            self.assertFalse(q.waiter_identity(state2,command2,owner,1000,pgid))

    def test_started_attempt_blocks_handoff_without_sending_signal(self):
        with TemporaryDirectory() as folder, patch.object(q,'ROOT',Path(folder)):
            p=Path(folder)/'case'/'BASELINE'/'attempt_000';p.mkdir(parents=True)
            with patch.object(q.os,'killpg') as kill:
                with self.assertRaises(ValueError):q.require_no_attempts()
                kill.assert_not_called()

    def test_preload_resource_race_returns_wait_without_torch_or_attempts(self):
        job=q.groups(self.plan())[0]
        with patch.object(q,'manifest',return_value=dict(groups=[job])), \
             patch.object(q.old,'inventory',return_value=[dict(index='0',uuid='GPU-a',free=6000)]), \
             patch.dict(q.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}), \
             patch.object(q,'require_no_attempts') as attempts:
            self.assertEqual(q.worker(job['key']),75)
            attempts.assert_called_once_with(job['case_keys'])

    def test_execution_tags_restore_historical_writers(self):
        d=q.frozen.original;written=[]
        def writer(path,value):written.append((Path(path).name,value))
        with patch.object(d,'signed',side_effect=writer) as original, \
             patch.object(d.a40.source,'signed',side_effect=writer) as baseline, \
             patch.object(q,'manifest',return_value=dict(plan_sha256='frozen')), \
             patch.object(q,'digest',return_value='manifest-sha'):
            with q.execution():
                d.signed(Path('run.json'),dict(status='complete'))
                d.a40.source.signed(Path('run.json'),dict(status='complete'))
            self.assertIs(d.signed,original);self.assertIs(d.a40.source.signed,baseline)
        self.assertEqual(len(written),2)
        self.assertTrue(all(row['queue_execution']['plan_sha256']=='frozen' for name,row in written))


if __name__=='__main__':unittest.main()
