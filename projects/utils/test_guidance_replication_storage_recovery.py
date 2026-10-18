"""Storage relocation, component-only reuse and cost accounting; no GPU calls."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from utils import migrate_guidance_replication_storage as move
from utils import recover_guidance_replication_storage as recovery


def source_fixture():
    cases=[dict(key=f'{i}_s{s}') for i in ('00020','00021','00022') for s in (1000,2000)]
    plan=dict(cases=cases,config={'lr':.01},grid=list(range(30)))
    source=dict(case=cases[3],plan_sha256='plan',status='running',method_nfe=300,diagnostic_nfe=120,
        config=plan['config'],grid=plan['grid'],gates={k:dict(passed=True) for k in ('training','observer','archive','cached_geometry')},
        calls=[dict(phase='method',t=t) for t in plan['grid'][::-1]*10]+[dict(phase='probe',t=1)]*120,
        rows=[{}]*30,probes=[{}]*5,loss_history=[1.]*10)
    return plan,source


def case_fixture(key):
    probes=[dict(step=k,t=k,pairs=[dict(s_dir=1.,s_dir_per_state_rms=1.)]*2) for k in range(5)]
    rows=[dict(guidance_variation3_per_u=1.,guidance_rms=1.) for k in range(5)]
    return dict(case=dict(key=key),status='complete',method_nfe=300,diagnostic_nfe=120,
                guidance_valid=30,usable_snapshots=5,probes=probes,rows=rows)


class MigrationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'home/A2a';source.mkdir(parents=True)
        (source/'plan.json').write_text(json.dumps(dict(experiment='guidance_replication_plan',status='prepared')))
        child=source/'00021_s2000';child.mkdir();(child/'trajectory.pt').write_bytes(b'partial tensor')
        (child/'run.json.tmp').write_bytes(b'')
        return source,root/'data/A2a'

    def test_migration_preserves_partial_evidence_and_original_path(self):
        with tempfile.TemporaryDirectory() as folder:
            source,dest=self.fixture(Path(folder));before=move.inventory(source)
            move.migrate(source,dest,require_other_filesystem=False,reserve_bytes=0)
            self.assertTrue(source.is_symlink());self.assertEqual(source.resolve(),dest.resolve())
            self.assertEqual(move.inventory(source),before);self.assertEqual(move.inventory(dest),before)
            self.assertFalse(list(source.parent.glob('*.verified_root_copy_*')))
            # A second call recognizes the existing alias and never deletes source data.
            move.migrate(source,dest,require_other_filesystem=False,reserve_bytes=0)

    def test_copy_failure_keeps_original_directory_and_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            source,dest=self.fixture(Path(folder));before=move.inventory(source)
            with patch.object(move.shutil,'copytree',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):move.migrate(source,dest,require_other_filesystem=False,reserve_bytes=0)
            self.assertFalse(source.is_symlink());self.assertEqual(move.inventory(source),before)

    def test_source_changed_during_copy_is_not_deleted_or_switched(self):
        with tempfile.TemporaryDirectory() as folder:
            source,dest=self.fixture(Path(folder));original=move.shutil.copytree
            def changed(src,dst,*args,**kwargs):
                value=original(src,dst,*args,**kwargs)
                if Path(src)==source:(source/'new_result.txt').write_text('new work')
                return value
            with patch.object(move.shutil,'copytree',side_effect=changed):
                with self.assertRaisesRegex(RuntimeError,'original changed'):
                    move.migrate(source,dest,require_other_filesystem=False,reserve_bytes=0)
            self.assertFalse(source.is_symlink());self.assertTrue((source/'new_result.txt').exists())

    def test_existing_destination_and_nested_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,dest=self.fixture(Path(folder));dest.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'already exists'):move.migrate(source,dest,False,0)
            (source/'link').symlink_to(source/'plan.json')
            with self.assertRaisesRegex(ValueError,'nested symlink'):move.inventory(source)

    def test_alias_failure_rolls_back_original_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            source,dest=self.fixture(Path(folder));before=move.inventory(source)
            with patch.object(Path,'symlink_to',side_effect=OSError('no metadata space')):
                with self.assertRaises(OSError):move.migrate(source,dest,False,0)
            self.assertFalse(source.is_symlink());self.assertEqual(move.inventory(source),before)
            self.assertEqual(move.inventory(dest),before)


class RecoveryContractTests(unittest.TestCase):
    def test_only_pre_output_observation_are_loaded_not_truncated_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan,source=source_fixture();recovery.write_json(root/'plan.json',plan)
            child=root/'00021_s2000';child.mkdir()
            for p in (child/'pre_last_state.pt',child/'last_opt.pt',root/'00021_measurement.pt'):
                with zipfile.ZipFile(p,'w') as z:z.writestr('tensor/data',b'CRC-checked component fixture')
            (child/'trajectory.pt').write_bytes(b'truncated and unusable')
            (child/'run.json.tmp').write_bytes(b'')
            source['plan_sha256']=recovery.digest(root/'plan.json')
            source['measurement']=dict(path=str(root/'00021_measurement.pt'),sha256=recovery.digest(root/'00021_measurement.pt'))
            recovery.write_json(child/'run.json',source)
            _,_,files=recovery.component_files(root,plan)
            self.assertEqual(len(files),4)
            self.assertNotIn(str(child/'trajectory.pt'),files)
            (child/'pre_last_state.pt').write_bytes(b'truncated')
            with self.assertRaises(zipfile.BadZipFile):recovery.component_files(root,plan)

    def test_only_complete_training_component_can_be_reused(self):
        plan,source=source_fixture();recovery.validate_source_metadata(source,plan,'plan')
        changed=copy.deepcopy(source);changed['gates']['training']['passed']=False
        with self.assertRaisesRegex(ValueError,'component gate'):recovery.validate_source_metadata(changed,plan,'plan')
        changed=copy.deepcopy(source);changed['calls'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):recovery.validate_source_metadata(changed,plan,'plan')
        changed=copy.deepcopy(source);changed['plan_sha256']='different'
        with self.assertRaises(ValueError):recovery.validate_source_metadata(changed,plan,'plan')

    def test_full_stage_cost_retains_interrupted_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan,_=source_fixture();active={}
            for case in plan['cases']:
                key=case['key'];directory=root/key;directory.mkdir()
                record=case_fixture(key)
                original=copy.deepcopy(record)
                if key=='00021_s2000':
                    original['status']='running';record['method_nfe']=0
                recovery.write_json(directory/'run.json',original);active[key]=record
            retry=root/'recovery_runs/fourth';retry.mkdir(parents=True)
            recovery.write_json(retry/'run.json',active['00021_s2000'])
            registry=dict(attempts=[dict(directory='recovery_runs/fourth',status='complete')])
            summary=recovery.stage_summary(root,plan,registry,active)
            self.assertEqual(summary['method_nfe'],1800);self.assertEqual(summary['diagnostic_nfe'],840)
            self.assertEqual(summary['completed_cases'],6)
            self.assertEqual(summary['classification'],'PILOT_READABILITY_SUPPORTED')

    def test_failed_recovery_attempt_cost_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan,_=source_fixture()
            retry=root/'recovery_runs/failed';retry.mkdir(parents=True)
            recovery.write_json(retry/'run.json',dict(status='failed',method_nfe=0,diagnostic_nfe=37))
            summary=recovery.stage_summary(root,plan,dict(attempts=[dict(directory='recovery_runs/failed',status='failed')]),{})
            self.assertEqual(summary['diagnostic_nfe'],37);self.assertEqual(summary['completed_cases'],0)

    def test_recovery_does_not_call_optimizer_or_new_measurement_rng(self):
        tree=ast.parse(Path(recovery.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recover_case')
        calls={n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertNotIn('optimize',calls);self.assertNotIn('randn',calls)

    def test_source_scalar_comparison_uses_original_neighbors_and_tolerances(self):
        rows=[]
        for k in range(30):
            rows.append(dict(step=k,t=999-k,t_prev=998-k,parameter_index=29-k,noise_u=k/30,
                residual_norm=1.,raw_x0_clip_fraction=.1,cosine_clean=None if k<2 else .3,
                cosine_latent=None if k<1 else .3,guidance_cosine=None if k<1 else .3,
                physical_cosine=None if k<1 else .3,guidance_rms=.1))
        recovery.compare_source_rows(rows,copy.deepcopy(rows))
        bad=copy.deepcopy(rows);bad[5]['t']=123
        with self.assertRaisesRegex(ValueError,'grid/index'):recovery.compare_source_rows(rows,bad)
        bad=copy.deepcopy(rows);bad[5]['residual_norm']=2.
        with self.assertRaisesRegex(ValueError,'residual'):recovery.compare_source_rows(rows,bad)


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU tensor validation awaits server PyTorch')
class TensorValidationTests(unittest.TestCase):
    def test_pre_snapshot_shape_and_finite_validation(self):
        import torch
        plan=dict(grid=list(range(30)))
        state=dict(zeta=torch.ones(30),D=torch.ones(30,3,256,256),init_noise=torch.ones(1,3,256,256),
                   tau=torch.arange(30),cpu_rng=torch.get_rng_state(),cuda_rng=[torch.get_rng_state()])
        output=torch.ones(1,3,256,256);y=torch.ones(1,3,64,64)
        recovery.validate_tensors(state,output,y,plan)
        state['zeta'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'PRE tensor'):recovery.validate_tensors(state,output,y,plan)


if __name__=='__main__':unittest.main()
