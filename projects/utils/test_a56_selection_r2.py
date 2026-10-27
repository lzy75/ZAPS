"""Regression: inventory metadata must not exhaust genuinely unused experimental images."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils import a56_selection_r2 as r


class SelectionTests(unittest.TestCase):
    def test_inventory_not_usage_but_real_failed_planned_unknown_references_are(self):
        data={'ffhq':Path('/tmp/a56_ffhq').resolve()}
        case=lambda i:dict(image=str(data['ffhq']/f'{i:05d}.png'),image_sha256=f'hash{i}')
        doc=dict(cases=[dict(case(32),status='failed'),dict(case(33),status='planned')],selection_audit={'pool':[case(i) for i in range(32,45)]},unknown_reference=case(34),pools={'ffhq':{'selected':[case(35)]}})
        v=r.collect_usage(doc,data)
        self.assertEqual(v['used_names']['ffhq'],{'00032','00033','00034','00035'})
        self.assertEqual(v['used_hashes'],{f'hash{i}' for i in range(32,36)})
        self.assertEqual(len(v['legacy_names']['ffhq']),13)
        self.assertEqual(v['ignored'][0]['pointer'],'selection_audit/pool')

    def test_whole_candidate_pool_regression_and_frozen_deterministic_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);project=p/'projects';(project/'results/prior').mkdir(parents=True)
            data={d:p/d for d in ('ffhq','imagenet')};all_images=[];cases=[]
            for d,folder in data.items():
                folder.mkdir()
                for i in range(32,44):
                    f=folder/f'{i:05d}.png';f.write_bytes((d+str(i)).encode())
                    c=dict(image=str(f),image_id=f.stem,image_sha256=r.original.source.digest(f));all_images.append(c)
                    if i==32:cases.append(c)
            (project/'results/prior/plan.json').write_text(json.dumps(dict(cases=cases,selection_audit={'pool':all_images})))
            with patch.object(r,'PROJECTS',project):
                a,audit=r.history_selection(data,p/'output');b,_=r.history_selection(data,p/'output')
            self.assertEqual(a,b);self.assertEqual({d:len(v) for d,v in a.items()},{'ffhq':6,'imagenet':4})
            for d in data:
                self.assertEqual(audit['pools'][d]['actual_used_names'],1)
                self.assertEqual(audit['pools'][d]['legacy_recursive_used_names'],12)
                self.assertEqual(audit['pools'][d]['legacy_unused_unique'],0)
                self.assertEqual(audit['pools'][d]['corrected_unused_unique'],11)
                self.assertNotIn('00032',{c['image_id'] for c in a[d]})

    def test_existing_frozen_original_plan_forbids_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=p/'old';old.mkdir();(old/'plan.json').write_text('{}')
            with patch.object(r,'OLD_ROOT',old),patch.object(r,'BASE',p/'new'):
                with self.assertRaisesRegex(ValueError,'already has a frozen plan'):r.preflight_once()
            self.assertEqual((old/'plan.json').read_text(),'{}')

    def test_zero_call_selection_failure_is_preserved_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=p/'old/A56';old.parent.mkdir()
            log=old.parent/'session.log';log.write_text("history_selection\nValueError: Need6 unused unique images; available=0. No substitution after outcomes.\n")
            original_bytes=log.read_bytes()
            with patch.object(r,'OLD_ROOT',old),patch.object(r,'BASE',p/'new'):
                r.preflight_once();first,_=r.proof();payload=first.read_bytes()
                r.preflight_once();self.assertEqual(first.read_bytes(),payload)
            self.assertEqual(log.read_bytes(),original_bytes)
            self.assertFalse((old/'plan.json').exists())

    def test_new_plan_binds_revision_and_archived_failure_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=p/'old/A56';old.parent.mkdir();base=p/'revision';root=base/'A56S2'
            (old.parent/'session.log').write_text("history_selection\nNeed6 unused unique images; available=0\n")
            with patch.object(r,'OLD_ROOT',old),patch.object(r,'BASE',base),patch.object(r,'ROOT',root):
                r.preflight_once();root.mkdir()
                with r.bindings():
                    r.original.signed(root/'plan.json',dict(worker_identity_files={},identity_files={},limits='A standalone no-learning replay is not yet confirmed.'))
                plan=r.original.source.load_signed(root/'plan.json')
                self.assertEqual(plan['implementation_revision'],'A56_SELECTION_R2')
                self.assertTrue(set(r.own_sources()).issubset(plan['worker_identity_files']))
                for f,sha in plan['identity_files'].items():
                    self.assertTrue(Path(f).is_relative_to(root))
                    self.assertEqual(r.original.source.digest(f),sha)
                self.assertTrue(Path(plan['selection_preflight_snapshot']).is_file())

    def test_original_process_bindings_restored(self):
        before=r.original.ROOT,r.original.CARD,r.original.BRANCH,r.original.history_selection,r.original.signed
        with r.bindings():
            self.assertIs(r.original.history_selection,r.history_selection)
            self.assertEqual(r.original.ROOT,str(r.ROOT))
        self.assertEqual(before,(r.original.ROOT,r.original.CARD,r.original.BRANCH,r.original.history_selection,r.original.signed))

if __name__=='__main__':unittest.main()
