"""A2a archive must preserve failed evidence, bytes, source, HEAD and index."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils import archive_guidance_replication_stage as archive


def fixture(parent):
    root = parent/'data0/A2a_20261018'
    root.mkdir(parents=True)
    (root/'plan.json').write_text('{}')
    bad = root/'00021_s2000'
    bad.mkdir()
    (bad/'run.json').write_text('{"status":"running"}')
    (bad/'run.json.tmp').write_bytes(b'')
    (bad/'trajectory.pt').write_bytes(b'PK truncated tensor')
    (root/'geometry.csv').write_bytes(b'k,t\r\n0,999\r\n')
    (root.parent/'A2a_repair_20261018.log').write_text('repair complete\n')
    active = {}
    for image in ('00020', '00021', '00022'):
        for seed in (1000, 2000):
            key = f'{image}_s{seed}'
            active[key] = dict(case=dict(image_id=image, sampling_seed=seed), probes=[dict(t=319)],
                               rows=[dict(step=0, t=999, guidance_cosine=None, guidance_rms=1.),
                                     dict(step=1, t=319, guidance_cosine=.9, guidance_rms=.01)])
    summary = dict(completed_cases=6, same_t_descriptive_associations=[dict(t=319, n=6, spearman=.2)],
                   method_nfe=1800, diagnostic_nfe=840)
    (root/'recovery_summary.json').write_text(json.dumps(summary))
    return root, summary, active


class ArchiveTests(unittest.TestCase):
    def test_archive_preserves_partial_evidence_and_hashes_excluded_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root, summary, active = fixture(parent)
            before = archive.inventory(root)
            alias = parent/'cohort_alias'
            alias.symlink_to(root, target_is_directory=True)
            bundle, manifest = archive.export_bundle(alias, summary, active, parent/'projects')
            self.assertEqual(archive.inventory(root), before)
            self.assertTrue((bundle/'source/00021_s2000/run.json.tmp').is_file())
            self.assertEqual(json.loads((bundle/'source/00021_s2000/run.json').read_text())['status'], 'running')
            self.assertEqual(manifest['excluded'][0]['sha256'], archive.sha256(root/'00021_s2000/trajectory.pt'))
            self.assertFalse(any(bundle.rglob('*.pt')))
            self.assertTrue((bundle/'sidecars/A2a_repair_20261018.log').is_file())
            saved = json.loads((bundle/'derived/recomputed_recovery_summary.json').read_text())
            self.assertEqual(saved['method_nfe']+saved['diagnostic_nfe'], 2640)
            for item in manifest['files']:
                self.assertEqual(archive.sha256(bundle/item['relative_path']), item['sha256'])

    def test_nested_symlink_and_source_change_abort_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root, summary, active = fixture(parent)
            link = root/'outside'
            link.symlink_to(parent/'data0', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Nested symlink'):
                archive.export_bundle(root, summary, active, parent/'projects')
            link.unlink()
            original = shutil.copyfile
            changed = []
            def changing(source, target):
                result = original(source, target)
                if not changed:
                    changed.append(True)
                    (root/'new_during_export.json').write_text('{}')
                return result
            with patch.object(archive.shutil, 'copyfile', side_effect=changing):
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    archive.export_bundle(root, summary, active, parent/'projects')
            self.assertEqual(list((parent/'projects/server_records').iterdir()), [])

    def test_missing_case_or_stale_summary_cannot_pass_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, summary, active = fixture(Path(temporary))
            with patch.object(archive.recovery, 'validate_plan'), patch.object(archive.recovery, 'registry_for', return_value={}), \
                    patch.object(archive.recovery, 'load_active', return_value=active), \
                    patch.object(archive.recovery, 'stage_summary', return_value=summary):
                self.assertEqual(archive.audit_source(root)[0], summary)
                active['00020_s1000'] = None
                with self.assertRaisesRegex(ValueError, 'all six'):
                    archive.audit_source(root)
                active['00020_s1000'] = {}
                (root/'recovery_summary.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Saved recovery_summary'):
                    archive.audit_source(root)

    def test_git_bytes_and_custom_message_preserve_head_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args])
            git('init', '-q')
            git('config', 'user.name', 'Archive Test')
            git('config', 'user.email', 'archive-test@example.invalid')
            (repo/'.gitattributes').write_text('*.csv text eol=lf\n')
            git('add', '.gitattributes')
            git('commit', '-qm', 'base')
            head = git('rev-parse', 'HEAD')
            (repo/'unrelated.txt').write_text('user staged edit')
            git('add', 'unrelated.txt')
            index_hash = archive.sha256(repo/'.git/index')
            root, summary, active = fixture(repo)
            bundle, manifest = archive.export_bundle(root, summary, active, repo/'projects')
            commit = archive.commit_results(bundle, 'results/a2a-test', repo, message='Archive complete A2a\n')
            self.assertEqual(git('rev-parse', 'HEAD'), head)
            self.assertEqual(archive.sha256(repo/'.git/index'), index_hash)
            self.assertEqual(git('show', '-s', '--format=%s', commit).decode().strip(), 'Archive complete A2a')
            self.assertNotIn(b'unrelated.txt', git('ls-tree', '-r', '--name-only', commit).splitlines())
            for item in manifest['files']:
                stored = git('show', commit+':'+str((bundle/item['relative_path']).relative_to(repo.resolve())))
                self.assertEqual(stored, (bundle/item['relative_path']).read_bytes())


if __name__ == '__main__':
    unittest.main()
