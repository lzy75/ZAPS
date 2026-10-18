"""Archive provenance and isolated Git-ref safety; no Torch/UNet required."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from utils import archive_state_indicator_stage as archive


def fixture(projects):
    next_path = None
    for number in reversed(range(4)):
        folder = projects/'results'/str(number); folder.mkdir(parents=True)
        record = dict(experiment=archive.KINDS[number], status='complete', gates={'one': dict(passed=True)})
        if number == 0: record.update(diagnostic_nfe=30, method_nfe=0, rows=[{}]*5)
        if next_path:
            record.update(source=str(next_path), source_sha256=archive.sha256(next_path))
        path = folder/'run.json'; path.write_text(json.dumps(record)); next_path = path
    return next_path.parent


class ArchiveTests(unittest.TestCase):
    def test_only_completed_a0b_siblings_of_the_same_trace_are_added(self):
        with tempfile.TemporaryDirectory() as folder:
            projects = Path(folder)/'projects'; probe = fixture(projects)
            trace = projects/'results/2/run.json'
            sibling = projects/'results/diag_cached_direction/one'; sibling.mkdir(parents=True)
            record = dict(experiment='cached_conditional_mean_direction', status='complete',
                          source=str(trace), source_sha256=archive.sha256(trace), gates={'one': dict(passed=True)})
            (sibling/'run.json').write_text(json.dumps(record))
            self.assertEqual(len(archive.source_chain(probe, projects)), 5)
            record['gates']['one']['passed'] = False
            (sibling/'run.json').write_text(json.dumps(record))
            self.assertEqual(len(archive.source_chain(probe, projects)), 4)

    def test_failed_or_changed_source_cannot_be_archived(self):
        with tempfile.TemporaryDirectory() as folder:
            projects = Path(folder)/'projects'; probe = fixture(projects)
            self.assertEqual(len(archive.source_chain(probe, projects)), 4)
            tensor = projects/'results/2/trajectory.pt'; tensor.write_bytes(b'original tensor')
            probe_json = probe/'run.json'; saved = json.loads(probe_json.read_text())
            saved['identity_files'] = {str(tensor): archive.sha256(tensor)}
            probe_json.write_text(json.dumps(saved))
            self.assertEqual(len(archive.source_chain(probe, projects)), 4)
            tensor.write_bytes(b'changed tensor')
            with self.assertRaisesRegex(ValueError, 'tensor fingerprint'): archive.source_chain(probe, projects)
            tensor.write_bytes(b'original tensor')
            path = projects/'results/2/run.json'; record = json.loads(path.read_text())
            record['status'] = 'failed'; path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): archive.source_chain(probe, projects)

    def test_binary_inventory_and_verbatim_copy_preserve_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            projects = Path(folder)/'projects'; probe = fixture(projects)
            tensor = probe/'geometry.pt'; tensor.write_bytes(b'private-tensor')
            csv = probe/'probes.csv'; csv.write_text('t,s_dir\n903,1\n')
            before = {str(p): archive.sha256(p) for p in (tensor, csv, probe/'run.json')}
            bundle, manifest = archive.export_bundle(archive.source_chain(probe, projects), projects)
            self.assertFalse(any(p.suffix == '.pt' for p in bundle.rglob('*')))
            self.assertEqual(manifest['excluded'][0]['sha256'], archive.sha256(tensor))
            self.assertTrue((bundle/'00_frozen_time_guidance_local_probe/probes.csv').is_file())
            for p, h in before.items(): self.assertEqual(archive.sha256(p), h)
            for f in manifest['files']: self.assertEqual(archive.sha256(bundle/f['relative_path']), f['sha256'])

    def test_results_commit_does_not_switch_head_or_include_other_staged_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            def git(*args): return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
            git('init', '-q'); git('config', 'user.name', 'Archive Test'); git('config', 'user.email', 'archive-test@example.invalid')
            (repo/'main.txt').write_text('base'); git('add', 'main.txt'); git('commit', '-qm', 'base')
            head = git('rev-parse', 'HEAD')
            (repo/'main.txt').write_text('unrelated unstaged edit')
            (repo/'staged.txt').write_text('unrelated staged edit'); git('add', 'staged.txt')
            index_sha = archive.sha256(repo/'.git/index')
            bundle = repo/'projects/server_records/one'; bundle.mkdir(parents=True)
            (bundle/'manifest.json').write_text('{}')
            commit = archive.commit_results(bundle, 'results/indicator-test', repo)
            self.assertEqual(git('rev-parse', 'HEAD'), head)
            self.assertEqual(archive.sha256(repo/'.git/index'), index_sha)
            self.assertEqual(git('show', commit+':main.txt'), 'base')
            self.assertEqual(git('show', commit+':projects/server_records/one/manifest.json'), '{}')
            entries = git('ls-tree', '-r', '--name-only', commit).splitlines()
            self.assertNotIn('staged.txt', entries)
            self.assertEqual((repo/'main.txt').read_text(), 'unrelated unstaged edit')
            with self.assertRaises(ValueError): archive.commit_results(bundle, 'master', repo)


if __name__ == '__main__': unittest.main()
