"""Archive byte integrity, source isolation, idempotence and Git/index preservation."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from utils import archive_guidance_feasibility as a


class ArchiveTests(unittest.TestCase):
    def fixture(self, base):
        roots={}
        for label in ('A2a','A2b0','A2c','A2d'):
            root=base/'data'/label;root.mkdir(parents=True)
            (root/'run.json').write_bytes(b'{"note":"CRLF preserved"}\r\n')
            (root/'run.sha256').write_text(a.sha256(root/'run.json')+'\n')
            (root/'branches.pt').write_bytes(b'binary fixture')
            roots[label]=root
        projects=base/'repo/projects';projects.mkdir(parents=True)
        return roots,projects

    def test_text_checksum_and_excluded_inventory_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            roots,projects=self.fixture(Path(folder))
            before={k:a.d.archive.inventory(v) for k,v in roots.items()}
            bundle,manifest=a.export(roots,{'completed_cases':6},projects)
            self.assertEqual(len(manifest['files']),8)
            self.assertEqual(len(manifest['excluded']),4)
            for label in roots:
                self.assertEqual((bundle/label/'run.json').read_bytes(),(roots[label]/'run.json').read_bytes())
                self.assertFalse((bundle/label/'branches.pt').exists())
            self.assertEqual(before,{k:a.d.archive.inventory(v) for k,v in roots.items()})
            again,_=a.export(roots,{'completed_cases':6},projects)
            self.assertEqual(again,bundle)
            (bundle/'A2d/run.json').write_text('edited')
            with self.assertRaises(ValueError):a.export(roots,{'completed_cases':6},projects)

    def test_nested_link_and_destination_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            roots,projects=self.fixture(Path(folder))
            (roots['A2d']/'link.json').symlink_to(roots['A2c']/'run.json')
            with self.assertRaises(ValueError):a.export(roots,{},projects)
            (roots['A2d']/'link.json').unlink()
            with self.assertRaises(ValueError):a.export(roots,{},roots['A2d'])

    def test_result_commit_preserves_head_index_and_exact_crlf_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            roots,projects=self.fixture(Path(folder));repo=projects.parent.resolve()
            def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
            git('init','-q');git('config','user.name','Test');git('config','user.email','test@example.invalid')
            (repo/'tracked.txt').write_text('initial\n');git('add','tracked.txt');git('commit','-qm','initial')
            (repo/'tracked.txt').write_text('staged\n');git('add','tracked.txt')
            head=git('rev-parse','HEAD');index=git('write-tree')
            bundle,_=a.export(roots,{},projects)
            commit=a.archive_commit(bundle,'results/test',repo)
            self.assertEqual(git('rev-parse','HEAD'),head)
            self.assertEqual(git('write-tree'),index)
            relative=str((bundle/'A2d/run.json').relative_to(repo))
            self.assertEqual(git('show',commit+':'+relative),(roots['A2d']/'run.json').read_bytes())
            self.assertEqual(a.archive_commit(bundle,'results/test',repo),commit)


if __name__=='__main__':unittest.main()
