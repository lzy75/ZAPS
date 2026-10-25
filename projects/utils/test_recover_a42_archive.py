import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.recover_a42_archive import commit_bytes,disk_full,external_objects,git


class ArchiveRecoveryTests(unittest.TestCase):
    def repository(self,root):
        repo=root/'repo';repo.mkdir();git(repo,['init','--initial-branch=main'])
        git(repo,['config','user.name','Archive Test']);git(repo,['config','user.email','archive-test@example.invalid'])
        (repo/'tracked.txt').write_text('base\n');(repo/'.gitattributes').write_text('*.png filter=broken\n')
        git(repo,['add','tracked.txt','.gitattributes'])
        git(repo,['-c','core.hooksPath=/dev/null','commit','-m','base'])
        (repo/'tracked.txt').write_text('staged user work\n');git(repo,['add','tracked.txt'])
        (repo/'tracked.txt').write_text('unstaged user work\n')
        bundle=repo/'projects/server_records/example';bundle.mkdir(parents=True)
        payload=b'\x89PNG\r\n\x1a\narchive byte test\x00\xff'
        (bundle/'panel.png').write_bytes(payload);(bundle/'.gitattributes').write_text('* -text\n')
        manifest=dict(files=[dict(path='panel.png',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())])
        (bundle/'manifest.json').write_text(json.dumps(manifest)+'\n')
        return repo,bundle

    def test_missing_clean_filter_bypassed_bytes_index_and_checkout_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo,bundle=self.repository(root)
            head=git(repo,['rev-parse','HEAD']).stdout;index=(repo/'.git/index').read_bytes()
            env=dict(os.environ,GIT_INDEX_FILE=str(root/'legacy_index'))
            git(repo,['read-tree','HEAD'],env)
            failure=git(repo,['-c','filter.broken.clean=false','-c','filter.broken.required=true',
                              'add','-f','--',str(bundle.relative_to(repo))],env,allow_failure=True)
            self.assertNotEqual(failure.returncode,0)
            commit=commit_bytes(bundle,'results/a42-recovery-test',repo,root/'data_work')
            blob=git(repo,['show',commit+':projects/server_records/example/panel.png']).stdout
            self.assertEqual(blob,(bundle/'panel.png').read_bytes())
            self.assertEqual((repo/'.git/index').read_bytes(),index)
            self.assertEqual(git(repo,['rev-parse','HEAD']).stdout,head)
            self.assertEqual((repo/'tracked.txt').read_text(),'unstaged user work\n')
            self.assertEqual(commit_bytes(bundle,'results/a42-recovery-test',repo,root/'data_work'),commit)
            with self.assertRaises(ValueError):commit_bytes(bundle,'main',repo,root/'data_work')

    def test_external_object_store_remains_readable_by_normal_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo,bundle=self.repository(root)
            store,objects=external_objects(repo,root/'data_objects')
            with patch.dict(os.environ,{'GIT_ALTERNATE_OBJECT_DIRECTORIES':str(objects)}):
                commit=commit_bytes(bundle,'results/a42-external-test',repo,root/'data_work',store)
            self.assertEqual(git(repo,['cat-file','-t',commit]).stdout,b'commit\n')
            self.assertEqual(git(repo,['rev-parse','results/a42-external-test']).stdout.decode().strip(),commit)
            self.assertTrue(any(p.is_file() for p in store.rglob('*')))
            self.assertIn(str(store),(objects/'info/alternates').read_text().splitlines())

    def test_disk_full_classification_does_not_guess_other_errors(self):
        self.assertTrue(disk_full(RuntimeError('fatal: No space left on device')))
        self.assertFalse(disk_full(RuntimeError('fatal: Permission denied')))
        self.assertFalse(disk_full(RuntimeError('fatal: index.lock exists')))


if __name__=='__main__':unittest.main()
