"""Pure protocol, old-zero-cost guard and new-plan checks; no model imports."""
import contextlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from utils import diffpir_numeric_transfer as rules
from utils import diag_diffpir_numeric_transfer as driver
from utils import diag_diffpir_multiplier_transfer as failed_driver


class Tests(unittest.TestCase):
    def test_four_arm_budget_and_upstream(self):
        self.assertEqual(rules.verify_upstream()['commit'],'2a9898129a1b274131b98746e5b364bc20adc1e1')
        cases=rules.cases(dict(ffhq='/ffhq',imagenet='/imagenet'))
        self.assertEqual(len(cases)*len(rules.ARMS)*100,4800)
        self.assertTrue(all(c['arms']==list(rules.ARMS) for c in cases))

    def test_prepare_preserves_failed_parent_and_refuses_nonzero_cost(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'A71';source.mkdir();failed=base/'A79';failed.mkdir()
            (source/'plan.json').write_text('{}');(failed/'plan.json').write_text('{}')
            driver.files.write(failed/'summary.json',dict(actual_new_cost=dict(method_forward_attempts=0)))
            np.save(source/'motion_kernel.npy',np.ones((3,3),np.float32)/9)
            folders={v:base/v for v in rules.DATASETS}
            for v,folder in folders.items():
                folder.mkdir()
                for image in rules.IDS[v]:(folder/(image+'.png')).write_bytes(b'fixed image fixture')
            old=dict(cases=[dict(image=str(folders['ffhq']/'00226.png'))],motion_kernel_sha256=driver.files.digest(source/'motion_kernel.npy'))
            args=types.SimpleNamespace(root=str(base/'A79R1'),a71_root=str(source),failed_a79_root=str(failed),ffhq_dir=str(folders['ffhq']),imagenet_dir=str(folders['imagenet']))
            real_digest=driver.files.digest;real_isfile=Path.is_file
            def digest(path):
                for name,h in rules.CHECKPOINTS.values():
                    if str(path).endswith('/modules/models/'+name):return h
                return real_digest(path)
            def isfile(path):return '/modules/models/' in str(path) or real_isfile(path)
            before=(failed/'summary.json').read_bytes()
            with patch.object(driver.parent,'check',return_value=old),patch.object(failed_driver,'check',return_value={}),patch.object(driver.files,'digest',side_effect=digest),patch.object(Path,'is_file',isfile),contextlib.redirect_stdout(io.StringIO()):
                driver.prepare(args)
                p=driver.check(Path(args.root));self.assertEqual(p['budget']['method_forwards'],4800)
                driver.summarize(args.root)
                summary=driver.files.read(Path(args.root)/'summary.json')
                self.assertEqual((summary['completed_runs'],summary['planned_runs']),(0,48))
                self.assertEqual((failed/'summary.json').read_bytes(),before)
                driver.files.write(failed/'summary.json',dict(actual_new_cost=dict(method_forward_attempts=1)))
                args.root=str(base/'blocked')
                with self.assertRaisesRegex(ValueError,'nonzero model costs'):driver.prepare(args)


if __name__=='__main__':unittest.main()
