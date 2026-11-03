"""Pure upstream identity, native grid, paper parameter, budget, CLI and source-freeze checks."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from utils import diffpir_multiplier_transfer as rules
from utils import diag_diffpir_multiplier_transfer as driver


class Tests(unittest.TestCase):
    def test_pinned_upstream_and_native_design(self):
        meta=rules.verify_upstream();self.assertEqual(meta['commit'],'2a9898129a1b274131b98746e5b364bc20adc1e1')
        seq=rules.sequence();self.assertEqual((len(seq),seq[0],seq[-1],len(set(seq))),(100,0,999,100))
        c=rules.cases(dict(ffhq='/ffhq',imagenet='/imagenet'))
        self.assertEqual(len(c)*2*100,2400);self.assertEqual({v['sigma_model'] for v in c},{.1})
        self.assertEqual({v['sigma_physical'] for v in c},{.05})
        for v in c:
            self.assertEqual((v['lambda_'],v['zeta']),rules.PAPER_PARAMETERS[v['dataset']][v['task']])
            self.assertEqual(rules.configuration(v)['guidance_scale'],1.)
        self.assertEqual({v['dataset'] for v in c if v['shard']==0},{'ffhq'})
        self.assertEqual({v['dataset'] for v in c if v['shard']==1},{'imagenet'})

    def test_prepare_does_not_reuse_old_y_or_mutate_parent_and_summary_zero_cost(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'A71';source.mkdir();root=base/'A79'
            (source/'plan.json').write_text('{}');np.save(source/'motion_kernel.npy',np.ones((3,3),np.float32)/9)
            folders={v:base/v for v in rules.DATASETS}
            for v,folder in folders.items():
                folder.mkdir()
                for image in rules.IDS[v]:(folder/(image+'.png')).write_bytes(b'fixed image fixture')
            old=dict(cases=[dict(image=str(folders['ffhq']/'00226.png'))],motion_kernel_sha256=driver.files.digest(source/'motion_kernel.npy'))
            args=types.SimpleNamespace(root=str(root),a71_root=str(source),ffhq_dir=str(folders['ffhq']),imagenet_dir=str(folders['imagenet']))
            def digest(path):
                for name,h in rules.CHECKPOINTS.values():
                    if str(path).endswith('/modules/models/'+name):return h
                return real_digest(path)
            real_digest=driver.files.digest;real_isfile=Path.is_file
            def isfile(path):return '/modules/models/' in str(path) or real_isfile(path)
            before=(source/'plan.json').read_bytes()
            with patch.object(driver.parent,'check',return_value=old),patch.object(driver.files,'digest',side_effect=digest),patch.object(Path,'is_file',isfile),contextlib.redirect_stdout(io.StringIO()):driver.prepare(args)
            p=driver.check(root);self.assertEqual(p['budget']['method_forwards'],2400)
            self.assertEqual(p['budget']['input_vjp'],0);self.assertNotIn('observations',p)
            self.assertEqual((source/'plan.json').read_bytes(),before)
            with contextlib.redirect_stdout(io.StringIO()):driver.summarize(root)
            s=driver.files.read(root/'summary.json');self.assertEqual((s['completed_runs'],s['planned_runs']),(0,24));self.assertEqual(s['actual_new_cost']['method_forward_attempts'],0)

    def test_worker_cli_requires_pinned_device(self):
        cmd=[sys.executable,'utils/diag_diffpir_multiplier_transfer.py','--worker','--check-worker-cli']
        bad=subprocess.run(cmd,cwd=driver.PROJECTS,text=True,capture_output=True);self.assertNotEqual(bad.returncode,0)
        good=subprocess.run(cmd+['--shard','0','--device-uuid','GPU-test'],cwd=driver.PROJECTS,text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(good.stdout)['model_calls'],0)


if __name__=='__main__':unittest.main()
