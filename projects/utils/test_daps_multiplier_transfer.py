"""Pure pinned-reference, config, budget, parent/source and CLI checks."""
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
from utils import daps_multiplier_transfer as rules
from utils import diag_daps_multiplier_transfer as driver


class Tests(unittest.TestCase):
    def test_official_reference_configs_and_counts(self):
        self.assertEqual(rules.verify_upstream()['commit'],rules.COMMIT)
        self.assertEqual(len(rules.verify_upstream()['files']),23)
        for task in rules.TASKS:
            cfg=rules.configuration(task);s=cfg['sampler'];m=s['mcmc_sampler_config']
            self.assertEqual(s['annealing_scheduler_config']['num_steps']*s['diffusion_scheduler_config']['num_steps'],1000)
            self.assertEqual(s['annealing_scheduler_config']['num_steps']*m['num_steps'],20000)
            self.assertEqual((m['tau'],m['prior_solver'],m['mc_algo']),(.01,'gaussian','langevin'))
            self.assertEqual(cfg['operator']['sigma'],.05)
            self.assertEqual(m['lr'],1e-4 if task=='super_resolution' else 5e-5)

    def test_prepare_parent_immutable_and_no_old_measurement_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'A71';source.mkdir();root=base/'A81'
            (source/'plan.json').write_text('{}');driver.files.write(source/'summary.json',dict(completed_runs=72,planned_runs=72))
            cases=[]
            for image in rules.IDS:
                p=base/(image+'.png');p.write_bytes(b'fixed image fixture')
                cases.append(dict(image_id=image,image=str(p)))
            old=dict(cases=cases,sources={})
            real_digest=driver.files.digest;real_isfile=Path.is_file
            def digest(path):return rules.CHECKPOINT[1] if str(path).endswith('/modules/models/'+rules.CHECKPOINT[0]) else real_digest(path)
            def isfile(path):return '/modules/models/' in str(path) or real_isfile(path)
            args=types.SimpleNamespace(root=str(root),a71_root=str(source))
            with patch.object(driver.parent,'check',return_value=old),patch.object(driver.files,'digest',side_effect=digest),patch.object(Path,'is_file',isfile),contextlib.redirect_stdout(io.StringIO()):
                driver.prepare(args);p=driver.check(root)
                self.assertEqual(p['budget']['method_forwards'],12000)
                self.assertEqual(p['budget']['operator_gradient_attempts'],240000)
                self.assertEqual(len(p['cases']),6)
                self.assertNotIn('observations',p)
                self.assertEqual({c['task'] for c in p['cases'] if c['shard']==0},{'super_resolution'})
                driver.summarize(root);s=driver.files.read(root/'summary.json')
                self.assertEqual((s['completed_runs'],s['planned_runs']),(0,12))
                self.assertEqual(s['actual_new_cost']['method_forward_attempts'],0)
                self.assertEqual((source/'plan.json').read_text(),'{}')
                driver.files.write(source/'summary.json',dict(completed_runs=71,planned_runs=72))
                args.root=str(base/'blocked')
                with self.assertRaisesRegex(ValueError,'complete A71'):driver.prepare(args)

    def test_worker_cli_requires_allocated_device(self):
        cmd=[sys.executable,'utils/diag_daps_multiplier_transfer.py','--worker','--check-worker-cli']
        self.assertNotEqual(subprocess.run(cmd,cwd=driver.PROJECTS,capture_output=True).returncode,0)
        row=subprocess.run(cmd+['--shard','0','--device-uuid','GPU-fixture'],cwd=driver.PROJECTS,text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(row.stdout)['model_calls'],0)


if __name__=='__main__':unittest.main()
