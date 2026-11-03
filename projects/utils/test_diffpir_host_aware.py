"""Pure mathematical and frozen-parent checks; no Torch or model imports."""
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
from utils import host_aware_shrink as policy
from utils import noise_signal_shrink as original
from utils import diffpir_multiplier_transfer as old_rules
from utils import diffpir_host_aware_transfer as rules
from utils import diag_diffpir_host_aware as driver


class Tests(unittest.TestCase):
    def test_ray_optimum_host_identity_and_zero_penalty(self):
        # Scalar H=2, rho=3, r=4: q=(H^T H+rho)^-1 H^T r.
        q=8/7;d=2*q;a=4*d;b=d*d;prior=3*q*q
        row=policy.select(a,16,b,1,1,prior)
        self.assertAlmostEqual(row['gamma'],15/16)
        self.assertLess(row['stationarity_relative_error'],1e-15)
        def objective(g):return .5*(g*d-row['weight']*4)**2+.5*prior*g*g
        self.assertGreater(objective(row['gamma']-.01),objective(row['gamma']))
        self.assertGreater(objective(row['gamma']+.01),objective(row['gamma']))
        self.assertEqual(policy.select(a,16,b,1,1,0)['gamma'],original.select(a,16,b,1,1)['gamma'])
        self.assertEqual(policy.select(-a,16,b,1,1,prior)['gamma'],0)
        self.assertEqual(policy.select(0,0,0,1,1,0)['gamma'],0)
        self.assertEqual(policy.select(1,1,1,1,2,0)['gamma'],0)
        with self.assertRaises(ValueError):policy.select(a,16,b,1,1,-1)

    def test_prepare_reuses_parent_y_and_refuses_incomplete_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'A79R1';source.mkdir();(source/'plan.json').write_text('{}')
            driver.files.write(source/'summary.json',dict(completed_runs=48,planned_runs=48,actual_new_cost=dict(method_forward_attempts=4800)))
            cases=old_rules.cases(dict(ffhq=base/'ffhq',imagenet=base/'imagenet'))
            kernels={};checkpoints={}
            for task in rules.TASKS:
                path=source/(task+'.npy');path.write_bytes(b'frozen kernel')
                kernels[task]=dict(path=str(path),sha256=driver.files.digest(path))
            for dataset in rules.DATASETS:
                path=base/(dataset+'.pt');path.write_bytes(b'checkpoint fixture')
                checkpoints[dataset]=dict(path=str(path),sha256=driver.files.digest(path))
            def record(c,arm):
                return dict(status='complete',case=c,arm=arm,metrics=dict(psnr=25.,ssim=.8,lpips=.1),full_seconds=1.)
            for c in cases:
                image=Path(c['image']);image.parent.mkdir(exist_ok=True);image.write_bytes(b'image fixture')
                folder=source/'observations'/c['key'];folder.mkdir(parents=True)
                y=folder/'y.pt';y.write_bytes(b'existing observation')
                driver.files.write(folder/'observation.json',dict(path=str(y),sha256=driver.files.digest(y),plan_sha256=driver.files.digest(source/'plan.json')))
                for arm in (rules.REFERENCE_SOURCE_ARM,rules.CACHED_SOURCE_ARM):
                    f=source/'cases'/c['key']/arm;f.mkdir(parents=True)
                    driver.files.write(f/'run.json',record(c,arm));driver.files.write(f/'trace.json',[])
                driver.files.write(source/'metrology'/(c['key']+'.json'),dict(stable_admission_passed=True))
            old=dict(cases=cases,checkpoints=checkpoints,kernels=kernels,sources={},protocol={},sequence=rules.sequence(),actual_grid=[999-v for v in rules.sequence()])
            args=types.SimpleNamespace(root=str(base/'A80'),a79r1_root=str(source))
            before=(source/'summary.json').read_bytes()
            with patch.object(driver.parent,'check',return_value=old),patch.object(driver.parent,'completed',side_effect=lambda root,c,arm:record(c,arm)),contextlib.redirect_stdout(io.StringIO()):
                driver.prepare(args);p=driver.check(args.root)
                self.assertEqual(p['budget']['method_forwards'],2400)
                self.assertEqual(len(p['cases']),12)
                c=p['cases'][0];obs=driver.observation(Path(args.root),p,c,'unused')
                self.assertEqual(Path(obs['path']).read_bytes(),b'existing observation')
                self.assertNotEqual(obs['path'],obs['source_A79R1_y'])
                driver.summarize(args.root)
                s=driver.files.read(Path(args.root)/'summary.json')
                self.assertEqual((s['completed_runs'],s['planned_runs']),(0,24))
                self.assertEqual(s['actual_new_cost']['method_forward_attempts'],0)
                self.assertEqual((source/'summary.json').read_bytes(),before)
                driver.files.write(source/'summary.json',dict(completed_runs=47,planned_runs=48))
                args.root=str(base/'blocked')
                with self.assertRaisesRegex(ValueError,'completed A79R1'):driver.prepare(args)

    def test_worker_cli_requires_explicit_pinned_device(self):
        cmd=[sys.executable,'utils/diag_diffpir_host_aware.py','--worker','--check-worker-cli']
        bad=subprocess.run(cmd,cwd=driver.PROJECTS,text=True,capture_output=True)
        self.assertNotEqual(bad.returncode,0)
        good=subprocess.run(cmd+['--shard','1','--device-uuid','GPU-fixture'],cwd=driver.PROJECTS,text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(good.stdout)['model_calls'],0)


if __name__=='__main__':unittest.main()
