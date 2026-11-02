"""A78 pure source-reuse, profile, cache-audit, and allocated-GPU contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from utils import dps_feedback_ablation as profiles
from utils import dps_cached_feedback_diagnosis as cached
from utils import diag_dps_imagenet_feedback_ablation as driver
from utils import diag_dps_imagenet_transfer as previous
from utils import a78_session as session


class Tests(unittest.TestCase):
    def fixture(self,base):
        source=base/'A77';cases=[];records={};observations={};traces={};files=driver.files
        files.write(source/'plan.json',dict(fixture=True))
        files.write(source/'summary.json',dict(completed_runs=12,planned_runs=12,actual_new_cost=dict(method_forward_attempts=12000)))
        for task in profiles.TASKS:
            for i in ('00067','00151','00193'):
                c=dict(key=task+i,image_id=i,task=task,dataset='imagenet',sigma=.05,shard=len(cases)%2);cases.append(c)
                obs=source/'observations'/c['key']/'y.pt';obs.parent.mkdir(parents=True);obs.write_bytes(b'immutable y fixture')
                files.write(obs.parent/'observation.json',dict(path=str(obs),sha256=files.digest(obs)));observations[c['key']]=str(obs)
                for arm in ('DPS','SHRINK'):
                    m=dict(psnr=25.,ssim=.7,lpips=.2,measurement_mse=.0025)
                    if arm=='SHRINK':m.update(psnr=24.,ssim=.8,lpips=.1)
                    r=dict(status='complete',case=c,arm=arm,metrics=m,full_seconds=600.)
                    records[c['key'],arm]=r;folder=source/'cases'/c['key']/arm;files.write(folder/'run.json',r)
                    trace=[dict(t=999-k,gamma=1.+int(i)/1000,weight=.5,residual_energy=4.,noise_energy=2.,response_energy=1.) for k in range(1000)]
                    files.write(folder/'trace.json',trace)
                    if arm=='SHRINK':traces[c['key']]=trace
        old=dict(cases=cases,inputs={},observations=observations,checkpoint={},operator_root=str(base/'A71'),parent=str(base/'A75'),sources={},dps_root='/existing/DPS',diffusion_config={},task_configs={})
        return source,old,records,traces

    def test_prepare_reuses_y_and_physical_root_and_counts_only_new_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source,old,records,traces=self.fixture(base);root=base/'A78'
            with patch.object(previous,'check',return_value=old),patch.object(previous,'completed',side_effect=lambda _,c,a:records[c['key'],a]),contextlib.redirect_stdout(io.StringIO()):
                driver.prepare(types.SimpleNamespace(root=str(root),a77_root=str(source)))
            p=driver.check(root);self.assertEqual(p['operator_root'],old['operator_root']);self.assertEqual(p['observations'],old['observations'])
            self.assertEqual(len(p['references']),12)
            with contextlib.redirect_stdout(io.StringIO()):driver.summarize(root)
            s=driver.files.read(root/'summary.json');self.assertEqual((s['completed_runs'],s['planned_runs'],s['new_completed_runs']),(12,24,0))
            self.assertEqual(s['actual_new_cost']['method_forward_attempts'],0)
            self.assertEqual(s['source_A77_cost']['method_forward_attempts'],12000)
            self.assertEqual(s['cached_diagnosis']['method_nfe'],0)
            for c in p['cases']:
                row=driver.files.read(c['profiles']['FIXED_LOO']['path'])
                self.assertNotIn(c['key'],row['donors'])
            for c in p['cases']:
                for arm in profiles.RUN_ARMS:
                    r=copy.deepcopy(records[c['key'],'SHRINK']);r['arm']=arm;r['metrics']['psnr']-=.2
                    r['counts']=dict(forward_attempts=1000,input_vjp_attempts=1000,optimizer_backward_attempts=0)
                    driver.files.write(root/'cases'/c['key']/arm/'run.json',r)
            with contextlib.redirect_stdout(io.StringIO()):driver.summarize(root)
            s=driver.files.read(root/'summary.json')
            self.assertEqual((s['completed_runs'],s['new_completed_runs']),(24,12))
            self.assertEqual(s['actual_new_cost']['method_forward_attempts'],12000)
            self.assertEqual(len(s['matching_tests']),2)
            self.assertTrue(all(t['development_matching_candidate'] for t in s['matching_tests']))

    def test_quality_labels_cannot_affect_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,old,records,traces=self.fixture(Path(tmp));a=profiles.profiles(old['cases'],traces)
            diagnosis=cached.diagnose(old['cases'],list(records.values()),traces)
            self.assertTrue(all(r['psnr_down_ssim_up_lpips_down'] for r in diagnosis['psnr_negative_images']))
            self.assertEqual(diagnosis['by_case'][0]['final_measurement_mse_noise_ratio']['SHRINK'],.0025/.05**2)
            for r in records.values():r['metrics']['psnr']=100.
            self.assertEqual(a,profiles.profiles(old['cases'],traces))

    def test_gpu_filter_and_worker_cli(self):
        rows=[dict(index=0,uuid='GPU-a',free_bytes=7*1024**3),dict(index=1,uuid='GPU-b',free_bytes=11*1024**3)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(session,'gpu_inventory',return_value=rows):
            chosen=session.allocation(Path(tmp),'0,1',None)
            self.assertEqual({v['uuid'] for v in chosen.values()},{'GPU-b'})
            with self.assertRaises(ValueError):session.allocation(Path(tmp),'0',None)
        command=[sys.executable,'utils/diag_dps_imagenet_feedback_ablation.py','--worker','--check-worker-cli']
        bad=subprocess.run(command,cwd=driver.PROJECTS,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0)
        good=subprocess.run(command+['--shard','0','--device-uuid','GPU-test'],cwd=driver.PROJECTS,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(good.stdout)['model_calls'],0)


if __name__=='__main__':unittest.main()
