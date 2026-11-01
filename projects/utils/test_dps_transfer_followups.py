"""Pure protocol and real worker CLI checks."""
import copy,json,subprocess,sys,unittest,tempfile,types
from unittest.mock import patch
from pathlib import Path
from utils import dps_feedback_ablation as a
from utils import dps_imagenet_transfer as b
class Tests(unittest.TestCase):
 def test_profiles_exclude_recipient_and_preserve_time(self):
  cases=[dict(key=i,image_id=i,task='super_resolution',sigma=.05) for i in ('a','b','c')]
  traces={i:[dict(t=999-k,gamma=v+k/1000) for k in range(1000)] for i,v in zip(('a','b','c'),(1.,3.,5.))}
  out=a.profiles(cases,traces)
  self.assertEqual(out['a']['FIXED_LOO']['donors'],['b','c']);self.assertEqual(out['a']['FIXED_LOO']['gamma'][400],4.4)
  self.assertEqual(out['c']['MISMATCH']['donors'],['a'])
  changed=copy.deepcopy(traces)
  for r in changed['a']:r['gamma']=100.
  self.assertEqual(out['a'],a.profiles(cases,changed)['a'])
  changed['b'].pop()
  with self.assertRaises(ValueError):a.profiles(cases,changed)
 def test_new_model_budget_and_resource_gate(self):
  cases=b.cases('/existing/imagenet');self.assertEqual(len(cases)*len(b.ARMS)*1000,12000)
  self.assertEqual({c['image_id'] for c in cases},set(b.IDS));self.assertEqual({c['dataset'] for c in cases},{'imagenet'})
  self.assertFalse(b.admission(dict(free_bytes=9*1024**3))['admitted'])
  self.assertTrue(b.admission(dict(free_bytes=8*1024**3),2*1024**3)['admitted'])
  self.assertFalse(b.admission(dict(free_bytes=10*1024**3),0,10*1024**3)['admitted'])
 def test_prepare_and_zero_new_call_summary_reuse_only_completed_parent(self):
  from utils import diag_dps_feedback_ablation as driver
  from utils import diag_dps_multiplier_transfer as previous
  files=driver.files
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);source=base/'A75';root=base/'A76';cases=[];records={}
   files.write(source/'plan.json',{'fixture':True});files.write(source/'summary.json',dict(completed_runs=18,planned_runs=18,actual_new_cost={'method_forward_attempts':18000}))
   for task in a.TASKS:
    for i in ('a','b','c'):
     c=dict(key=task+i,image_id=i,task=task,sigma=.05,shard=len(cases)%2);cases.append(c)
     for arm in ('DPS','SHRINK'):
      folder=source/'cases'/c['key']/arm
      r=dict(status='complete',case=c,arm=arm,metrics=dict(psnr=25.,ssim=.8,lpips=.1),full_seconds=120.)
      records[c['key'],arm]=r;files.write(folder/'run.json',r)
      files.write(folder/'trace.json',[dict(t=999-k,gamma=1.) for k in range(1000)])
   old=dict(cases=cases,inputs={},observations={},checkpoint={},parent=str(base/'A71'),sources={},dps_root='/existing/DPS',diffusion_config={},task_configs={})
   with patch.object(previous,'check',return_value=old),patch.object(previous,'completed',side_effect=lambda _,c,arm:records[c['key'],arm]):
    driver.prepare(types.SimpleNamespace(root=str(root),a75_root=str(source)))
   plan=driver.check(root);self.assertEqual(len(plan['references']),12)
   self.assertEqual(plan['operator_root'],str(base/'A71'))
   driver.summarize(root);result=files.read(root/'summary.json')
   self.assertEqual((result['completed_runs'],result['planned_runs'],result['new_completed_runs']),(12,24,0))
   self.assertEqual(result['actual_new_cost']['method_forward_attempts'],0)
   self.assertEqual(result['source_A75_cost']['method_forward_attempts'],18000)
 def test_worker_cli_requires_explicit_device(self):
  for name in ('dps_feedback_ablation','dps_imagenet_transfer'):
   command=[sys.executable,'utils/diag_'+name+'.py','--worker','--check-worker-cli']
   project=Path(__file__).resolve().parents[1]
   bad=subprocess.run(command,cwd=project,text=True,capture_output=True);self.assertNotEqual(bad.returncode,0)
   good=subprocess.run(command+['--shard','1','--device-uuid','GPU-test'],cwd=project,text=True,capture_output=True,check=True)
   self.assertEqual(json.loads(good.stdout)['model_calls'],0)
if __name__=='__main__':unittest.main()
