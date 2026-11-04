"""A88 protocol and real CLI tests; no neural model calls."""
import copy,json,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from utils import dps_missing_tasks as rules

class Tests(unittest.TestCase):
 def test_only_missing_cells_and_official_task_scales(self):
  import yaml
  base=Path(__file__).resolve().parents[2]/'DPS/configs'
  config=yaml.safe_load((base/'diffusion_config.yaml').read_text())
  tasks={t:yaml.load((base/(t+'_config.yaml')).read_text(),Loader=yaml.FullLoader) for t in rules.TASKS}
  rules.validate(config,tasks)
  self.assertEqual(rules.SCALES,{'gaussian_deblur':.3,'inpainting':.5})
  for dataset in rules.IDS:
   cases=rules.cases(dataset,'/existing/'+dataset)
   self.assertEqual(len(cases)*len(rules.ARMS)*rules.STEPS,12000)
   self.assertEqual({c['task'] for c in cases},set(rules.TASKS))
   self.assertEqual({c['image_id'] for c in cases},set(rules.IDS[dataset]))
   self.assertEqual([sum(c['shard']==s for c in cases) for s in (0,1)],[3,3])
   self.assertEqual(len({c['key'] for c in cases}),6)
  wrong=copy.deepcopy(tasks);wrong['inpainting']['conditioning']['params']['scale']=.3
  with self.assertRaises(ValueError):rules.validate(config,wrong)
 def test_dataset_specific_resource_gate_and_peak_margin(self):
  gib=1024**3
  self.assertTrue(rules.admission('ffhq',dict(free_bytes=6*gib))['admitted'])
  self.assertFalse(rules.admission('imagenet',dict(free_bytes=9*gib))['admitted'])
  self.assertTrue(rules.admission('imagenet',dict(free_bytes=8*gib),2*gib)['admitted'])
  self.assertFalse(rules.admission('imagenet',dict(free_bytes=10*gib),0,10*gib)['admitted'])
 def test_prepare_freezes_real_entry_and_zero_model_partial_summary(self):
  from utils import diag_dps_missing_tasks as driver
  from utils import diag_dps_multiplier_transfer as previous
  files=driver.files
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);source=base/'A75';physical=base/'A71';root=base/'A88'
   files.write(source/'plan.json',{'fixture':True});files.write(source/'summary.json',dict(completed_runs=18,planned_runs=18))
   files.write(physical/'plan.json',{});(physical/'motion_kernel.npy').write_bytes(b'fixture only')
   directory=base/'images';directory.mkdir()
   for image in rules.IDS['ffhq']:(directory/(image+'.png')).write_bytes(b'fixture input')
   checkpoint=base/'checkpoint.pt';checkpoint.write_bytes(b'fixture only')
   old=dict(parent=str(physical),dps_root=str(driver.PROJECTS.parent/'DPS'),sources={})
   before={str(f):files.digest(f) for f in (source/'plan.json',source/'summary.json',physical/'plan.json')}
   with patch.object(previous,'check',return_value=old),patch.dict(rules.CHECKPOINTS,{'ffhq':(str(checkpoint),files.digest(checkpoint))}):
    driver.prepare(types.SimpleNamespace(root=str(root),a75_root=str(source),dataset='ffhq',ffhq_dir=str(directory)))
   frozen=driver.check(root);self.assertEqual(frozen['budget']['method_forwards'],12000)
   self.assertEqual(frozen['task_configs']['inpainting']['conditioning']['params']['scale'],.5)
   driver.summarize(root);summary=files.read(root/'summary.json')
   self.assertEqual((summary['completed_runs'],summary['planned_runs']),(0,12))
   self.assertEqual(summary['actual_new_cost']['method_forward_attempts'],0)
   self.assertEqual(summary['groups'],[])
   for f,h in before.items():self.assertEqual(files.digest(f),h)
 def test_worker_cli_requires_explicit_allocation(self):
  project=Path(__file__).resolve().parents[1]
  command=[sys.executable,'utils/diag_dps_missing_tasks.py','--worker','--check-worker-cli']
  bad=subprocess.run(command,cwd=project,text=True,capture_output=True)
  self.assertNotEqual(bad.returncode,0)
  good=subprocess.run(command+['--shard','1','--device-uuid','GPU-test'],cwd=project,text=True,capture_output=True,check=True)
  self.assertEqual(json.loads(good.stdout)['model_calls'],0)
  status=subprocess.run([sys.executable,'utils/a88_session.py','--help'],cwd=project,text=True,capture_output=True)
  self.assertEqual(status.returncode,0,status.stderr)

if __name__=='__main__':unittest.main()
