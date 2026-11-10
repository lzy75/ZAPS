"""A102 shared-prefix, method-equivalent 300-NFE FFHQ comparison."""
import argparse, csv, gc, json, statistics, shutil, subprocess, sys, time, traceback
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import shared_prefix_feedback as rules,shared_prefix_feedback_runtime as runtime
from utils import diag_equal_nfe_feedback as fixtures
from utils import historical_ffhq_gain as files,historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils import diag_noise_feasible_zaps as common
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import write,resource

PARENT=fixtures.PARENT
CARD=PROJECTS/'experiment_cards/A102_shared_prefix_feedback_20261110.md'

def check(root):
 p=files.read(Path(root)/'plan.json');files.require(p['experiment']==rules.EXPERIMENT,'Wrong A102 plan')
 for name,sha in p['sources'].items():files.require(files.digest(PROJECTS/name)==sha,'Source changed: '+name)
 for name,sha in p['inputs'].items():files.require(files.digest(name)==sha,'Input changed: '+name)
 files.require(files.digest(Path(root)/'motion_kernel.npy')==p['motion_kernel_sha256'],'Kernel changed');return p

def prepare(args):
 import numpy as np
 root=Path(args.root).resolve();parent=Path(args.a71_root).resolve()
 if (root/'plan.json').exists():check(root);print('A102 frozen plan reused.');return
 files.require(not root.exists() or not any(root.iterdir()),'Nonempty A102 root retained')
 from utils.diag_shrink_multitask import check as parent_check
 pp=parent_check(parent);files.require(files.read(parent/'summary.json')['completed_runs']==72,'Complete A71 anchor required')
 cases=rules.cases(args.ffhq_dir)
 for c in cases:files.require(Path(c['image']).is_file(),'Missing image');c['image_sha256']=files.digest(c['image'])
 root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=4*files.GIB,'Need4GiB')
 root.mkdir();from utils.diag_ffhq_four_tasks_unit import original_motion_kernel
 np.save(root/'motion_kernel.npy',original_motion_kernel(0),allow_pickle=False)
 names=('utils/shared_prefix_feedback.py','utils/shared_prefix_feedback_runtime.py','utils/diag_shared_prefix_feedback.py',
  'utils/a102_session.py','utils/test_shared_prefix_feedback.py','utils/test_shared_prefix_feedback_runtime.py',
  'utils/equal_nfe_feedback_runtime.py','utils/equal_nfe_feedback_runtime_r1.py','utils/equal_nfe_feedback_runtime_r3.py',
  'utils/diag_equal_nfe_feedback.py','utils/shrink_multitask_runtime.py','utils/noise_signal_shrink.py',
  'utils/paper_clock_runtime.py','utils/historical_ffhq_gain.py','modules/zaps_algorithm.py','modules/wavelet.py',
  'run_a102_20261110.sh',str(CARD.relative_to(PROJECTS)))
 protocol=dict(scope='FFHQ IDs00000-00009 x4 tasks x sigma_model.05; three suffixes from one exact common prefix.',
  accounting='Common prefix270F/9B once, each suffix30F/1B. Method-equivalent per arm300F/10B; actual joint case360F/12B. Both reported.',
  pairing='Same learned zeta/D, initial, RNG, y, operator and grid by construction. No GT action/output selection.',
  baseline='Local September17 ZAPS configuration; not author benchmark parity.',
  controllers='Native ZAPS suffix; A71 closed scalar; closed diagonal with real H-adjoint and forward-exact native-q straight-through only for final unused update.',
  decision='Per task: mean PSNR>=+.1dB, >=7/10 wins, SSIM nondecrease, LPIPS nonincrease. All cells retained.',
  limits='Shared-prefix causal suffix comparison at sigma.05; no multi-noise/data/model or acceleration claim.')
 p=dict(experiment=rules.EXPERIMENT,cases=cases,checkpoint=pp['checkpoint'],grid=files.grid(),arms=list(rules.ARMS),
  configs={t:rules.configuration(t) for t in rules.TASKS},motion_kernel_sha256=files.digest(root/'motion_kernel.npy'),
  budget=dict(cases=40,method_equivalent_per_arm=dict(forwards=300,backwards=10),actual_joint=dict(forwards=14400,backwards=480),input_vjp=0,input_jvp=0),
  sources={name:files.digest(PROJECTS/name) for name in names},inputs={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')},protocol=protocol)
 write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md');print('A102 PREPARE:40 cases, actual14400F/480B; equivalent300F/10B per arm.',flush=True)

def operator(root,c,device,count=None):
 import numpy as np
 return runtime.make_operator(c,device,count=count,kernel=np.load(Path(root)/'motion_kernel.npy',allow_pickle=False))

def rng_equal(a,b):
 import torch
 return torch.equal(a['cpu'],b['cpu']) and len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))

def validate(count,F,B):
 files.require(count['forwards']==count['forward_attempts']==F,'Forward count mismatch')
 files.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==B,'Backward mismatch')
 files.require(count.get('input_vjp_attempts',0)==count.get('input_jvp_attempts',0)==0,'Input derivative forbidden')

def run_case(root,p,c,model,uuid):
 import torch
 from modules.main_single import load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 folder=Path(root)/'cases'/c['key'];record=folder/'run.json'
 if record.exists():files.require(files.read(record)['status']=='complete','Prior failed A102 case retained');return
 files.require(resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted'],'Resource lost')
 obs=fixtures.observation(root,c,model.device);folder.mkdir(parents=True,exist_ok=False)
 row=dict(status='running',case=c,plan_sha256=files.digest(Path(root)/'plan.json'),counts={},arms={},cost_lower_bound=True);write(record,row)
 try:
  y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as pc:
   (prefix,prefix_seconds)=clock.clocks(torch.cuda.synchronize,lambda:runtime.train_prefix(model,operator(root,c,model.device,pc),y,c))
  validate(pc,270,9);files.require(pc['calls']==files.grid()[::-1]*9,'Prefix grid differs');row['counts']['prefix']=pc;write(record,row)
  final_rng=None
  for arm in rules.ARMS:
   with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as sc:
    (payload,suffix_seconds)=clock.clocks(torch.cuda.synchronize,lambda arm=arm:runtime.run_suffix(model,operator(root,c,model.device,sc),y,c,arm,prefix))
   output,detail=payload;validate(sc,30,1);files.require(sc['calls']==files.grid()[::-1],'Suffix grid differs')
   if final_rng is None:final_rng=detail['final_rng']
   else:files.require(rng_equal(final_rng,detail['final_rng']),'Suffix RNG differs')
   metrics=common.metrics(output,load_image_as_tensor(c['image']).cpu())
   torch.save(output,folder/(arm+'.pt'));tensor_to_image(output.squeeze(0),denormalize=True).save(folder/(arm+'.png'))
   write(folder/(arm+'_trace.json'),detail['trace'])
   row['arms'][arm]=dict(metrics=metrics,suffix_seconds=suffix_seconds,
    method_equivalent_seconds=prefix_seconds+suffix_seconds,counts=sc,module_counts=detail['module_counts'],loss=detail['loss'],
    artifacts={name:files.digest(folder/name) for name in (arm+'.pt',arm+'.png',arm+'_trace.json')})
   write(record,row);print('A102 COMPLETE:',c['key'],arm,metrics,flush=True)
  row.update(status='complete',prefix_seconds=prefix_seconds,prefix_losses=prefix['losses'],counts={**row['counts'],'prefix':pc},
   peak_reserved_bytes=torch.cuda.max_memory_reserved(),cost_lower_bound=False);write(record,row)
 except BaseException as e:
  row.update(status='failed',error=repr(e),traceback=traceback.format_exc());write(record,row);raise

def worker(args):
 import torch
 from modules.main_single import load_diffusion_model
 from utils.a54_session import exclusive
 root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
 with exclusive(folder/'worker.lock'):
  model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
  try:
   with frozen_network(model.model):
    for c in p['cases']:
     if c['shard']==args.shard:run_case(root,p,c,model,args.device_uuid)
   write(folder/'status.json',dict(status='complete'))
  except BaseException as e:write(folder/'status.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise

def summarize(root):
 root=Path(root);p=check(root);rows=[files.read(x) for x in (root/'cases').glob('*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
 for task in rules.TASKS:
  cell=[r for r in done if r['case']['task']==task]
  for arm in rules.ARMS:
   if cell:groups.append(dict(task=task,arm=arm,images=len(cell),quality={k:statistics.mean(r['arms'][arm]['metrics'][k] for r in cell) for k in ('psnr','ssim','lpips')},
    mean_equivalent_seconds=statistics.mean(r['arms'][arm]['method_equivalent_seconds'] for r in cell)))
  for cand,ref in (('CLOSED_SCALAR_SUFFIX','ZAPS_SUFFIX'),('CLOSED_DIAG_SUFFIX','ZAPS_SUFFIX'),('CLOSED_DIAG_SUFFIX','CLOSED_SCALAR_SUFFIX')):
   if cell:
    delta={k:statistics.mean(r['arms'][cand]['metrics'][k]-r['arms'][ref]['metrics'][k] for r in cell) for k in ('psnr','ssim','lpips')};wins=sum(r['arms'][cand]['metrics']['psnr']>r['arms'][ref]['metrics']['psnr'] for r in cell)
    comparisons.append(dict(task=task,candidate=cand,reference=ref,images=len(cell),delta=delta,psnr_wins=wins,positive=len(cell)==10 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0 and wins>=7))
 summary=dict(experiment=rules.EXPERIMENT,classification='SHARED_PREFIX_EQUIVALENT_300_NFE',completed_cases=len(done),planned_cases=40,groups=groups,comparisons=comparisons,
  actual_cost=dict(forwards=sum(r.get('counts',{}).get('prefix',{}).get('forwards',0)+sum(a.get('counts',{}).get('forwards',0) for a in r.get('arms',{}).values()) for r in rows)),cost_is_lower_bound=len(done)<40,protocol=p['protocol'])
 write(root/'summary.json',summary);print('A102 Summary:',json.dumps(summary,ensure_ascii=False),flush=True);return summary

def archive(root):
 from utils import archive_refinement_confirmation as exporter
 exporter.BRANCH=rules.BRANCH;bundle,commit=exporter.export_and_commit({'A102':Path(root)},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',commit,'Branch:',rules.BRANCH);subprocess.run(['git','push','origin',rules.BRANCH],cwd=PROJECTS.parent,check=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=rules.ROOT);p.add_argument('--a71-root',default=PARENT);p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--shard',type=int);p.add_argument('--device-uuid')
 g=p.add_mutually_exclusive_group(required=True)
 for n in ('prepare','worker','summarize','archive'):g.add_argument('--'+n,action='store_true')
 a=p.parse_args()
 if a.prepare:prepare(a)
 elif a.worker:worker(a)
 elif a.summarize:summarize(a.root)
 else:archive(a.root)
if __name__=='__main__':main()
