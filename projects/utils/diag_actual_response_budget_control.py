"""A104 driver: complete-clock ZAPS300/ZAPS450/actual-response445 comparison."""
import argparse,gc,json,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import actual_response_budget_control as rules,actual_response_budget_runtime as runtime
from utils import diag_equal_budget_ray_diagonal as parent_driver
from utils import historical_ffhq_gain as files,historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock,diag_noise_feasible_zaps as common
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import write,resource

CARD=PROJECTS/'experiment_cards/A104_actual_response_budget_control_20261111.md'


def check(root):
 p=files.read(Path(root)/'plan.json');files.require(p['experiment']==rules.EXPERIMENT,'Wrong A104 plan')
 for name,sha in p['sources'].items():files.require(files.digest(PROJECTS/name)==sha,'Source changed: '+name)
 for name,sha in p['inputs'].items():files.require(files.digest(name)==sha,'Input changed: '+name)
 return p


def bind_record(path,bound):
 path=Path(path);row=files.read(path);files.require(row['status']=='complete','Incomplete reference')
 bound[str(path)]=files.digest(path)
 for name,sha in row['artifacts'].items():
  artifact=path.parent/name;files.require(files.digest(artifact)==sha,'Changed reference artifact');bound[str(artifact)]=sha
 return str(path)


def prepare(args):
 root=Path(args.root).resolve();parent=Path(args.a100_root).resolve()
 if (root/'plan.json').exists():check(root);print('A104 frozen plan reused.');return
 files.require(not root.exists() or not any(root.iterdir()),'Nonempty A104 root retained')
 pp=parent_driver.check(parent);files.require(files.read(parent/'summary.json')['completed_runs']==24,'Complete A100 required')
 bound={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')};cases=[]
 for index,c0 in enumerate(pp['cases']):
  c=dict(c0);c['shard']=index%2;c['references']={}
  c['references']['ZAPS300']=bind_record(c['comparison_references']['BASE'],bound)
  c['references']['REAL_RAY445']=bind_record(parent/'cases'/c['key']/'RAY_K1_EQ'/'run.json',bound)
  c['references']['REAL_DIAG445']=bind_record(parent/'cases'/c['key']/'DIAG_K1_EQ'/'run.json',bound)
  for name in ('image','entry','observation'):
   path=Path(c[name]);files.require(path.exists(),'Missing '+name);bound[str(path)]=files.digest(path)
  cases.append(c)
 files.require(len(cases)==12 and all(c['sigma']==.05 for c in cases),'A104 fixed12 cases required')
 names=('utils/actual_response_budget_control.py','utils/actual_response_budget_runtime.py','utils/diag_actual_response_budget_control.py',
  'utils/a104_session.py','utils/test_actual_response_budget_control.py','utils/equal_budget_ray_diagonal.py','utils/full_unroll_response.py',
  'utils/response_target_runtime.py','utils/shared_prefix_feedback.py','utils/paper_clock_runtime.py','run_a104_20261111.sh',str(CARD.relative_to(PROJECTS)))
 protocol=dict(scope='Frozen A100 three FFHQ images x four tasks x sigma_model.05; same observations/operators/seeds/grid.',
  accounting='NFE counts denoiser forwards only. ZAPS300=300F/10 optimizer backwards; ZAPS450=450F/15B; response methods=445F/9B/29VJP/29JVP. Operator and wavelet calls separate.',
  comparison='Primary same-derivative comparison diagonal445 vs ray445. ZAPS450 is a near-NFE control for extra forwards, not derivative-equivalent. Full method-equivalent clocks are reported.',
  pairing='Response arms share exactly one independently reproduced 270F/9B learned prefix, then independently replay the same RNG. ZAPS arms are independent from-seed executions.',
  gates='ZAPS300 must reproduce frozen BASE; response arms must reproduce frozen A100 outputs; learned prefix snapshot must reproduce frozen entry. No failed-case retry or quality fallback.',
  limits='Small reused development panel. No author benchmark, universal improvement, acceleration or posterior guarantee.')
 root.mkdir(parents=True);p=dict(experiment=rules.EXPERIMENT,parent=str(parent),a71_parent=pp['a71_parent'],cases=cases,checkpoint=pp['checkpoint'],grid=pp['grid'],arms=list(rules.ARMS),
  sources={name:files.digest(PROJECTS/name) for name in names},inputs=bound,protocol=protocol,
  budget=dict(cases=12,method_costs={a:rules.expected_cost(a) for a in rules.ARMS},actual_joint_forwards_per_case=1370))
 write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md');print('A104 PREPARE:12 cases; 300/450/445 NFE paths frozen.',flush=True)


def reference_tensor(record):
 import torch
 row=files.read(record);candidates=[name for name in row['artifacts'] if name.endswith('output.pt') or name=='output.pt']
 files.require(len(candidates)==1,'Ambiguous reference output')
 return torch.load(Path(record).parent/candidates[0],map_location='cpu',weights_only=True)


def relative_error(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp(min=1e-30))


def validate(count,expected):
 files.require(count['forwards']==count['forward_attempts']==expected['forwards'],'NFE mismatch')
 files.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==expected['optimizer_backward'],'Optimizer backward mismatch')
 files.require(count.get('input_vjp_attempts',0)==expected['input_vjp'],'VJP mismatch')
 files.require(count.get('input_jvp_attempts',0)==expected['input_jvp'],'JVP mismatch')


def run_case(root,p,c,model,uuid):
 import torch
 from modules.main_single import load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 folder=Path(root)/'cases'/c['key'];record=folder/'run.json'
 if record.exists():files.require(files.read(record)['status']=='complete','Prior failed A104 case retained');return
 files.require(resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted'],'Resource lost');folder.mkdir(parents=True)
 row=dict(status='running',case=c,arms={},counts={},cost_is_lower_bound=True);write(record,row)
 try:
  y=torch.load(c['observation'],map_location='cpu',weights_only=True)['y'].to(model.device);gt=load_image_as_tensor(c['image']).cpu()
  # Independent native controls.
  for arm,epochs in (('ZAPS300',10),('ZAPS450',15)):
   gc.collect();torch.cuda.empty_cache()
   with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count:
    def native_operation():
     op=runtime.make_operator(p['a71_parent'],c,model.device);clock.count_operator(op,count)
     return runtime.train_zaps(model,op,y,c,epochs)
    (payload,seconds)=clock.clocks(torch.cuda.synchronize,native_operation)
   output,detail=payload;validate(count,rules.expected_cost(arm));gate=None
   if arm=='ZAPS300':gate=relative_error(output,reference_tensor(c['references'][arm]));files.require(gate<=1e-6,'ZAPS300 reproduction failed')
   metrics=common.metrics(output,gt);torch.save(output,folder/(arm+'.pt'));tensor_to_image(output.squeeze(0),denormalize=True).save(folder/(arm+'.png'))
   row['arms'][arm]=dict(metrics=metrics,method_seconds=seconds,counts=count,reproduction_relative_L2=gate,losses=detail['losses'],artifacts={n:files.digest(folder/n) for n in (arm+'.pt',arm+'.png')});write(record,row)
   print('A104 COMPLETE:',c['key'],arm,metrics,flush=True)
  # One shared, independently reproduced 270F/9B prefix for the two response parameterizations.
  gc.collect();torch.cuda.empty_cache()
  with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as pc:
   def prefix_operation():
    op=runtime.make_operator(p['a71_parent'],c,model.device);clock.count_operator(op,pc)
    return runtime.train_zaps(model,op,y,c,9)
   (prefix_payload,prefix_seconds)=clock.clocks(torch.cuda.synchronize,prefix_operation)
  _,prefix=prefix_payload;validate(pc,dict(forwards=270,optimizer_backward=9,input_vjp=0,input_jvp=0))
  cached=torch.load(c['entry'],map_location='cpu',weights_only=True);prefix_gate=clock.snapshot_equal(prefix['entry'],cached);files.require(prefix_gate['passed'],'Learned prefix reproduction failed')
  final_rng=None
  for arm in rules.RESPONSE_ARMS:
   with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as sc:
    op=runtime.make_operator(p['a71_parent'],c,model.device);clock.count_operator(op,sc)
    (payload,suffix_seconds)=clock.clocks(torch.cuda.synchronize,lambda arm=arm:runtime.response_suffix(model,op,y,c,arm,prefix,sc))
   output,trace,rng=payload;expected=rules.expected_cost(arm);validate(sc,dict(forwards=175,optimizer_backward=0,input_vjp=29,input_jvp=29))
   if final_rng is None:final_rng=rng
   else:
    from utils.response_target_runtime import rng_equal
    files.require(rng_equal(final_rng,rng),'Response-arm RNG differs')
   gate=relative_error(output,reference_tensor(c['references'][arm]));files.require(gate<=1e-6,'A100 response reproduction failed')
   metrics=common.metrics(output,gt);torch.save(output,folder/(arm+'.pt'));tensor_to_image(output.squeeze(0),denormalize=True).save(folder/(arm+'.png'));write(folder/(arm+'_trace.json'),trace)
   method_counts=dict(forwards=270+sc['forwards'],optimizer_backward_attempts=9,input_vjp_attempts=sc['input_vjp_attempts'],input_jvp_attempts=sc['input_jvp_attempts'])
   files.require(method_counts['forwards']==expected['forwards'],'Equivalent method NFE mismatch')
   row['arms'][arm]=dict(metrics=metrics,method_seconds=prefix_seconds+suffix_seconds,prefix_seconds=prefix_seconds,suffix_seconds=suffix_seconds,counts=method_counts,
    suffix_counts=sc,reproduction_relative_L2=gate,artifacts={n:files.digest(folder/n) for n in (arm+'.pt',arm+'.png',arm+'_trace.json')});write(record,row)
   print('A104 COMPLETE:',c['key'],arm,metrics,flush=True)
  row.update(status='complete',prefix_counts=pc,prefix_gate=prefix_gate,cost_is_lower_bound=False);write(record,row)
 except BaseException as e:row.update(status='failed',error=repr(e),traceback=traceback.format_exc());write(record,row);raise


def worker(args):
 import torch
 from modules.main_single import load_diffusion_model
 from utils.a54_session import exclusive
 from utils import response_target_runtime as rt,a95_checkpoint_scope_r1 as repair
 root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
 with exclusive(folder/'worker.lock'):
  model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
  try:
   with frozen_network(model.model),patch.object(rt,'forward_ad_checkpoint_scope',repair.forward_ad_checkpoint_scope):
    probe={};rt.checkpoint_probe(model,p['grid'][16],probe);write(folder/'checkpoint_probe.json',probe)
    for c in p['cases']:
     if c['shard']==args.shard:run_case(root,p,c,model,args.device_uuid)
   write(folder/'status.json',dict(status='complete'))
  except BaseException as e:write(folder/'status.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise


def summarize(root):
 root=Path(root);p=check(root);rows=[files.read(x) for x in (root/'cases').glob('*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
 for task in rules.TASKS:
  cell=[r for r in done if r['case']['task']==task]
  for arm in rules.ARMS:
   if cell:groups.append(dict(task=task,arm=arm,images=len(cell),quality={k:statistics.mean(r['arms'][arm]['metrics'][k] for r in cell) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['arms'][arm]['method_seconds'] for r in cell),cost=rules.expected_cost(arm)))
  for cand,ref in (('ZAPS450','ZAPS300'),('REAL_RAY445','ZAPS300'),('REAL_DIAG445','ZAPS300'),('REAL_RAY445','ZAPS450'),('REAL_DIAG445','ZAPS450'),('REAL_DIAG445','REAL_RAY445')):
   if cell:
    delta={k:statistics.mean(r['arms'][cand]['metrics'][k]-r['arms'][ref]['metrics'][k] for r in cell) for k in ('psnr','ssim','lpips')}
    comparisons.append(dict(task=task,candidate=cand,reference=ref,images=len(cell),delta=delta,psnr_wins=sum(r['arms'][cand]['metrics']['psnr']>r['arms'][ref]['metrics']['psnr'] for r in cell)))
 summary=dict(experiment=p['experiment'],classification='FULL_CLOCK_300_450_445_NFE_CONTROL',completed_cases=len(done),planned_cases=12,groups=groups,comparisons=comparisons,
  actual_joint_cost=dict(forwards=1370*len(done),optimizer_backward=34*len(done),input_vjp=58*len(done),input_jvp=58*len(done)),
  summed_method_equivalent_cost=dict(forwards=1640*len(done),optimizer_backward=43*len(done),input_vjp=58*len(done),input_jvp=58*len(done)),
  cost_is_lower_bound=len(done)<12,protocol=p['protocol'])
 write(root/'summary.json',summary);print('A104 Summary:',json.dumps(summary,ensure_ascii=False),flush=True);return summary


def archive(root):
 from utils import archive_refinement_confirmation as exporter
 exporter.BRANCH=rules.BRANCH;bundle,commit=exporter.export_and_commit({'A104':Path(root)},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',commit,'Branch:',rules.BRANCH);subprocess.run(['git','push','origin',rules.BRANCH],cwd=PROJECTS.parent,check=True)


def main():
 q=argparse.ArgumentParser();q.add_argument('--root',default=rules.ROOT);q.add_argument('--a100-root',default=rules.PARENT);q.add_argument('--shard',type=int);q.add_argument('--device-uuid')
 g=q.add_mutually_exclusive_group(required=True)
 for name in ('prepare','worker','summarize','archive'):g.add_argument('--'+name,action='store_true')
 a=q.parse_args()
 if a.prepare:prepare(a)
 elif a.worker:worker(a)
 elif a.summarize:summarize(a.root)
 else:archive(a.root)
if __name__=='__main__':main()
