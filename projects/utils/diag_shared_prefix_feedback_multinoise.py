"""A103 multi-noise wrapper around the completed A102 mechanics."""
import argparse,json,statistics,shutil,subprocess,sys
from pathlib import Path
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_shared_prefix_feedback as base
from utils import shared_prefix_feedback_multinoise as rules
from utils import shared_prefix_feedback_runtime_multinoise as runtime
from utils import historical_ffhq_gain as files
from utils.diag_paper_clock_alignment import write
CARD=PROJECTS/'experiment_cards/A103_shared_prefix_multinoise_20261111.md'
base.rules=rules;base.runtime=runtime

def check(root):
 p=files.read(Path(root)/'plan.json');files.require(p['experiment']==rules.EXPERIMENT,'Wrong A103 plan')
 for n,s in p['sources'].items():files.require(files.digest(PROJECTS/n)==s,'Source changed: '+n)
 for n,s in p['inputs'].items():files.require(files.digest(n)==s,'Input changed: '+n)
 files.require(files.digest(Path(root)/'motion_kernel.npy')==p['motion_kernel_sha256'],'Kernel changed');return p

def prepare(a):
 import numpy as np
 root=Path(a.root).resolve();parent=Path(a.a71_root).resolve()
 if (root/'plan.json').exists():check(root);print('A103 plan reused.');return
 files.require(not root.exists() or not any(root.iterdir()),'Nonempty A103 root')
 from utils.diag_shrink_multitask import check as pc
 pp=pc(parent);files.require(files.read(parent/'summary.json')['completed_runs']==72,'Complete A71 required')
 cases=rules.cases(a.ffhq_dir)
 for c in cases:files.require(Path(c['image']).is_file(),'Missing image');c['image_sha256']=files.digest(c['image'])
 root.parent.mkdir(parents=True,exist_ok=True);root.mkdir();from utils.diag_ffhq_four_tasks_unit import original_motion_kernel
 np.save(root/'motion_kernel.npy',original_motion_kernel(0),allow_pickle=False)
 names=('utils/shared_prefix_feedback_multinoise.py','utils/shared_prefix_feedback_runtime_multinoise.py',
  'utils/diag_shared_prefix_feedback_multinoise.py','utils/a103_session.py','utils/test_shared_prefix_feedback_multinoise.py',
  'utils/shared_prefix_feedback.py','utils/shared_prefix_feedback_runtime.py','utils/diag_shared_prefix_feedback.py',
  'utils/equal_nfe_feedback_runtime.py','utils/equal_nfe_feedback_runtime_r1.py','utils/equal_nfe_feedback_runtime_r3.py',
  'run_a103_20261111.sh',str(CARD.relative_to(PROJECTS)))
 p=dict(experiment=rules.EXPERIMENT,cases=cases,checkpoint=pp['checkpoint'],grid=files.grid(),arms=list(rules.ARMS),
  configs={t:rules.configuration(t) for t in rules.TASKS},motion_kernel_sha256=files.digest(root/'motion_kernel.npy'),
  sources={n:files.digest(PROJECTS/n) for n in names},inputs={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')},
  budget=dict(method_NFE=300,planned_cases=80,actual_joint_forwards=80*360,input_vjp=0,input_jvp=0),
  protocol=dict(scope='Same10 FFHQ x4tasks x sigma.025/.10; A102 sigma.05 not rerun.',NFE='Each arm method-equivalent300F; backward and wallclock separate.',qualitative='Fixed IDs00001/00004/00008, no outcome selection.',limits='Local paired noise-boundary confirmation; no new dataset/model.'))
 write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md');print('A103 PREPARE:80 cases, each arm300 NFE.',flush=True)

def worker(a):
 with patch.object(base,'check',check):base.worker(a)

def summarize(root):
 root=Path(root);p=check(root);rows=[files.read(x) for x in (root/'cases').glob('*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
 for task in rules.TASKS:
  for sigma in rules.SIGMAS:
   cell=[r for r in done if r['case']['task']==task and r['case']['sigma']==sigma]
   for arm in rules.ARMS:
    if cell:groups.append(dict(task=task,sigma=sigma,arm=arm,images=len(cell),quality={k:statistics.mean(r['arms'][arm]['metrics'][k] for r in cell) for k in ('psnr','ssim','lpips')},mean_equivalent_seconds=statistics.mean(r['arms'][arm]['method_equivalent_seconds'] for r in cell)))
   for cand,ref in (('CLOSED_SCALAR_SUFFIX','ZAPS_SUFFIX'),('CLOSED_DIAG_SUFFIX','ZAPS_SUFFIX'),('CLOSED_DIAG_SUFFIX','CLOSED_SCALAR_SUFFIX')):
    if cell:
     d={k:statistics.mean(r['arms'][cand]['metrics'][k]-r['arms'][ref]['metrics'][k] for r in cell) for k in ('psnr','ssim','lpips')};w=sum(r['arms'][cand]['metrics']['psnr']>r['arms'][ref]['metrics']['psnr'] for r in cell)
     comparisons.append(dict(task=task,sigma=sigma,candidate=cand,reference=ref,images=len(cell),delta=d,psnr_wins=w,positive=len(cell)==10 and d['psnr']>=.1 and d['ssim']>=0 and d['lpips']<=0 and w>=7))
 summary=dict(experiment=rules.EXPERIMENT,classification='SHARED_PREFIX_300_NFE_MULTI_NOISE',completed_cases=len(done),planned_cases=80,groups=groups,comparisons=comparisons,cost_is_lower_bound=len(done)<80,protocol=p['protocol']);write(root/'summary.json',summary);print('A103 Summary:',json.dumps(summary,ensure_ascii=False));return summary

def report(root):
 from PIL import Image,ImageDraw,ImageFont
 root=Path(root);p=check(root);out=root/'qualitative';out.mkdir(exist_ok=True);font=ImageFont.load_default();labels=('GT','Observation',*rules.ARMS)
 for task in rules.TASKS:
  for sigma in rules.SIGMAS:
   canvas=Image.new('RGB',(1280,858),'white');draw=ImageDraw.Draw(canvas)
   for ri,image_id in enumerate(rules.PANEL_IMAGES):
    c=next(x for x in p['cases'] if x['task']==task and x['sigma']==sigma and x['image_id']==image_id);obs=root/'observations'/c['key']
    paths=[obs/'ground_truth.png',obs/'observation.png']+[root/'cases'/c['key']/(arm+'.png') for arm in rules.ARMS]
    for ci,(label,path) in enumerate(zip(labels,paths)):
     im=Image.open(path).convert('RGB').resize((256,256));canvas.paste(im,(ci*256,ri*286));draw.text((ci*256+4,ri*286+260),label,fill='black',font=font)
   canvas.save(out/f'{task}_sigma{round(sigma*1000):03d}.png')
 print('A103 QUALITATIVE:',len(list(out.glob('*.png'))),flush=True)

def archive(root):
 from utils import archive_refinement_confirmation as ex
 ex.BRANCH=rules.BRANCH;b,c=ex.export_and_commit({'A103':Path(root)},projects=PROJECTS);print('Archive:',b,'Commit:',c);subprocess.run(['git','push','origin',rules.BRANCH],cwd=PROJECTS.parent,check=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=rules.ROOT);p.add_argument('--a71-root',default=base.PARENT);p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--shard',type=int);p.add_argument('--device-uuid');g=p.add_mutually_exclusive_group(required=True)
 for n in ('prepare','worker','summarize','report','archive'):g.add_argument('--'+n,action='store_true')
 a=p.parse_args()
 if a.prepare:prepare(a)
 elif a.worker:worker(a)
 elif a.summarize:summarize(a.root)
 elif a.report:report(a.root)
 else:archive(a.root)
if __name__=='__main__':main()
