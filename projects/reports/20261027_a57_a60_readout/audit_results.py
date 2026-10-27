"""Read-only arithmetic and lightweight archive receipt audit; no Torch/model calls."""
import hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
mean=statistics.mean
close=lambda a,b:math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
heads=read(ROOT/'remote_heads.json');receipts=read(ROOT/'remote_receipts.json');data={};proof=[]
for r in receipts:
 n=r['experiment'];summary=ROOT/'received'/f'{n}_summary.json';manifest=ROOT/'received'/f'{n}_manifest.json'
 for f,key in ((summary,'summary_blob_sha'),(manifest,'manifest_blob_sha')):
  blob=hashlib.sha1(b'blob '+str(f.stat().st_size).encode()+b'\0'+f.read_bytes()).hexdigest()
  assert blob==r[key],(n,key,'Git blob mismatch')
 m=read(manifest);entry=next(x for x in m['files'] if x['path']=='summary.json')
 assert entry['bytes']==summary.stat().st_size and entry['sha256']==hashlib.sha256(summary.read_bytes()).hexdigest()
 data[n]=read(summary);proof.append(dict(**r,observed_branch_head=heads[n],manifest_summary_sha256=entry['sha256'],summary_hash_verified=True,scope='Fetched summary/manifest only. No PT download or tensor/model replay.'))
expected={'A58':'e931715a8d325740a81461fdea05b55b88eefaea','A59':'ee50d15f173a99a43fbc59919b413637f9a71dd8','A60':'a3bb71d9b8c80398c5d933c0ea0bf09f8752e3f6'}
assert all(heads[k]==v for k,v in expected.items())
# Pasted files may omit bulky cached predictions/per-image fields; compare every supplied value.
attachments={'A57':'57749f08-6ae2-40b1-857d-1642cf929c7e','A58':'6f68e654-fbd5-4806-bed0-38f8b3fdd467','A59':'85c331ae-884e-4083-975e-4a64ec5749c4'}
def subset(a,b):
 if isinstance(a,dict):return isinstance(b,dict) and all(k in b and subset(v,b[k]) for k,v in a.items())
 if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(subset(x,y) for x,y in zip(a,b))
 if isinstance(a,str) and isinstance(b,str):return ''.join(a.split())==''.join(b.split())
 return a==b
for n,token in attachments.items():
 text=(Path('/Users/liziyue.22/.codex/attachments')/token/'已粘贴的文本.txt').read_text()
 pasted,_=json.JSONDecoder().raw_decode(text[text.index('{'):])
 assert subset(pasted,data[n]),(n,'User excerpt differs from uploaded result')
# A57 numerical equivalence, only five reused image clusters.
s=data['A57'];assert len(s['per_image'])==10
assert all(all(r['comparison'][k] for k in ('fully_resolved','equivalent','initial_equal','innovations_equal','source_cpu_rng_equal')) for r in s['per_image'])
assert max(r['comparison']['relative_error'] for r in s['per_image'])==s['max_relative_error']<1e-6
assert s['method_nfe']==s['input_backward_attempts']==s['input_backward_completed']==300 and s['optimizer_updates']==0
a57=dict(all_ten_equivalent=True,max_relative_error=s['max_relative_error'],forward_per_case=30,forward_count_reduction_vs_inherited_300_protocol=.9,input_VJP_per_case=30,mean_metrics={t:{k:mean(r['metrics'][k] for r in s['per_image'] if r['case']['task']==t) for k in ('psnr','ssim','lpips')} for t in ('super_resolution','inpainting')},max_abs_delta_vs_cached_U={k:max(abs(r['delta_vs_cached_U'][k]) for r in s['per_image']) for k in ('psnr','ssim','lpips')},scope='Old five FFHQ images, SR/IP, one seed; equivalence to cached U, not original C or all future inputs. No10x walltime claim.')
# A58 fixed numerical qualification, not a quality failure.
s=data['A58'];rows=s['numerical_rows'];pcg=[r for r in rows if r['variant']=='PCG64']
assert len(rows)==24 and len(pcg)==8 and not all(r['gate']['passed'] for r in pcg)
assert s['diagnostic_nfe']==s['method_nfe']==s['completed_cases']==0
assert sum(r['gate']['operator_calls'] for r in rows)==s['metrology_operator_calls']
a58=dict(pcg_passes=sum(r['gate']['passed'] for r in pcg),pcg_solves=8,high_noise_failed=[dict(task=r['task'],rhs=r['rhs'],relative_residual=r['gate']['relative_residual']) for r in pcg if not r['gate']['passed']],low_noise_passes=sum(r['gate']['passed'] and r['t']==319 for r in pcg),quality_not_run=True,scope='Stop this128-cap periodic-preconditioner admission protocol; not mathematical impossibility of blur inversion.')
# A59 independent aggregate/contrast arithmetic from archived endpoints and frozen cross-validation predictions.
s=data['A59'];pred=s['analysis']['predictions'];bank={(r['case']['image_id'],r['case']['task']):r['metrics'] for r in s['per_image']}
assert len(pred)==30 and s['analysis']['usable_states']==30
for target in ('psnr','ssim','lpips','loss'):
 for name in ('STAGE','PHYSICAL','GEOMETRY'):
  value=mean(abs(r['values'][target][name]-r['values'][target]['truth']) for r in pred)
  assert close(value,s['analysis']['targets'][target]['mae'][name])
 for r in pred:
  b=bank[r['image_id'],r['task']]
  assert close(r['values'][target]['truth'],b[f"K{r['step']}_G1.5"][target]-b[f"K{r['step']}_G0.5"][target])
for g in s['groups']:
 selected=[r['metrics'] for r in s['per_image'] if r['case']['task']==g['task']]
 for arm,deltas in g['mean_delta_vs_U'].items():
  for target,value in deltas.items():assert close(mean(r[arm][target]-r['U_BASE'][target] for r in selected),value)
# Additional post-hoc readout only: MAE improvement is not automatically a useful control policy.
extra={}
for task in ('super_resolution','inpainting','ALL'):
 rows=[r for r in pred if task=='ALL' or r['task']==task];stats={}
 for name in ('STAGE','PHYSICAL','GEOMETRY'):
  gains=[]
  for r in rows:
   label=r['values']['psnr'];gamma=1.5 if label[name]>0 else .5;b=bank[r['image_id'],r['task']]
   gains.append(b[f"K{r['step']}_G{gamma:g}"]['psnr']-b['U_BASE']['psnr'])
  stats[name]=dict(MAE=mean(abs(r['values']['psnr'][name]-r['values']['psnr']['truth']) for r in rows),sign_correct=sum((r['values']['psnr'][name]>0)==(r['values']['psnr']['truth']>0) for r in rows),n=len(rows),cached_forced_binary_gain_vs_U=mean(gains))
 stats['action_differences_GEOMETRY_vs_STAGE']=sum((r['values']['psnr']['GEOMETRY']>0)!=(r['values']['psnr']['STAGE']>0) for r in rows)
 extra[task]=stats
a59=dict(primary=s['analysis']['targets']['psnr'],per_task_and_cached_posthoc=extra,posthoc_limits='New arithmetic of existing cached endpoints/predictions ONLY; no new UNet, no prospective policy, no retuning. Each row is a single-step intervention, not a composed multi-step rollout. Gamma1 no-action option absent from forced binary diagnostic; do not deploy based on its MAE alone.')
# A60 same-run means, coverage, resource counts.
s=data['A60'];rows=s['per_image'];assert len(rows)==10
for g in s['groups']:
 group=[r for r in rows if r['case']['task']==g['task']]
 for arm,armdata in g['arms'].items():
  for ref,deltas in armdata['mean_delta'].items():
   for target,value in deltas.items():assert close(mean(r['metrics'][arm][target]-r['metrics'][ref][target] for r in group),value)
for arm in ('CURRENT','LAG2','LAG4'):
 assert sum(r['input_VJPs'][arm] for r in rows)==s['input_VJPs_by_arm'][arm]
 assert close(sum(r['arm_seconds'][arm] for r in rows),s['measured_seconds_by_arm'][arm])
 assert all(r['resolved'][arm]==30 for r in rows)
a60=dict(fully_resolved_all_paths=True,lag2_PSNR_worse_images=sum(r['metrics']['LAG2']['psnr']<r['metrics']['CURRENT']['psnr'] for r in rows),lag4_PSNR_worse_images=sum(r['metrics']['LAG4']['psnr']<r['metrics']['CURRENT']['psnr'] for r in rows),total_forward=900,input_VJP_by_arm=s['input_VJPs_by_arm'],groups=s['groups'],scope='Direction lag includes old x,t,v; not a matched frozen-Jacobian-on-current-v test. No quality-preserving low-VJP arm confirmed.')
result=dict(lightweight_return_receipts=proof,user_excerpt_numeric_exact_text_whitespace_equivalent={'A57':True,'A58':True,'A59':True},A60_user_inline_key_fields_checked=True,A57=a57,A58=a58,A59=a59,A60=a60,new_returned_cost=dict(method_forward=data['A57']['method_nfe'],diagnostic_forward=sum(data[n]['diagnostic_nfe'] for n in ('A58','A59','A60')),forward_total=sum(data[n]['method_nfe']+data[n]['diagnostic_nfe'] for n in data),input_VJP_total=sum(data[n]['input_backward_attempts'] for n in data),A58_operator_calls=data['A58']['metrology_operator_calls']),limits='Only retrieved JSON summary/manifest and recomputed arithmetic. No binary tensors/models replayed. A56 user reports incomplete; no A56 result branch found this turn.')
(ROOT/'arithmetic_and_return_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('RETURN_AND_ARITHMETIC_AUDIT_PASS: four uploaded summaries/manifests; blob+SHA256 verified; user excerpts match; A57/A59/A60 arithmetic recomputed; no UNet calls.')
print('NEW COST',result['new_returned_cost'])
print('A59 posthoc forced binary mean versus U',extra['ALL']['GEOMETRY']['cached_forced_binary_gain_vs_U'])
