"""CPU-only early report of all twelve frozen FFHQ cases; zero new UNet calls."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import a56_selection_r2 as frozen
from utils.a54_session import digest,exclusive,read,write

ROOT=frozen.ROOT
OUT=Path('/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/ffhq_readout_20261027')
BRANCH='results/a56-ffhq-early-readout-20261027'
ARMS=frozen.original.rules.ARMS
METRICS=('psnr','ssim','lpips')
REFS=('C_REPLAY','C_IEEE','V_PHYSICAL','Q_ORIGINAL','TIME_GAIN')
TASKS=frozen.original.rules.TASKS
CARD=PROJECTS/'experiment_cards/A56_FFHQ_early_readout_20261027.md'


def sources():
    return {str(p):digest(p) for p in (Path(__file__),CARD,PROJECTS/'utils/test_a56_ffhq_readout.py',PROJECTS/'utils/metrics.py')}


def validate_rows(rows):
    if len(rows)!=12 or any(r['case']['dataset']!='ffhq' for r in rows):
        raise ValueError('Need ALL twelve FFHQ cases; no completed-subset selection')
    ids={}
    for task in TASKS:
        group=[r for r in rows if r['case']['task']==task]
        ids[task]={r['case']['image_id'] for r in group}
        if len(group)!=6 or len(ids[task])!=6:
            raise ValueError('Need six unique frozen FFHQ images per task')
    if ids[TASKS[0]]!=ids[TASKS[1]] or len({r['case']['key'] for r in rows})!=12:
        raise ValueError('Paired task/cohort identities differ')
    for r in rows:
        if set(r['metrics'])!=set(ARMS) or any(not math.isfinite(float(r['metrics'][a][m])) for a in ARMS for m in METRICS):
            raise ValueError('Six finite metric arms required')


def aggregate(rows):
    validate_rows(rows)
    result=[]
    for task in TASKS:
        selected=[r for r in rows if r['case']['task']==task]
        effects={ref:{m:mean(r['metrics']['U_CURRENT'][m]-r['metrics'][ref][m] for r in selected) for m in METRICS} for ref in REFS}
        wins={ref:sum(r['metrics']['U_CURRENT']['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in selected) for ref in REFS}
        resolved=all(r['resolved'][a]==30 for r in selected for a in ('U_CURRENT','V_PHYSICAL','Q_ORIGINAL','TIME_GAIN'))
        def passed(ref):
            d=effects[ref]
            return bool(resolved and d['psnr']>=.05 and d['ssim']>=-.002 and d['lpips']<=.005 and wins[ref]>=4)
        result.append(dict(task=task,image_clusters=6,fully_resolved=bool(resolved),
            arm_means={a:{m:mean(r['metrics'][a][m] for r in selected) for m in METRICS} for a in ARMS},
            U_minus=effects,U_positive_images=wins,
            worst_image_U_minus={ref:{m:min(r['metrics']['U_CURRENT'][m]-r['metrics'][ref][m] for r in selected) if m!='lpips' else max(r['metrics']['U_CURRENT'][m]-r['metrics'][ref][m] for r in selected) for m in METRICS} for ref in REFS},
            quality_transfer_candidate=passed('C_IEEE'),
            pullback_direction_transfer_candidate=bool(passed('V_PHYSICAL') and passed('Q_ORIGINAL')),
            per_image=selected))
    return result


def inputs():
    d=frozen.original;p=d.check(ROOT)
    cases=[c for c in p['cases'] if c['dataset']=='ffhq']
    if len(cases)!=12:
        raise ValueError('Frozen FFHQ cohort must contain exactly12 cases')
    files={str(ROOT/'plan.json'):digest(ROOT/'plan.json')};items=[]
    queue_path=ROOT/'dataset_queue_execution'/'manifest.json'
    queue=read(queue_path);queue_sha=digest(queue_path)
    if queue['plan_sha256']!=files[str(ROOT/'plan.json')]:raise ValueError('Queue/plan provenance differs')
    for path,sha in {**queue['queue_sources'],**queue['snapshots']}.items():
        if digest(path)!=sha:raise ValueError('Bound queue source/snapshot changed: '+path)
    files[str(queue_path)]=queue_sha
    for c in cases:
        found=d.finished(ROOT,p,c)
        if found is None:raise ValueError('FFHQ not complete, no early subset: '+c['key'])
        f,r=found;b,br=d.baseline_done(ROOT,p,c,'BASELINE')
        if any(row.get('queue_execution',{}).get('manifest_sha256')!=queue_sha for row in (r,br)):
            raise ValueError('Completed FFHQ record queue provenance differs')
        for path in (f,b,Path(c['image'])):
            files[str(path)]=digest(path)
        for name,sha in r['artifacts'].items():files[str(f.parent/name)]=sha
        items.append((c,f,r,br))
    return p,items,files


def evaluate(items,files):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    target=OUT/'evaluation.json'
    own=sources()
    if target.exists():
        v=frozen.original.source.load_signed(target)
        if v['inputs']!=files or v['report_sources']!=own:
            raise ValueError('Cached FFHQ evaluation source/input mismatch')
        validate_rows(v['rows']);return v
    ledger=ROOT/'early_FFHQ_evaluation.json'
    timing=dict(kind='USER_AUTHORIZED_EARLY_FFHQ_EVALUATION',source_plan_sha256=files[str(ROOT/'plan.json')],
        all12_FFHQ_outputs_frozen_before_labels=True,all20_blinded_protocol_not_claimed=True,
        ImageNet_confirmation_not_complete_at_user_request=True,report_path=str(OUT),report_sources=own,
        note='User requested FFHQ results before blocked ImageNet. The historical all20 evaluation flag must be read with this amendment; no method/cohort retuning is authorized by this readout.')
    if ledger.exists():
        if frozen.original.source.load_signed(ledger)!=timing:raise ValueError('Early evaluation timing ledger differs')
    else:frozen.original.source.signed(ledger,timing)
    torch.set_num_threads(2);rows=[];start=time.perf_counter()
    with torch.no_grad():
        for c,f,r,br in items:
            gt=load_image_as_tensor(c['image']).cpu()
            bank=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
            if set(bank)!=set(ARMS) or any(tuple(x.shape)!=(1,3,256,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) for x in bank.values()):
                raise ValueError('Saved output tensor contract differs')
            metrics={a:compute_all_metrics(bank[a],gt,lpips_net='alex') for a in ARMS}
            rows.append(dict(case=c,metrics=metrics,resolved={a:sum(t['readout'].get('resolved',False) for t in r['traces'][a]) for a in ('U_CURRENT','V_PHYSICAL','Q_ORIGINAL','TIME_GAIN')}))
            print('FFHQ EVALUATED:',c['key'],len(rows),'/12; CPU metrics only, new UNet=0.',flush=True)
    validate_rows(rows)
    v=dict(inputs=files,report_sources=own,rows=rows,metric_seconds=time.perf_counter()-start,
        metric_convention='Original utils.metrics; uint8 PSNR/SSIM and clamped[-1,1] Alex LPIPS',
        timing_amendment=timing,new_UNet_forward=0,new_input_VJP=0,new_LPIPS_evaluations=72)
    frozen.original.source.signed(target,v)
    return v


def panels(items,evaluation):
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    folder=OUT/'panels';folder.mkdir(exist_ok=True)
    lookup={r['case']['key']:r for r in evaluation['rows']}
    for c,f,r,br in items:
        target=folder/(c['key']+'.png')
        if target.exists():continue
        bank=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
        names=('GT','C_IEEE','U_CURRENT','V_PHYSICAL','Q_ORIGINAL','TIME_GAIN')
        sheet=Image.new('RGB',(1668,390),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),c['key']+' | FROZEN FFHQ OUTPUTS | EARLY DATASET REPORT; ImageNet PENDING',fill='black',font=font(14))
        for j,a in enumerate(names):
            left=12+276*j;draw.text((left,40),a,fill='black',font=font(14))
            if a=='GT':im=Image.open(c['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
            else:
                x=bank[a][0].numpy().transpose(1,2,0);im=Image.fromarray(((np.clip(x,-1,1)+1)*127.5).astype(np.uint8))
                m=lookup[c['key']]['metrics'][a]
                draw.text((left,66),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
            sheet.paste(im,(left,96))
        sheet.save(target)


def output(p,items,evaluation):
    gs=aggregate(evaluation['rows'])
    result=dict(experiment='A56_FFHQ_EARLY_READOUT',status='FFHQ_COMPLETE_IMAGENET_NOT_CONFIRMED',
        classification='FFHQ_FROZEN_DIRECTION_TRANSFER_IN_'+str(sum(g['pullback_direction_transfer_candidate'] for g in gs))+'_OF_2_TASKS',
        completed_FFHQ_cases=12,FFHQ_image_clusters=6,full_A56_planned_cases=20,
        source_plan_sha256=digest(ROOT/'plan.json'),method_nfe=sum(br['nfe'] for c,f,r,br in items),
        diagnostic_nfe=sum(r['nfe'] for c,f,r,br in items),input_backward_attempts=sum(r['backward_attempts'] for c,f,r,br in items),
        new_UNet_forward=0,new_input_VJP=0,new_LPIPS_evaluations=72,CPU_metric_seconds=evaluation['metric_seconds'],groups=gs,
        timing_amendment=evaluation['timing_amendment'],
        limits='Six frozen unused-accessible-history FFHQ images, paired SR/IP, one seed, known sigma=.05. Original A56 engineering thresholds unchanged. Dataset-level early evaluation explicitly requested before ImageNet; not the original all20-before-first-label protocol. No four-task, ImageNet, curvature, broad generalization or new NFE-saving claim. Inherited300-forward accounting is not equal VJP/operator/walltime.')
    frozen.original.source.signed(OUT/'summary.json',result)
    with (OUT/'per_image.csv').open('w',newline='') as stream:
        cols=['image_id','task','arm',*METRICS,*['delta_C_IEEE_'+m for m in METRICS]]
        w=csv.DictWriter(stream,fieldnames=cols);w.writeheader()
        for row in evaluation['rows']:
            for a in ARMS:w.writerow(dict(image_id=row['case']['image_id'],task=row['case']['task'],arm=a,**row['metrics'][a],**{'delta_C_IEEE_'+m:row['metrics'][a][m]-row['metrics']['C_IEEE'][m] for m in METRICS}))
    lines=['# A56 FFHQ提前汇总','',result['classification'],'',
        '6张冻结新FFHQ图 × SR/IP；全部12例输出在本次评价前已完成。ImageNet尚不确认。',
        '用户要求提前读FFHQ，本次明确修订评价时间；不再宣称完整20例一直等全部输出后才首次看GT。','',
        'U：当前物理长度K||v||、当前响应方向u/||u||；V：同长度的物理v方向；Q：同长度的原引导q方向；TIME：旧FFHQ冻结时间增益乘当前原q。V/Q/TIME也含当前状态信息，均不是整方法“无状态”。',
        'C_REPLAY/C_IEEE分别为本地原配置的native/IEEE精度参照，不是已认证作者基准。','']
    for g in gs:
        lines += ['## '+g['task'],'', '| 方法 | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|']
        for a,m in g['arm_means'].items():lines.append(f'| {a} | {m["psnr"]:.4f} | {m["ssim"]:.5f} | {m["lpips"]:.5f} |')
        lines+=['','| U相对参照 | ΔPSNR | ΔSSIM | ΔLPIPS | PSNR正向图/6 |','|---|---:|---:|---:|---:|']
        for a,m in g['U_minus'].items():lines.append(f'| {a} | {m["psnr"]:+.4f} | {m["ssim"]:+.5f} | {m["lpips"]:+.5f} | {g["U_positive_images"][a]} |')
        lines+=['',f'四状态路径完全解决：{g["fully_resolved"]}；相对C工程候选：{g["quality_transfer_candidate"]}；相对V与Q方向迁移候选：{g["pullback_direction_transfer_candidate"]}。','']
    lines += ['## 成本与范围','',f'来源计算：{result["method_nfe"]}方法 + {result["diagnostic_nfe"]}诊断前向；输入VJP {result["input_backward_attempts"]}。本次新UNet=0；CPU Alex LPIPS评价72次。','',
        '所有逐图指标见per_image.csv；12张GT/C/U/V/Q/TIME对比图见panels/。原PT与计划未改。后续不得在这些图上调参后仍称独立确认。','',
        'ImageNet后续先做有上限的float32原模型显存计量，分别覆盖原优化和当前方向VJP路径；记录allocated/reserved/空闲显存/耗时/OOM，所有UNet尝试计费。12GiB是工程准入，不是已测最低需求。若当前约10.5GiB可用空间仍不能安全覆盖峰值，再独立验证已有activation checkpoint或CPU保存反向张量，先过实际模型输出/VJP等价门控，再运行同一冻结队列。不得静默换fp16、缩图、换checkpoint或换样本；资源不支持时ImageNet记资源受限未完成，FFHQ不再等待。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    console=dict(result,groups=[{k:v for k,v in g.items() if k!='per_image'} for g in gs])
    print('A56 FFHQ Summary:',json.dumps(console,ensure_ascii=False,allow_nan=False),flush=True)
    print('FFHQ report and panels:',OUT,flush=True)


def main():
    # CPU metric work must never compete with the live GPU queue.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    OUT.mkdir(parents=True,exist_ok=True)
    with exclusive(OUT.parent/'ffhq_readout_20261027.lock'):
        p,items,files=inputs();evaluation=evaluate(items,files)
        for path,sha in files.items():
            if digest(path)!=sha:raise ValueError('Frozen FFHQ input changed while reporting: '+path)
        panels(items,evaluation);output(p,items,evaluation)
        shutil.copyfile(CARD,OUT/'experiment_card.md')
        shutil.copyfile(ROOT/'early_FFHQ_evaluation.json',OUT/'early_FFHQ_evaluation.json')
        snap=OUT/'source_records';snap.mkdir(exist_ok=True)
        for name in ('plan.json','plan.sha256'):
            shutil.copyfile(ROOT/name,snap/('A56_'+name))
        shutil.copyfile(ROOT/'dataset_queue_execution'/'manifest.json',snap/'queue_manifest.json')
        for c,f,r,br in items:
            shutil.copyfile(f,snap/(c['key']+'_sampling.json'))
            shutil.copyfile(f.with_suffix('.sha256'),snap/(c['key']+'_sampling.sha256'))
            b=ROOT/c['key']/'BASELINE'/'attempt_000'/'run.json'
            shutil.copyfile(b,snap/(c['key']+'_baseline.json'))
            shutil.copyfile(b.with_suffix('.sha256'),snap/(c['key']+'_baseline.sha256'))
        write(OUT/'source_inventory.json',dict(original_inputs=files,report_sources=sources(),new_UNet_forward=0))
        from unittest.mock import patch
        from utils import archive_refinement_confirmation as archive
        try:
            with patch.object(archive,'BRANCH',BRANCH):bundle,commit=archive.export_and_commit(dict(FFHQ_REPORT=OUT))
        except (subprocess.SubprocessError,OSError) as error:
            print('RESULTS_ARCHIVE_PENDING:',repr(error),'; FFHQ summary/panels retained at',OUT,'; no GPU rerun.',flush=True)
            return
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
        try:
            subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),timeout=45,check=True)
            print('RESULTS_PUSH_PASS: A56 FFHQ early readout and panels uploaded; no GPU rerun.',flush=True)
        except (subprocess.SubprocessError,OSError) as error:
            print('RESULTS_PUSH_PENDING:',repr(error),'; archive retained; git push origin '+BRANCH,flush=True)


if __name__=='__main__':main()
