"""Read existing audited summaries; create conference-style tables, never run models."""
from pathlib import Path
import csv, hashlib, json, statistics

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
METRICS=('psnr','ssim','lpips')
TASKS={'super_resolution':'四倍超分辨率','inpainting':'随机遮挡修复','gaussian_deblur':'高斯去模糊','motion_deblur':'运动去模糊'}
NAMES={'C_REPLAY':'本地零样本自适应采样（原精度）','C_IEEE':'本地零样本自适应采样（精度匹配）',
 'Q_ORIGINAL':'保留原引导方向，匹配物理长度','V_PHYSICAL':'仅物理方向，匹配物理长度',
 'U_CURRENT':'当前响应方向，实时幅度匹配','TIME_GAIN':'当前响应方向，冻结时间幅度',
 'OLD_U':'当前响应方向，预算内求解与回退'}
PAPER={('ffhq','super_resolution'):(26.63,.768,.104),('ffhq','inpainting'):(27.79,.813,.078),
 ('ffhq','gaussian_deblur'):(26.06,.757,.121),('ffhq','motion_deblur'):(24.16,.709,.141),
 ('imagenet','super_resolution'):(23.82,.718,.186),('imagenet','gaussian_deblur'):(22.45,.682,.225)}

def load(rel):return json.loads((REPO/rel).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def val(v,k):return '—' if v is None else f'{v:.3f}'
def ranks(rows):
    result={}
    for k in METRICS:
        # Rank distinct displayed values; equal values at publication precision share a rank.
        values=sorted({round(r[k],3) for r in rows if r[k] is not None},reverse=k!='lpips')
        result[k]={v:i+1 for i,v in enumerate(values)}
    return result
def fmt(v,k,rank=None,latex=False):
    s=val(v,k)
    if rank and v is not None:
        p=rank[k].get(round(v,3))
        if p==1:s=('\\textbf{'+s+'}') if latex else '**'+s+'**'
        elif p==2:s=('\\underline{'+s+'}') if latex else '<u>'+s+'</u>'
    return s
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    raw=load('projects/reports/20261028_method_ablation/aggregated.json')
    lookup={(r['dataset'],r['task'],r['arm']):r for r in raw['rows']}
    results=[];mainrows=[];maintex=[];ablations=[];ablationrows=[];ablationtex=[]
    for d in ('ffhq','imagenet'):
        for t in TASKS:
            paper=PAPER.get((d,t));base=lookup.get((d,t,'C_REPLAY' if t in ('super_resolution','inpainting') else 'C_IEEE'))
            candidate=lookup.get((d,t,'U_CURRENT' if t in ('super_resolution','inpainting') else 'OLD_U'))
            rank=ranks([r for r in (base,candidate) if r])
            values=[];tx=[]
            for r,canrank in ((dict(zip(METRICS,paper)) if paper else None,False),(base,True),(candidate,True)):
                values += [fmt(r[k] if r else None,k,rank if canrank else None) for k in METRICS]
                tx += [fmt(r[k] if r else None,k,rank if canrank else None,True) for k in METRICS]
            group=f'{d.upper()} / {TASKS[t]}'
            mainrows.append([group]+values);maintex.append([group]+tx)
            for name,r,n in [('原论文结果',dict(zip(METRICS,paper)) if paper else None,1000 if paper else None),('本地探索网格基线',base,base['images'] if base else None),('当前重点候选：预算内求解',candidate,candidate['images'] if candidate else None)]:
                results.append(dict(dataset=d,task=TASKS[t],method=name,images=n,
                  **{k:r[k] if r else None for k in METRICS},complete_reconstruction_wall_seconds=None,
                  time_status='原文仅单列总体14.71秒，未按任务报告' if name=='原论文结果' else '同口径完整重建计时未完成',
                  cohort='原文验证集，与本地不配对' if name=='原论文结果' else ('未用图像' if t in ('super_resolution','inpainting') else '复用开发图像'),
                  paper_setting_reproduction=False if name!='原论文结果' else None))
            arms=['C_IEEE','Q_ORIGINAL','V_PHYSICAL','U_CURRENT','TIME_GAIN'] if t in ('super_resolution','inpainting') else ['C_IEEE','OLD_U']
            rr=[lookup[(d,t,a)] for a in arms if (d,t,a) in lookup];rank=ranks(rr)
            for a in arms:
                r=lookup.get((d,t,a))
                if r is None:continue
                name=NAMES[a];name+='（预算内求解）' if a=='U_CURRENT' else ''
                ablations.append(dict(dataset=d,task=TASKS[t],method=name,images=r['images'],**{k:r[k] for k in METRICS},
                    complete_reconstruction_wall_seconds=None,last_unroll_replay_seconds=r['mean_last30_seconds'],
                    input_backward_per_last30=r['input_VJPs_per_last30'],scope='末轮回放时间，不是完整重建时间'))
                ablationrows.append([group,name]+[fmt(r[k],k,rank) for k in METRICS]+['未测',f"{r['mean_last30_seconds']:.2f}"])
                ablationtex.append([group,name]+[fmt(r[k],k,rank,True) for k in METRICS]+['未测',f"{r['mean_last30_seconds']:.2f}"])
    headers=['数据集 / 任务','原文峰值信噪比','原文结构相似度','原文感知距离','本地峰值信噪比','本地结构相似度','本地感知距离','当前峰值信噪比','当前结构相似度','当前感知距离']
    main_md=table(headers,mainrows)
    abl_md=table(['数据集 / 任务','方法或消融','峰值信噪比 ↑','结构相似度 ↑','感知距离 ↓','完整重建秒 ↓','末轮回放秒'],ablationrows)
    (HERE/'MAIN_TABLE.md').write_text(main_md+'\n');(HERE/'ABLATION_TABLE.md').write_text(abl_md+'\n')
    for name,rr in [('main_comparison.csv',results),('module_ablations.csv',ablations)]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
    original=load('projects/reports/20261027_a56_ffhq_received/A56_plan.json');g=original['grid']
    grid_audit=dict(declared_schedule=original['configs']['ffhq']['super_resolution']['schedule'],
       executed_grid=g,executed_thirds_counts=[sum(lo<=v<=hi for v in g) for lo,hi in [(0,333),(334,666),(667,999)]],
       grid_overridden_at='utils/diag_multitask_state_transfer.py::training',
       warning='实际执行网格不等于原文三段15/10/5；本地消融有效，不能当原文配置复现')
    assert grid_audit['executed_thirds_counts']==[6,5,19]
    old=load('projects/reports/20261026_a49_a50r2/A53_cached_audit.json')
    angle=[]
    for g in old['groups']:
        rr=[dict(g['mean_metrics'][a],name=n) for a,n in [('STATE','当前响应方向，增加余弦衰减'),('NORM_ONLY','当前响应方向，不增加余弦衰减')]];rank=ranks(rr)
        for r in rr:angle.append([TASKS[g['task']],r['name']]+[fmt(r[k],k,rank) for k in METRICS])
    (HERE/'ANGLE_ABLATION.md').write_text(table(['任务：五张旧图','幅度规则','峰值信噪比 ↑','结构相似度 ↑','感知距离 ↓'],angle)+'\n')
    lag=load('projects/reports/20261027_a57_a60_readout/received/A60_summary.json')
    lagrows=[]
    for g in lag['groups']:
        for arm,desc in [('LAG2','每两步刷新方向'),('LAG4','每四步刷新方向')]:
            m=g['arms'][arm]['mean_delta']['CURRENT'];lagrows.append([TASKS[g['task']],desc]+[f'{m[k]:+.3f}' for k in METRICS])
    (HERE/'REFRESH_ABLATION.md').write_text(table(['任务：五张旧图','改动','峰值信噪比变化','结构相似度变化','感知距离变化'],lagrows)+'\n')
    # booktabs output is an includable fragment for a parent document supporting Chinese.
    tex=['% Requires booktabs, graphicx and a parent document with CJK support.','% Literature rows are contextual: no ranking across different cohorts/hardware.']
    for label,hs,rs in [('main-comparison',headers,maintex),('module-ablations',['数据集/任务','方法','峰值信噪比','结构相似度','感知距离','完整秒','回放秒'],ablationtex)]:
        left=1 if label=='main-comparison' else 2
        tex += ['\\begin{table*}[t]','\\centering','\\scriptsize','\\setlength{\\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%', '\\begin{tabular}{'+'l'*left+'r'*(len(hs)-left)+'}',r'\toprule',' & '.join(hs)+r' \\',r'\midrule']
        tex += [' & '.join(row).replace('—','--')+r' \\' for row in rs]
        tex += [r'\bottomrule',r'\end{tabular}}',r'\caption{同任务同协议的本地结果中，加粗为最优，下划线为次优；原文结果只作背景。完整重建计时未完成；回放时间不用于总体加速结论。}',f'\\label{{tab:{label}}}',r'\end{table*}','']
    (HERE/'conference_tables.tex').write_text('\n'.join(tex))
    paths=['projects/reports/20261028_method_ablation/aggregated.json','projects/reports/20261027_a56_ffhq_received/A56_plan.json','projects/reports/20261026_a49_a50r2/A53_cached_audit.json','projects/reports/20261027_a57_a60_readout/received/A60_summary.json','projects/reports/20261027_a57_a60_readout/received/A57_summary.json']
    data=dict(main_rows=results,ablation_rows=ablations,grid_audit=grid_audit,paper_rows=[dict(dataset=d,task=t,psnr=v[0],ssim=v[1],lpips=v[2]) for (d,t),v in PAPER.items()],
        paper_time=dict(seconds=14.71,network_forward_calls=300,hardware='NVIDIA A100',scope='原论文主文表3单列总体时间；没有逐数据集逐任务时间，不与本地3090混排'),
        local_inputs=[dict(path=n,sha256=sha(REPO/n)) for n in paths],
        ranking='按展示的三位小数同分并列；只排同任务同样本本地质量，不排文献或缺失时间',
        missing=['公开原文设置下的同队列复现','全部方法统一起止点的完整计时','移除物理逆作用的独立消融','ImageNet两项去模糊新方法结果'],
        no_model_execution=True)
    source_pdf=Path('/Users/liziyue.22/Desktop/论文文档/已读论文/ZAPS.pdf')
    if source_pdf.exists():
        data['paper_source']=dict(path=str(source_pdf),sha256=sha(source_pdf),complete_pages_visually_checked=[11,12,20],
          paper_tables=['主文表1','主文表2','主文表3','补充材料表1'],official_supplement='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114-supp.pdf',
          official_supplement_hardware='NVIDIA A100',missing_time='14.71秒未分任务报告')
    if (HERE/'REPORT.md').exists():data['report_sha256']=sha(HERE/'REPORT.md')
    (HERE/'evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('PASS: eight dataset/task contexts, 24 main rows, 24 budget-limited local ablation rows; no new model execution.')

if __name__=='__main__':main()
