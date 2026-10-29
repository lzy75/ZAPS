"""CPU scientific figures from saved real images/metrics; no model calls."""
import csv
import html
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
TASK_NAMES={'super_resolution':'Super-resolution x4','gaussian_deblur':'Gaussian deblur','inpainting':'Random inpainting','motion_deblur':'Motion deblur'}
METHODS={'BASE':'ZAPS (local)','SHRINK':'Noise-aware shrinkage'}
FONT_CANDIDATES=('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf')
BOLD_CANDIDATES=('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/System/Library/Fonts/Supplemental/Arial Bold.ttf')

def font(size,bold=False):
    for f in BOLD_CANDIDATES if bold else FONT_CANDIDATES:
        if Path(f).exists():return ImageFont.truetype(f,size)
    return ImageFont.load_default()

def three_line(groups,path,title,notes=()):
    """Each group has label and rows(name,psnr,ssim,lpips,seconds,nfe)."""
    width=1480;rh=49;h=190+sum(len(g['rows'])*rh+24 for g in groups)+len(notes)*27
    im=Image.new('RGB',(width,h),'white');draw=ImageDraw.Draw(im);svg=[]
    def txt(x,y,value,size=23,bold=False,right=False,underline=False):
        value=str(value);face=font(size,bold);box=draw.textbbox((0,0),value,font=face);w=box[2]-box[0]
        if right:x-=w
        draw.text((x,y),value,fill='black',font=face)
        svg.append(f'<text x="{x}" y="{y+size}" font-family="serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}">{html.escape(value)}</text>')
        if underline:
            draw.line((x,y+size+4,x+w,y+size+4),fill='black',width=1);svg.append(f'<path d="M{x} {y+size+4}h{w}" stroke="black"/>')
    def line(y,thick):
        draw.line((40,y,width-40,y),fill='black',width=thick);svg.append(f'<path d="M40 {y}H{width-40}" stroke="black" stroke-width="{thick}"/>')
    txt(40,22,title,31,True);line(83,3)
    heads=[(40,'Task / noise'),(350,'Method'),(880,'PSNR ↑'),(1020,'SSIM ↑'),(1160,'LPIPS ↓'),(1320,'Time (s) ↓'),(1430,'NFE')]
    for x,v in heads:txt(x,103,v,22,right=x>=880)
    line(143,1);y=164;data=[]
    for g in groups:
        values={k:sorted({round(r[k],3 if k=='psnr' else 4 if k in ('ssim','lpips') else 2) for r in g['rows'] if r.get(k) is not None},reverse=k in ('psnr','ssim')) for k in ('psnr','ssim','lpips','seconds')}
        for j,r in enumerate(g['rows']):
            if j==0:txt(40,y,g['label'],21)
            txt(350,y,r['name'],22)
            for x,k in ((880,'psnr'),(1020,'ssim'),(1160,'lpips'),(1320,'seconds')):
                digits=3 if k=='psnr' else 4 if k in ('ssim','lpips') else 2;v=r.get(k)
                rank=values[k].index(round(v,digits))+1 if v is not None else None
                txt(x,y,'--' if v is None else f'{v:.{digits}f}',23,rank==1,True,rank==2)
            txt(1430,y,str(r.get('nfe','--')),22,right=True);data.append(dict(group=g['label'],**r));y+=rh
        y+=24
    line(y-8,3)
    for note in notes:txt(40,y+12,note,18);y+=27
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
    path.with_suffix('.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}">'+''.join(svg)+'</svg>')
    with path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['group']);writer.writeheader();writer.writerows(data)
    tex=['% Requires booktabs. No vertical rules.','\\begin{tabular}{llrrrrr}','\\toprule','Task / noise & Method & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & Time (s) & NFE \\\\','\\midrule']
    for g in groups:
        for j,r in enumerate(g['rows']):
            fields=[g['label'] if j==0 else '',r['name']]
            for k in ('psnr','ssim','lpips','seconds'):
                d=3 if k=='psnr' else 4 if k in ('ssim','lpips') else 2
                vals=sorted({round(q[k],d) for q in g['rows'] if q.get(k) is not None},reverse=k in ('psnr','ssim'))
                v=r.get(k);s='--' if v is None else f'{v:.{d}f}'
                if v is not None:
                    rank=vals.index(round(v,d));s=('\\textbf{'+s+'}') if rank==0 else ('\\underline{'+s+'}') if rank==1 else s
                fields.append(s)
            fields.append(str(r.get('nfe','--')));tex.append(' & '.join(fields)+' \\\\')
    tex+=['\\bottomrule','\\end{tabular}'];path.with_suffix('.tex').write_text('\n'.join(tex)+'\n')
    return path

def qualitative(rows,path,title,columns,roi=(80,64,176,160),note='Same crop and display for every method; real saved outputs; no enhancement.'):
    """rows: label, images dict, captions dict. columns: (key,label)."""
    cell=256;gap=18;left=42;top=110;rowh=438;w=left*2+len(columns)*(cell+gap)-gap;h=top+len(rows)*rowh+65
    im=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(im)
    draw.text((left,18),title,fill='black',font=font(24,True))
    for j,(key,label) in enumerate(columns):draw.text((left+j*(cell+gap),65),label,fill='black',font=font(20,True))
    for i,row in enumerate(rows):
        y=top+i*rowh;draw.text((left,y),row['label'],fill='black',font=font(18))
        for j,(key,label) in enumerate(columns):
            x=left+j*(cell+gap);source=Image.open(row['images'][key]).convert('RGB')
            if source.size!=(256,256):raise ValueError('Expected actual256 PNG; no implicit content change')
            im.paste(source,(x,y+28));draw.rectangle((x+roi[0],y+28+roi[1],x+roi[2],y+28+roi[3]),outline='#c22',width=1)
            crop=source.crop(roi).resize((128,128),Image.Resampling.NEAREST);im.paste(crop,(x,y+294));draw.rectangle((x,y+294,x+127,y+421),outline='#c22',width=1)
            caption=row.get('captions',{}).get(key,'')
            if caption:draw.text((x+136,y+307),caption,fill='black',font=font(13))
    draw.text((left,h-42),note,fill='black',font=font(14));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);im.save(path);return path

def render(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());plan=json.loads((root/'plan.json').read_text());out=root/'report';out.mkdir(exist_ok=True)
    for sigma in sorted({g['sigma'] for g in summary['groups']}):
        groups=[]
        for task in TASK_NAMES:
            condition=[g for g in summary['groups'] if g['task']==task and g['sigma']==sigma]
            expected=len({c['image_id'] for c in plan['cases']})
            if len(condition)!=2 or any(g['cases']!=expected for g in condition):continue
            rows=[]
            for arm in ('BASE','SHRINK'):
                selected=[g for g in summary['groups'] if g['task']==task and g['sigma']==sigma and g['arm']==arm]
                if selected:
                    g=selected[0];rows.append(dict(name=METHODS[arm]+f" (n={g['cases']})",**g['quality'],seconds=g['mean_full_seconds'],nfe=300))
            if rows:groups.append(dict(label=TASK_NAMES[task],rows=rows))
        three_line(groups,out/f'table_sigma{round(sigma*1000):03d}.png',f'FFHQ: matched reconstruction quality and full time | sigma={sigma:g}',
            ('Bold: best; underline: second. Rankings within the same task/noise only.',
             'Three fixed images. IEEE; final-unroll control; 300 NN forwards, no input backward.',
             'SR at sigma .05/.10 reuses A70 outputs/times; other cells are measured in A71.'))
        for task in TASK_NAMES:
            rows=[]
            for c in plan['cases']:
                if c['task']!=task or c['sigma']!=sigma:continue
                files={arm:root/'cases'/c['key']/arm/'run.json' for arm in ('BASE','SHRINK')}
                if not all(f.exists() for f in files.values()):continue
                records={a:json.loads(f.read_text()) for a,f in files.items()}
                if any(r['status']!='complete' for r in records.values()):continue
                obs=root/'observations'/c['key']
                images={'GT':obs/'ground_truth.png','OBS':obs/'observation.png',**{a:files[a].parent/'paired.png' for a in files}}
                captions={a:f"PSNR {records[a]['paired_metrics']['psnr']:.2f}\nSSIM {records[a]['paired_metrics']['ssim']:.3f}\nLPIPS {records[a]['paired_metrics']['lpips']:.3f}" for a in files}
                rows.append(dict(label='Image '+c['image_id'],images=images,captions=captions))
            if rows:qualitative(rows,out/f'qual_{task}_sigma{round(sigma*1000):03d}.png',TASK_NAMES[task]+f' | sigma={sigma:g}',(('GT','Ground truth'),('OBS','Observation'),('BASE','ZAPS (local)'),('SHRINK','Noise-aware shrinkage')))
    print('A71 TABLES_AND_REAL_FIGURES_READY:',out,flush=True)
