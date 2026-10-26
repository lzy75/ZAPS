"""Source-based SVG figures only; no model calls or PDF authoring."""
import hashlib,json,shutil
from pathlib import Path
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Circle,Polygon
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,white

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
BLUE=HexColor('#1966A3');TEAL=HexColor('#168878');ORANGE=HexColor('#C17A24');GRAY=HexColor('#64748B');INK=HexColor('#172B3A');LIGHT=HexColor('#EDF3F8')

def text(d,x,y,s,size=15,color=INK,anchor='start',bold=False):
    d.add(String(x,y,s,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=color,textAnchor=anchor))

def arrow(d,x1,y1,x2,y2,color=GRAY):
    d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=2))
    if x2>x1:d.add(Polygon([x2,y2,x2-8,y2-4,x2-8,y2+4],fillColor=color,strokeColor=color))
    else:d.add(Polygon([x2,y2,x2-4,y2+8,x2+4,y2+8],fillColor=color,strokeColor=color))

def box(d,x,y,w,h,title,body,color=BLUE):
    d.add(Rect(x,y,w,h,rx=9,ry=9,fillColor=LIGHT,strokeColor=color,strokeWidth=1.5))
    text(d,x+w/2,y+h-26,title,18,color,'middle',True)
    for j,s in enumerate(body):text(d,x+w/2,y+h-51-j*21,s,14,INK,'middle')

m=Drawing(1040,395);m.add(Rect(0,0,1040,395,fillColor=white,strokeColor=white))
text(m,18,366,'From a scalar heuristic to a current physical target and response direction',22,bold=True)
box(m,15,205,220,125,'Current observation',['x_t -> raw m_t','r = y - A(m_t)','known H, sigma_y'],GRAY)
box(m,290,253,225,86,'Physical target',['v = L^T S^(-1) r'],BLUE)
box(m,290,126,225,88,'Denoiser response',['u = J_m(x_t)^T v'],TEAL)
box(m,585,192,235,120,'Guidance displacement',['direction: u / ||u||','length: K(t,s) ||v||','T = K ||v|| u / ||u||'],BLUE)
box(m,875,205,150,97,'Original grid',['DDPM + T','same 30 points'],GRAY)
arrow(m,235,288,290,288);arrow(m,402,253,402,214,TEAL)
arrow(m,515,289,585,271,BLUE);arrow(m,515,167,585,222,TEAL);arrow(m,820,252,875,252,BLUE)
text(m,20,83,'Existing evidence: U direction beats V/Q on 5 OLD images in both SR and inpainting.',17,TEAL,bold=True)
text(m,20,54,'Not established: current scalar gain > time/shuffle; fresh-image transfer; curvature control.',16,GRAY)
text(m,20,25,'A59 tests fixed-time response; A60 tests direction reuse. No new GPU result is implied.',15,GRAY)
renderSVG.drawToFile(m,str(OUT/'motivation.svg'))

p55=REPO/'projects/reports/20261026_a55r2_readout/user_summary.json';p54=REPO/'projects/reports/20261026_a54_readout/user_summary.json'
a55=json.loads(p55.read_text());a54=json.loads(p54.read_text());payload=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (p55,p54)},panels=[])
d=Drawing(1000,720);d.add(Rect(0,0,1000,720,fillColor=white,strokeColor=white));text(d,25,687,'Direction and scalar gain are different experimental questions',23,bold=True)
text(d,25,661,'All 5 reused FFHQ images; seed1000, unit sigma=.05. Positive means current U is better.',15,GRAY)

def panel(x,y,w,h,title,series,ids):
    text(d,x,y+h+11,title,18,bold=True)
    allvalues=[z for name,vs,color in series for z in vs]+[0.]
    low,high=min(allvalues),max(allvalues);pad=max((high-low)*.18,.035);low-=pad;high+=pad
    left,bottom=x+55,y+42;right,top=x+w-12,y+h-35
    sx=lambda i:left+(right-left)*i/4;sy=lambda z:bottom+(top-bottom)*(z-low)/(high-low)
    for j in range(5):
        value=low+(high-low)*j/4;yy=sy(value);d.add(Line(left,yy,right,yy,strokeColor=HexColor('#E4EAF0'),strokeWidth=.7));text(d,left-8,yy-4,f'{value:.2f}',12,GRAY,'end')
    d.add(Line(left,sy(0),right,sy(0),strokeColor=GRAY,strokeDashArray=[4,3]))
    d.add(Line(left,bottom,left,top,strokeColor=GRAY));d.add(Line(left,bottom,right,bottom,strokeColor=GRAY))
    for i,image in enumerate(ids):text(d,sx(i),bottom-19,image,12,GRAY,'middle')
    for index,(name,vals,color) in enumerate(series):
        for i in range(4):d.add(Line(sx(i),sy(vals[i]),sx(i+1),sy(vals[i+1]),strokeColor=color,strokeWidth=2))
        for i,z in enumerate(vals):d.add(Circle(sx(i),sy(z),3.8,fillColor=color,strokeColor=color))
        text(d,x+55+index*190,y+h-17,name,13,color)
    text(d,x+55,y+5,'PSNR difference (dB); independent panel scales',12,GRAY)

for taskindex,task in enumerate(('super_resolution','inpainting')):
    group=next(g for g in a55['groups'] if g['task']==task);rows=group['per_image'];ids=[r['case']['image_id'] for r in rows]
    direction=[('U - V',[r['metrics']['U_CURRENT']['psnr']-r['metrics']['V_PHYSICAL']['psnr'] for r in rows],BLUE),
               ('U - Q',[r['metrics']['U_CURRENT']['psnr']-r['metrics']['Q_ORIGINAL']['psnr'] for r in rows],TEAL)]
    scalar=[('U - TIME',[r['metrics']['U_CURRENT']['psnr']-r['metrics']['TIME_GAIN']['psnr'] for r in rows],GRAY),
            ('U - SHUFFLE',[r['metrics']['U_CURRENT']['psnr']-r['metrics']['SHUFFLE_GAIN']['psnr'] for r in rows],ORANGE)]
    y=360 if taskindex==0 else 47;name='4x super-resolution' if taskindex==0 else 'Random inpainting'
    panel(12,y,482,253,name+': direction',direction,ids);panel(514,y,472,253,name+': scalar gain',scalar,ids)
    payload['panels'].append(dict(task=task,image_ids=ids,direction={n:v for n,v,c in direction},scalar={n:v for n,v,c in scalar}))
renderSVG.drawToFile(d,str(OUT/'direction_vs_gain.svg'))
(OUT/'figure_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')

records=[]
for task,label in [('super_resolution','sr'),('inpainting','ip')]:
    original=Path('/Users/liziyue.22/ZAPS-deliverables/analysis/A53_20261026/archive')/f'ffhq_{task}_00446_s1000/angular_comparison.png'
    target=OUT/f'example_{label}_00446.png';shutil.copyfile(original,target)
    records.append(dict(source=str(original),copy=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        transformed=False,cohort='OLD development',selection='Prespecified smallest image ID00446, not best-gain selection',
        caption='A53 original panel. NORM_ONLY equals cached U reused by A54/A55r2; STATE retains cosine. Not A55r2 V/Q pictures.'))
(OUT/'image_provenance.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print('Wrote2 SVG data figures and2 byte-preserving original A53 panels; no PDF/model calls.')
