"""Render REPORT.md itself to a portable CJK PDF; no model calls or extra results."""
import hashlib,html,json,re
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,NextPageTemplate,LongTable,TableStyle,Image,Preformatted
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
OUTPUT=ROOT/'output/pdf/ZAPS_20261027_briefing.pdf';OUTPUT.parent.mkdir(parents=True,exist_ok=True)
FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
pdfmetrics.registerFont(TTFont('BriefingCN',FONT))
pdfmetrics.registerFontFamily('BriefingCN',normal='BriefingCN',bold='BriefingCN',italic='BriefingCN',boldItalic='BriefingCN')
INK=colors.HexColor('#172B3A');BLUE=colors.HexColor('#1966A3');GRAY=colors.HexColor('#64748B');LIGHT=colors.HexColor('#EDF3F8')
body=ParagraphStyle('body',fontName='BriefingCN',fontSize=10.2,leading=15.7,textColor=INK,spaceAfter=6,wordWrap='CJK',splitLongWords=True)
h1=ParagraphStyle('title',parent=body,fontSize=19,leading=26,spaceAfter=15)
h2=ParagraphStyle('section',parent=body,fontSize=13.3,leading=20,spaceBefore=10,spaceAfter=7,keepWithNext=True,textColor=BLUE)
small=ParagraphStyle('small',parent=body,fontSize=9,leading=13.0,spaceAfter=4.5)
caption=ParagraphStyle('caption',parent=small,keepWithNext=True)
cell=ParagraphStyle('cell',parent=small,fontSize=8.5,leading=12.2,spaceAfter=0)
header=ParagraphStyle('thead',parent=cell,fontSize=8.7,textColor=colors.white)
code=ParagraphStyle('code',parent=body,fontSize=9.4,leading=14,leftIndent=10,rightIndent=10,spaceBefore=5,spaceAfter=9,backColor=LIGHT,borderPadding=8)


def inline(raw):
    escaped=html.escape(raw)
    def link(m):
        label,url=m.group(1),html.unescape(m.group(2))
        if not url.startswith(('https://','http://')):
            target=(ROOT/url).resolve()
            if target.is_relative_to(REPO):url='https://github.com/lzy75/ZAPS/blob/master/'+str(target.relative_to(REPO))
        return '<link href="'+html.escape(url,quote=True)+'" color="#1966A3">'+label+'</link>'
    escaped=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,escaped)
    escaped=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',escaped)
    escaped=re.sub(r'`([^`]+)`',r'<font color="#35526A">\1</font>',escaped)
    return escaped


def make_table(lines,width):
    data=[]
    for i,line in enumerate(lines):
        values=[v.strip() for v in line.strip().strip('|').split('|')]
        if i==1 and all(re.fullmatch(r':?-+:?',v) for v in values):continue
        data.append([Paragraph(inline(v),header if i==0 else cell) for v in values])
    n=len(data[0]);assert all(len(r)==n for r in data),'Malformed Markdown table';first=re.sub('<[^>]+>','',lines[0])
    if n==2:ratios=[.22,.78]
    elif n==3:
        ratios=[.15,.43,.42] if '日期' in first else [.23,.36,.41]
    elif n==4:ratios=[.20,.29,.18,.33]
    elif n==5:
        ratios=[.19,.30,.13,.13,.25] if '新增前向' in first else [.22,.195,.195,.195,.195]
    elif n==6:ratios=[.21,.13,.17,.17,.17,.15]
    else:ratios=[1/n]*n
    t=LongTable(data,colWidths=[width*r/sum(ratios) for r in ratios],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5.5),('BOTTOMPADDING',(0,0),(-1,-1),5.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('LINEBELOW',(0,0),(-1,0),.5,BLUE),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D9E3EC'))]))
    return t


def page(canvas,doc):
    w,h=canvas._pagesize;canvas.saveState();canvas.setFont('BriefingCN',8.1);canvas.setFillColor(GRAY)
    canvas.drawString(40,h-23,'ZAPS 状态感知引导研究 | 2026-10-27')
    canvas.drawRightString(w-40,h-23,'模型证据截至A55r2；A56-A60待运行')
    canvas.setStrokeColor(colors.HexColor('#D9E3EC'));canvas.line(40,h-29,w-40,h-29)
    canvas.drawString(40,20,'固定原30点；步长/采样位置暂停。')
    canvas.drawRightString(w-40,20,str(doc.page));canvas.restoreState()

source=ROOT/'REPORT.md';raw=source.read_text()
font=pdfmetrics.getFont('BriefingCN');missing=sorted({ord(c) for c in raw if ord(c)>31 and ord(c) not in font.face.charToGlyph})
if missing:raise ValueError('Missing CJK/math glyphs in report: '+','.join(f'U+{c:04X}' for c in missing))
portrait=A4;wide=landscape(A4)
doc=BaseDocTemplate(str(OUTPUT),pagesize=portrait,leftMargin=40,rightMargin=40,topMargin=43,bottomMargin=36,
    title='状态感知引导：动机、证据与27日上午关键验证',author='ZAPS research briefing',subject='Fixed-grid state guidance evidence and bounded experiments')
doc.addPageTemplates([PageTemplate(id='portrait',pagesize=portrait,frames=[Frame(40,36,portrait[0]-80,portrait[1]-79,id='p',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page),
    PageTemplate(id='landscape',pagesize=wide,frames=[Frame(35,35,wide[0]-70,wide[1]-78,id='l',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page)])
lines=raw.splitlines();story=[];i=0;mode='portrait';image_records=[]
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('<!-- page:'):
        new=line.split(':',1)[1].split('-->')[0].strip()
        if new!=mode or new=='landscape':story.extend([NextPageTemplate(new),PageBreak()])
        mode=new;i+=1;continue
    if line.startswith('# '):story.append(Paragraph(inline(line[2:]),h1));i+=1;continue
    if line.startswith('## '):story.append(Paragraph(inline(line[3:]),h2));i+=1;continue
    if line.startswith('```'):
        code_lines=[];i+=1
        while i<len(lines) and not lines[i].strip().startswith('```'):code_lines.append(lines[i]);i+=1
        story.append(Preformatted('\n'.join(code_lines),code));i+=1;continue
    if line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].strip().startswith('|'):table.append(lines[i].strip());i+=1
        story.extend([make_table(table,(wide[0]-70) if mode=='landscape' else (portrait[0]-80)),Spacer(1,9)]);continue
    match=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
    if match:
        label,path=match.groups();f=(ROOT/path).resolve();w,h=PILImage.open(f).size
        maxw=wide[0]-90 if mode=='landscape' else portrait[0]-80
        maxh=405 if mode=='landscape' else 360
        scale=min(maxw/w,maxh/h);story.append(Paragraph(inline(label),caption));story.append(Image(str(f),width=w*scale,height=h*scale,hAlign='CENTER'));story.append(Spacer(1,7))
        image_records.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));i+=1;continue
    if line.startswith('- '):story.append(Paragraph(inline(line[2:]),small,bulletText='•'));i+=1;continue
    parts=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('# ','## ','|','```','![','<!--','- ')):
        parts.append(lines[i].strip());i+=1
    story.append(Paragraph(inline(' '.join(parts)),body))
doc.build(story)
manifest=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pdf=str(OUTPUT),pdf_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    images=image_records,method='REPORT.md directly parsed; no extra narrative or removed content',missing_glyphs=[])
(ROOT/'output/pdf/render_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PDF CREATED:',OUTPUT)
