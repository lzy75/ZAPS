"""Render the complete briefing and its booktabs-style tables; no model calls."""
import hashlib,html,json,re
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,NextPageTemplate,LongTable,TableStyle,Preformatted,KeepTogether
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
DEST=OUT/'denoiser_response_briefing_20261028.pdf'
pdfmetrics.registerFont(TTFont('CN','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
pdfmetrics.registerFont(TTFont('LatinBold','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN',italic='CN',boldItalic='CN')
INK=colors.HexColor('#18252D');GRAY=colors.HexColor('#59656E')
body=ParagraphStyle('body',fontName='CN',fontSize=10.2,leading=16.3,textColor=INK,spaceAfter=7,wordWrap='CJK')
title=ParagraphStyle('title',parent=body,fontSize=20,leading=28,spaceAfter=16)
h2=ParagraphStyle('h2',parent=body,fontSize=14,leading=21,spaceBefore=13,spaceAfter=8,keepWithNext=True)
h3=ParagraphStyle('h3',parent=body,fontSize=11.5,leading=18,spaceBefore=8,spaceAfter=6,keepWithNext=True)
small=ParagraphStyle('small',parent=body,fontSize=8.7,leading=13.4,spaceAfter=5)
cell=ParagraphStyle('cell',parent=small,fontSize=8.2,leading=12.1,spaceAfter=0)
equation=ParagraphStyle('equation',parent=body,fontSize=10.7,leading=21,spaceBefore=6,spaceAfter=10,leftIndent=9,rightIndent=8)
code=ParagraphStyle('code',parent=body,fontSize=9.1,leading=15.1,leftIndent=8,backColor=colors.HexColor('#F4F5F6'),borderPadding=3,spaceBefore=0,spaceAfter=1)

def inline(raw):
    raw=raw.replace('—','-').replace('–','-').replace('‑','-')
    x=html.escape(raw)
    x=x.replace('&lt;u&gt;','<u>').replace('&lt;/u&gt;','</u>')
    def bold(m):
        s=m.group(1)
        return '<font name="LatinBold">'+s+'</font>' if re.fullmatch(r'[0-9.+% -]+',s) else '<b>'+s+'</b>'
    x=re.sub(r'\*\*(.*?)\*\*',bold,x)
    x=re.sub(r'`([^`]+)`',r'\1',x)
    x=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#234C6A">\1</link>',x)
    return x

MATH=[
 'x<sub>t</sub> = a<sub>t</sub>x<sub>0</sub> + b<sub>t</sub>ε；　m<sub>t</sub>(x<sub>t</sub>) = [x<sub>t</sub> - b<sub>t</sub>ε<sub>θ</sub>(x<sub>t</sub>,t)] / a<sub>t</sub>',
 '𝒜(m) = H((m+1)/2) = Lm+d；　L = H/2；　d = H(1)/2',
 'q<sub>t</sub> = (ζ<sub>t</sub>/a<sub>t</sub>)[I+b<sub>t</sub><super>2</super>WD<sub>t</sub>W<super>T</super>]L<super>T</super>[y-𝒜(clip m<sub>t</sub>)]',
 'r<sub>t</sub> = y-𝒜(m<sub>t</sub>)；　c<sub>t</sub> = b<sub>t</sub><super>2</super>/a<sub>t</sub>；　S<sub>t</sub> = σ<sub>y</sub><super>2</super>I+c<sub>t</sub>LL<super>T</super>',
 'δm<sub>t</sub> = c<sub>t</sub>L<super>T</super>S<sub>t</sub><super>-1</super>r<sub>t</sub> = c<sub>t</sub>v<sub>t</sub>；　v<sub>t</sub> = L<super>T</super>S<sub>t</sub><super>-1</super>r<sub>t</sub>',
 '(LL<super>T</super>+λ<sub>t</sub>I)w<sub>t</sub> = r<sub>t</sub>；　λ<sub>t</sub> = σ<sub>y</sub><super>2</super>/c<sub>t</sub>；　v<sub>t</sub> = L<super>T</super>w<sub>t</sub>/c<sub>t</sub>',
 'J<sub>t</sub> = ∂m<sub>t</sub>/∂x<sub>t</sub>；　u<sub>t</sub> = J<sub>t</sub><super>T</super>v<sub>t</sub>',
 'ℓ<sub>t</sub>(x) = ½r<sub>t</sub>(x)<super>T</super>S<sub>t</sub><super>-1</super>r<sub>t</sub>(x)<br/>-∇<sub>x</sub>ℓ<sub>t</sub>(x<sub>t</sub>) = J<sub>t</sub><super>T</super>L<super>T</super>S<sub>t</sub><super>-1</super>r<sub>t</sub> = u<sub>t</sub>',
 'g<sub>t</sub> = ‖v<sub>t</sub>‖<sub>2</sub>/‖u<sub>t</sub>‖<sub>2</sub>；　Δ<sub>t</sub> = K<sub>t,s</sub>g<sub>t</sub>u<sub>t</sub>',
 'K<sub>t,s</sub> = (1-ᾱ<sub>t</sub>/ᾱ<sub>s</sub>)/√(ᾱ<sub>t</sub>/ᾱ<sub>s</sub>)<br/>x<sub>s</sub> = P<sub>t,s</sub>(x<sub>t</sub>,clip m<sub>t</sub>,ξ<sub>t</sub>)+Δ<sub>t</sub>',
 '∇<sub>x_t</sub>log p(y|x<sub>t</sub>) = (a<sub>t</sub>/b<sub>t</sub><super>2</super>)[E(x<sub>0</sub>|x<sub>t</sub>,y)-E(x<sub>0</sub>|x<sub>t</sub>)]',
 'κ<sub>t,s</sub> = √(ᾱ<sub>s</sub>)(1-ᾱ<sub>t</sub>/ᾱ<sub>s</sub>)/b<sub>t</sub><super>2</super><br/>κ<sub>t,s</sub>c<sub>t</sub> = (1-ᾱ<sub>t</sub>/ᾱ<sub>s</sub>)/√(ᾱ<sub>t</sub>/ᾱ<sub>s</sub>) = K<sub>t,s</sub>'
]
# Arial Unicode lacks some mathematical alphabet glyphs; use the already-defined plain A.
MATH=[x.replace('𝒜','A') for x in MATH]

def figure():
    d=Drawing(510,105);labels=[('当前图像与时间','固定采样位置'),('去噪预测与残差','物理加权目标'),('当前去噪响应','方向与长度分离'),('预算检查与回退','形成下一状态')]
    for i,(a,b) in enumerate(labels):
        x=i*130;d.add(Rect(x,33,118,51,fillColor=colors.white,strokeColor=INK,strokeWidth=.7))
        d.add(String(x+59,63,a,fontName='CN',fontSize=9.6,textAnchor='middle',fillColor=INK))
        d.add(String(x+59,45,b,fontName='CN',fontSize=8.8,textAnchor='middle',fillColor=GRAY))
        if i<3:
            d.add(Line(x+118,58,x+129,58,strokeColor=INK,strokeWidth=.7));d.add(Polygon([x+129,58,x+125,61,x+125,55],fillColor=INK,strokeColor=INK))
    d.add(String(0,11,'当前没有选择新采样位置；实时幅度是否优于冻结规则，需要独立证据。',fontName='CN',fontSize=9,fillColor=GRAY))
    return d

def make_table(lines,width):
    raw=[]
    for i,line in enumerate(lines):
        vals=[v.strip() for v in line.strip().strip('|').split('|')]
        if i==1 and all(re.fullmatch(r':?-+:?',v) for v in vals):continue
        raw.append(vals)
    n=len(raw[0]);assert all(len(r)==n for r in raw),'Malformed table'
    if n==10:ratios=[.235]+[.085]*9
    elif n==7:ratios=[.20,.32,.10,.10,.10,.09,.09]
    elif n==6:ratios=[.24,.18,.18,.20,.10,.10]
    elif n==5:ratios=[.21,.39,.135,.135,.13]
    elif n==4:ratios=[.17,.25,.29,.29]
    elif n==2:ratios=[.23,.77]
    else:ratios=[1/n]*n
    use=ParagraphStyle('tiny',parent=cell,fontSize=7.8 if n==10 else 8.2,leading=11.5 if n==10 else 12.2)
    data=[[Paragraph(inline(v),use) for v in row] for row in raw]
    t=LongTable(data,colWidths=[width*r/sum(ratios) for r in ratios],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
       ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
       ('LINEABOVE',(0,0),(-1,0),1.0,INK),('LINEBELOW',(0,0),(-1,0),.65,INK),('LINEBELOW',(0,-1),(-1,-1),1.0,INK)]))
    if n in (7,10):
        for k in range(2,len(raw)):
            if raw[k][0]!=raw[k-1][0]:t.setStyle(TableStyle([('LINEABOVE',(0,k),(-1,k),.35,colors.HexColor('#A6AFB5'))]))
    return t

def page(canvas,doc):
    w,h=canvas._pagesize;canvas.saveState();canvas.setFont('CN',8);canvas.setFillColor(GRAY)
    canvas.drawString(38,h-22,'去噪响应引导研究 / 导师汇报稿 / 2026-10-28')
    canvas.drawRightString(w-38,h-22,'已完成证据；完整计时仍待补齐')
    canvas.drawString(38,20,'固定探索网格；实验暂停。文献引用不参与本地排名。')
    canvas.drawRightString(w-38,20,str(doc.page));canvas.restoreState()

def main():
    source=ROOT/'REPORT.md';text=source.read_text();font=pdfmetrics.getFont('CN')
    checked=re.sub('<[^>]+>','',text+''.join(MATH));missing=sorted({ord(c) for c in checked if ord(c)>31 and ord(c) not in font.face.charToGlyph})
    if missing:raise RuntimeError('Missing glyphs: '+str([hex(x) for x in missing]))
    p=A4;w=landscape(A4);doc=BaseDocTemplate(str(DEST),pagesize=p,leftMargin=39,rightMargin=39,topMargin=43,bottomMargin=37,
       title='固定采样网格下的去噪响应引导：动机、推导与实验依据',author='ZAPS research',subject='Conference-style evidence tables; no unmeasured runtime claim')
    doc.addPageTemplates([PageTemplate(id='portrait',pagesize=p,frames=[Frame(39,37,p[0]-78,p[1]-80,id='p',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page),
      PageTemplate(id='landscape',pagesize=w,frames=[Frame(35,37,w[0]-70,w[1]-80,id='l',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page)])
    story=[];lines=text.splitlines();i=0;mi=0;fig=False;tables=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),title));i+=1;continue
        if line.startswith('## '):
            label=line[3:]
            story.append(Paragraph(inline(label),h2));i+=1;continue
        if line.startswith('### '):story.append(Paragraph(inline(line[4:]),h3));i+=1;continue
        if line=='\\[':
            i+=1
            while i<len(lines) and lines[i].strip()!='\\]':i+=1
            story.append(Paragraph(MATH[mi]+'　('+str(mi+1)+')',equation));mi+=1;i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'):block.append(lines[i]);i+=1
            # Paragraphs permit CJK wrapping while preserving visible indentation.
            head=story.pop() if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='h3' else None
            story.append(KeepTogether(([head] if head else [])+[Paragraph(inline(s).replace('    ','　　'),code) for s in block]));i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i].strip());i+=1
            n=len(block[0].strip('|').split('|'));wide=n in (7,10);tables+=1
            if wide:
                story.extend([NextPageTemplate('landscape'),PageBreak()])
                if n==7:
                    rows=block[2:];groups=[]
                    for s in rows:
                        key=s.strip('|').split('|')[0].strip().split('/')[0].strip()
                        if not groups or groups[-1][0]!=key:groups.append((key,[]))
                        groups[-1][1].append(s)
                    for gi,(key,rows) in enumerate(groups):
                        if gi:story.append(PageBreak())
                        story.append(Paragraph('同协议模块消融：'+key,h2));story.append(make_table(block[:2]+rows,w[0]-70))
                        story.append(Spacer(1,8));story.append(Paragraph('加粗为最优，下划线为次优。完整重建耗时未测；末轮回放秒数不用于总体加速结论。',small))
                else:
                    story.append(Paragraph('原文背景与本地对照：文献引用不参与本地排名',h2));story.append(make_table(block,w[0]-70))
                    story.append(Spacer(1,8));story.append(Paragraph('原文每数据集1000图；本地新图人脸六张、自然图像四张；去模糊为旧五张人脸图。实际探索网格不同于原文。',small))
                story.extend([NextPageTemplate('portrait'),PageBreak()])
            else:story.extend([make_table(block,p[0]-78),Spacer(1,10)])
            continue
        if line.startswith('- '):story.append(Paragraph(inline(line[2:]),small,bulletText='•'));i+=1;continue
        block=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','```','\\[','- ')):
            block.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(block)),body))
        if not fig and '这个收窄有研究依据' in ' '.join(block):story.extend([Spacer(1,7),figure()]);fig=True
    assert mi==len(MATH) and tables>=7
    doc.build(story)
    record=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
       source=str(source),pdf=str(DEST),equations=mi,tables=tables,scope='All narrative and tables from REPORT.md; twelve equivalent equations typeset with consistent symbols; one explanatory diagram',missing_glyphs=[])
    (OUT/'render_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('PDF:',DEST)

if __name__=='__main__':main()
