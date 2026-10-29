from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage
import pypdfium2 as pdfium

HERE=Path(__file__).resolve().parent;OUT=HERE/'output';(OUT/'pdf').mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Song','/System/Library/Fonts/Supplemental/Songti.ttc',subfontIndex=6))
pdfmetrics.registerFont(TTFont('SongBold','/System/Library/Fonts/Supplemental/Songti.ttc',subfontIndex=1))
pdfmetrics.registerFont(TTFont('Times','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
pdfmetrics.registerFont(TTFont('TimesBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
styles={
 'title':ParagraphStyle('title',fontName='SongBold',fontSize=21,leading=29,spaceAfter=16),
 'h':ParagraphStyle('h',fontName='SongBold',fontSize=13,leading=21,spaceBefore=10,spaceAfter=6),
 'p':ParagraphStyle('p',fontName='Song',fontSize=10.5,leading=17,wordWrap='CJK',spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Song',fontSize=9,leading=14,wordWrap='CJK',spaceAfter=7),
 'eq':ParagraphStyle('eq',fontName='Times',fontSize=13,leading=22,spaceBefore=5,spaceAfter=10,leftIndent=14),
}
story=[]
def P(s,style='p'):story.append(Paragraph(s,styles[style]))
def eq(s):P(s,'eq')
def figure(name,width=505):
 f=OUT/'png'/name;w,h=PILImage.open(f).size;story.append(Image(str(f),width=width,height=width*h/w));story.append(Spacer(1,8))
def footer(canvas,doc):
 canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#555555'))
 canvas.drawString(42,24,'ZAPS 噪声校正残差投影 | 2026-10-29 | A70已完成，A71待服务器运行')
 canvas.drawRightString(A4[0]-42,24,str(doc.page))
P('噪声校正残差投影','title')
P('动机、推导、已有证据与四任务补充实验','h')
P('目标是在保留原ZAPS引导方向的基础上，实时决定每一步需要多强的校正。观测带噪时，不应把所有残差都当作可恢复信号。当前方法先估计残差中超过噪声的部分，再把它投影到原引导的观测响应上。')
P('1 反问题与原ZAPS更新','h')
eq('y = H(x<sub>0</sub>) + n, &nbsp; n ~ N(0, σ<sub>y</sub><super>2</super>I)')
P('当前本地实验使用模型[-1,1]坐标，噪声标准差也在同坐标中。下式alpha_bar表示标准累计信号系数；完整数学记号与证明见同目录REPORT.md。','small')
eq('m<sub>t</sub> = clip[(x<sub>t</sub> − √(1 − alpha_bar<sub>t</sub>) ε<sub>θ</sub>)/√alpha_bar<sub>t</sub>]')
eq('r<sub>t</sub> = y − H(m<sub>t</sub>), &nbsp; x<sub>s</sub> = x<sub>s</sub><super>(0)</super> + γ<sub>t</sub>q<sub>t</sub>')
P('m为原去噪干净预测，q为原ZAPS含学习权重和小波参数的引导，无条件项为原DDPM候选。新增部分只决定倍率，不构造JAPS式新方向。前9轮保持原生，仅第10轮的30步反馈。')
P('2 估计残差中的可校正信号','h')
P('采用局部工作模型r=e+n，信号误差e的方差为τ²，观测噪声方差已知为σ²。独立正态模型下，由后验配方得到信号误差的条件均值。')
eq('e ~ N(0, τ<super>2</super>I), &nbsp; E[e | r] = [τ<super>2</super>/(τ<super>2</super> + σ<sub>y</sub><super>2</super>)]r')
P('再用边缘残差能量估计非负信号方差。M是有效观测标量数；补洞只统计已观测的RGB位置。')
eq('tau_hat<super>2</super> = max(||r||<super>2</super>/M − σ<sub>y</sub><super>2</super>, 0)')
eq('ω = (1 − Mσ<sub>y</sub><super>2</super>/||r||<super>2</super>)<sub>+</sub>, &nbsp; e_hat = ωr')
P('残差接近噪声量级时，ω接近0；残差明显超过噪声时，保留更多待校正信号。ω是工作模型下的收缩系数，不是已校准的后验置信度。')
story.append(PageBreak())
P('连续倍率与创新边界','title')
P('3 从目标残差信号求倍率','h')
eq('d<sub>t</sub> = H(m<sub>t</sub> + q<sub>t</sub>/√alpha_bar<sub>t</sub>) − H(m<sub>t</sub>)')
P('d是A33式低成本同时间响应代理：不额外运行下一步去噪器，不计算新的小波近似梯度。')
eq('γ<sub>t</sub> = arg min<sub>γ≥0</sub> ½ ||ω<sub>t</sub>r<sub>t</sub> − γd<sub>t</sub>||<super>2</super>')
eq('γ<sub>t</sub> = ω<sub>t</sub> max(r<sub>t</sub><super>T</super>d<sub>t</sub>, 0) / ||d<sub>t</sub>||<super>2</super>')
P('ω决定有多少残差值得校正；内积判断原引导是否对准残差；响应能量把校正量换算成倍率。解析连续解不需要迭代次数、线搜索系数或新增β，也不设置倍率上限2。')
P('4 哪些属于已有方法，哪些可以作为改进贡献','h')
P('已有部分：正态后验收缩、非负方差估计、一维最小二乘及非负投影。2025年Peter Hoff官方讲义§6.3明确列出了与ω同型的正部经验贝叶斯公式。因此不能将该收缩公式宣称原创，也不能直接搬用James-Stein/SURE风险保证。')
P('候选贡献：把噪声校正残差和原ZAPS引导的观测响应相结合，构造低成本连续倍率控制器，并用强基线、错配实验和完整时钟验证其作用。目前是增量方法候选，是否成为论文主贡献取决于补充任务的稳定结果。')
P('边界：预测m通过引导依赖y，独立正态仅为解释模型；当前ζ的正幅度可能被连续重标抵消，但前9轮学习历史和D方向仍有效。当前没有实现轨迹曲率控制、步长自适应或节省NFE。')
P('5 当前证据','h')
P('A70四张新图上，正常/高噪声相对基线PSNR为+0.151/+1.825 dB，均4/4图正向。正常噪声LPIPS略差，高噪声三项质量均明显改善。相对跨图错配倍率，额外PSNR仅+0.021/+0.023 dB，两档3/4图正向：这是小幅实时匹配信号，不是全部收益来源。')
P('参考：ZAPS，ECCV2024，方法§3；Peter Hoff，Shrinkage and empirical Bayes，2025-02-20，§2、§6.3。完整链接、源记录与数学推导见REPORT.md。','small')
story.append(PageBreak())
P('已完成结果：三线表','title')
figure('A70_quality_time_three_line.png')
P('当前表：相同图像、观测、学习前缀与噪声的配对质量；完整10轮独立重建计时。粗体最优、下划线次优，按同噪声条件内排序。原论文1000图均值没有混入此排名。','small')
figure('historical_fixed_three_line.png')
P('历史表：复用A33报告继承的C/F记录，固定规则仅在最后一轮t319乘2。旧探索网格/物理域与当前不同，完整独立时钟缺失，不能填入当前同配置消融。','small')
P('6 当前调度与补充任务','h')
P('当前为15/10/5分段线性，每轮30点、共10轮；不是旧Karras ρ=3。当前网格没有319，不静默映射旧固定点。','p')
P('A71固定00226/00443/00963三图，四FFHQ任务，噪声0.025/0.05/0.10。复用SR已有12条记录，仅补60条完整运行；新增18000方法+1800配对诊断前向，0输入反向。原固定参照不重跑。','p')
P('服务器将自动保存GT、观测、基线和新方法的全部三图、统一局部ROI，以及每噪声三线表PNG/SVG/CSV/TeX。当前目录已提供A70实际输出对比图和旧固定参照原图裁板；未补造缺失GT或未完成任务。','p')
file=OUT/'pdf/ZAPS_shrink_rule_briefing.pdf'
SimpleDocTemplate(str(file),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=42,title='ZAPS噪声校正残差投影：推导、创新判断与实验').build(story,onFirstPage=footer,onLaterPages=footer)
doc=pdfium.PdfDocument(str(file));qa=HERE/'tmp/pdf_qa';qa.mkdir(parents=True,exist_ok=True)
for i in range(len(doc)):
 page=doc[i];page.render(scale=1.35).to_pil().save(qa/f'page_{i+1}.png')
print(file,'pages=',len(doc))
