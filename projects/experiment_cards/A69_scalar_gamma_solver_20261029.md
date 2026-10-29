# A69：固定A68目标，比较连续倍率求解器及反向是否值得

2026-10-29；用户明确要求命令、解释干净预测/输入反向/小波曲率/k/eta。
独立源码和目录，A67/A68原文件与判定不改。不是已运行模型或新优化理论。

## 目标与公式

固定A68 REAL_ISO的Phi(g)=||y-Hf_s(g)||²/(2sigma²)+||f_s(g)-f_s(0)||²/(2U_s)，
U_s=(1-alpha_bar_s)/sqrt(alpha_bar_s)，beta=1；不再每步付费测κ、不改物理坐标。
f_s(g)是真实下一时刻的raw干净预测，x_s(g)=DDPM(x_t)+g*q_t；q为原ZAPS学习引导。
实际生产更新仍保留原clamp；raw只作与A68相同的控制目标。

梯度对照：MODEL_NEWTON使用ZAPS近似J_s，AUTO_NEWTON用autograd直接求dPhi/dg；
只请求倍率标量的梯度，网络权重不更新。使用既有legacy-checkpoint兼容参数标志接口。
Jtilde_s=[I+(1-alpha_s)W D_s W^T]/sqrt(alpha_s)，wtilde=Jtilde_s q_t。
h=||H wtilde||²/sigma²+||wtilde||²/U_s，是1D目标的Gauss-Newton曲率预估，
不是真实Hessian、后验方差或整条轨迹曲率。原D学习服务于ZAPS，不能保证它准确等于网络导数。
MODEL梯度为-<H wtilde,y-Hf>/sigma²+<wtilde,f-f(0)>/U_s；AUTO用真实输入反向。
两组均p=max(0,g-gradient/h)-g、g_next=g+eta*p，没有固定倍率上限2，没有整数候选竞争。

干净预测把不同扩散噪声时刻映射回与观测对应的clean图像空间；不是GT或额外监督。
第二项控制偏离无当前q参考预测的程度。噪声sigma与原观测同模型[-1,1]坐标。
不重标目标、不增加beta；保留原ζ/D的学习历史和方向，连续调幅仍可能消除当前ζ正幅度。

## 内循环与停止

每个采样位置从g0=0开始，最多2次接受更新、3个前向评价点、2个梯度评价点。
梯度评价点数是上限，不要求每步跑满。eta从1开始；若实际Phi不满足
Phi(g+eta*p)<=Phi(g)+1e-4*eta*gradient*p，则eta减半，仍占同一前向预算。
不使用A68的10%响应预测偏差拒绝规则；实际下降才用于线搜索。
相对投影步残差R=|g-max(0,g-gradient/h)|/(1+g)，阈值.01，预先固定工程容差。
AUTO的已测梯度点R用于真实1D目标局部一阶检查；MODEL的R仅为近似梯度诊断。
第二次接受更新可能只有目标值没有剩余梯度预算，必须记final KKT unmeasured；
预算耗尽不记stationarity_met，不将0边界/保持动作自动称全局最优。
原始0点可在所有试探失败时保留，但不把倍率1设为竞争候选。

原小波变换/标量代数不产生额外UNet查询；真实梯度有输入反向成本。
选中点eps复用于下一正常步。前向、输入反向、checkpoint重算墙钟均报告。
终端s=-1直接评价clean_t+gq，梯度可解析，不调用不存在的下一网络。

## 样本、基线与消融

与A68相同4张旧FFHQ图00000–00003、sigma .05/.10、seed1000+ID，8条件4图簇。
30点15/10/5、10轮、lr=.01、ζinit=.1、Dinit=.2、db4 level3、DDPM fixed-small eta1、
历史经验bicubic引导映射，全部IEEE。前9轮原生，最后30步反馈；无全10轮控制。
GT仅合成观测和终局评价，不进入求解/停止/输出选择。

|组|用途|
|---|---|
|BASE|同场强ZAPS及完整时钟|
|FIRST_NOISE|A67有效参照|
|REAL_ISO|A68同一目标、旧探测/验收求解器|
|MODEL_NEWTON|新连续求解、近似梯度、无输入反向|
|AUTO_NEWTON|同求解器/曲率/预算，真实标量梯度|

保持相同BASE270前缀/参数/RNG的配对质量；另外每组独立完整执行测时钟和自身质量，
不把末30回放秒数叫完整时间。全部对照重测有同场计时目的，不是增加独立图数。
各组方法前向上限300/300/503/387/387；配对诊断上限30/30/233/117/117。
总<=15016 method+4216 diagnostic=19232 UNet；AUTO每次完整/配对至多58输入反向，
合计<=928输入反向；400优化反向。实际模型warmup/数值准入另计。

同时报告PSNR/SSIM/LPIPS、全流程秒数、NN与输入反向、倍率分布、最优性残差、
真/近似梯度区别、已收敛/预算终止/未测最终梯度比例。不以控制目标下降代替画质。
开发工程信号：对BASE PSNR均>=+.05，>=3/4图正向，均SSIM>=-.002、LPIPS<=+.005；
对FIRST及REAL_ISO完整呈现质量/成本取舍，不强迫全部指标正向。无信号则停止本求解器，
不扫eta、beta、容差救分；有信号再冻结未用图/seed确认。

## 执行与必要检查

Mac纯标量求解/实际子进程CLI；服务器必跑Torch默认梯度标志、NULL回放、实际输入反向
与legacy-checkpoint检查，不允许缺Torch跳过。真实FFHQ warmup先验证1次梯度且无参数.grad。
新目录diag_scalar_gamma_solver/A69_20261029；入口run_a69_20261029.sh。
断线安全、默认跟随摘要，--status查询，--archive-only只上传；双卡或单卡串行，失败不自动重试。
