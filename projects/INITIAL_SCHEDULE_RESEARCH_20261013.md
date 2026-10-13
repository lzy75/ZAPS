# 初始化调度补充研究：模型相关成本，而不是再次扫描网格形状

日期：2026-10-13。工作线 **I**。状态：**文献核查与实验提案，未实现、未运行、未获GPU执行授权**。
此文件为并行子任务的独立存档；不改变核心、默认参数、FFHQ、ImageNet或batch。
先向用户解释必要性与预算，确认后才建立正式实验卡和独立入口。

## 1. 本次结论与推荐顺序

没有发现一个有证据的、可直接移植到当前ZAPS的“通用最优30步网格”。
已有uniform、power、Karras、sigma/logsigma/logSNR只是改变密度形状；
本次优先考虑**使用当前checkpoint的实际去噪变化或轨迹误差来决定取点**。
只新增两个机制候选，先固定每轮30次模型调用，999→0完整覆盖：

1. **AYS-inspired discrete denoiser-cost DP**：以当前VP模型的前向噪声耦合，
   构造离线去噪器冻结误差成本，从合法离散节点中选30个节点。
   更接近当前随机采样器的研究方向，但简化估计不称完整AYS或已认证KL上界。
2. **GITS-inspired model-trajectory DP**：用同checkpoint的无条件PF-ODE
   teacher与局部Euler误差选点；最终重建仍用原ZAPS DDPM，不换求解器。
   这是ODE网格向随机逆问题的迁移，需实验判定，不能预先承诺提高质量。

这两项都在单图在线重建之前固定网格，不使用当前测试图GT选步；
它们属于初始化I线，不是状态感知A线。ImageNet首步幅度/方向的干预结果
不作为这两项的默认配置或新的原文基线。

## 2. 既有证据与去重

已读`AGENTS.md`、`EXPERIMENT_STANDARD_20261013.md`、`NEXT_TASKS_20261013.md`、
`IMAGENET_BASELINE_MANIFEST_20261013.json`、`PROGRESS_ARCHIVE_20261009.md`、
`SERVER_RECORD_AUDIT_20261013.md`和相关网格构造/核心代码。

FFHQ当前开发基座为SR×4、30×10=300 NFE、ζ/D联合学习、lr=.01、
ζinit=.1、Dinit=.2、fixed-small、eta1、last_opt、legacy_bicubic经验引导T。
**T不称精确H^T，这个基座也不称作者实现认证。**

| 固定初始化，既有10图 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| paper15/10/5 | 28.1905 | .8360 | .1265 |
| uniform30 | 28.3043 | .8438 | .1303 |
| Karras rho7 | 28.3348 | .8315 | .1551 |

数据来自上传分支`results/server-records-20261013`，提交`d2c8122`，
原服务器`results/diag_ffhq_timestep_multi/20260917_020040/metrics.csv`及
`logs/ffhq_timestep_multi_10.log`。本次读取了逐图表及实际网格，非新运行。
既有0–9图已经参与探索，后续10–19图若文件存在且未参与其他调参，
才可作为这次的预注册独立确认集。

既有状态调度Karras rho3平均+.1398 dB、9/10胜，但LPIPS+.0067；
uniform/paper状态增量约+.0321/+.0407 dB。不能因PSNR正增量就说综合成功。

**缓存限制**：本次看到的旧固定调度目录有CSV和日志，但未在这个文本目录
找到完整run.json/权重/H/y/RNG/source指纹。旧均值可以用于去重与先验排序，
还不能直接当新运行严格同观测配对的另一臂。服务器原始缓存可能另存，
先核查清单与张量；若身份不齐，明确标历史对照，不伪造精确复用。

不再扫描power/rho/logSNR；AYS/GITS只是在旧计划中提及，检索到的完成记录
尚无这两种模型成本构造实验。若服务器出现同机制已完成run，应先复用/去重。

## 3. 主来源核查：能借鉴什么，不能借名什么

### 3.1 AYS：必须区分三种含义

[AYS论文§3、附录B](https://arxiv.org/html/2404.14507v1)以随机求解器离散误差
的KLUB为目标，按模型、数据和求解器优化网格；Gaussian数据下的arctan
解析最优只针对PF-ODE的Euler/DDIM假设。自然图像不能仅估一个方差c，
就把arctan网格称为随机DDPM的最优AYS。

完整估计并不免费：原论文使用8192个数据样本；ImageNet256优化报告为
8张RTX6000约3–4小时。不能把“推理时零额外代价”误写成“调度搜索零成本”。
本项目不照搬这个资源规模，下面的有界离散成本是**自定义研究代理**，
不是原算法完整复现。[AYS论文附录B.1](https://arxiv.org/html/2404.14507v1#A2.SS1)

[官方quickstart](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html)
公开的快速网格是SD1.5、SDXL、DeepFloyd和SVD，并给出log-linear插值办法；
它没有提供当前FFHQ256/ImageNet-uncond权重可直接使用的认证网格。
公开10步数组还包含终端噪声节点，不能直接把数组长度当本项目的模型调用数。
不复制SD/SDXL数组，也不暗中把DDPM换成DPM-Solver++。

### 3.2 GITS：有官方代码，但模型/求解器/NFE都要重核

[GITS论文§4与附录A.4](https://arxiv.org/html/2405.11326v1)利用ODE轨迹的
局部离散误差建立成本矩阵，再用DP分配节点。生成任务的结果不是测量引导
ZAPS的质量保证，PF轨迹上的成本也不自动等于随机DDPM的误差。

[官方README](https://github.com/zju-pi/diff-sampler/tree/main/gits-main)的FFHQ
预训练模型是EDM64；ImageNet256示例为ADM classifier-guided，而不是这里的
uncond checkpoint。`num_steps`指timestamps，例如7个节点对应6个采样区间；
默认warmup256、teacher61节点，不能抄参数后写成30 NFE或免费初始化。

[官方gits_utils.py](https://github.com/zju-pi/diff-sampler/blob/main/gits-main/gits_utils.py)
缓存teacher states与局部方向后，可计算各跨段Euler终点误差，成本矩阵计算
本身无需新增UNet。公开DP是`C(current,next)+coeff*future_cost`，不是把全部
成本乘同一个系数。官方coeff=1.15主要用于极少步场景，不能假定对29个
区间仍合理：递归中远端权重可累乘。下面先固定coeff=1，不打开系数扫描；官方`dev`与
`l2`不同，提案明确采用l2，故名称是GITS-inspired，不是原结果复现。

### 3.3 额外检索的研究：本轮不增加第三个候选

| 方法 | 主来源/代码核查 | 本轮不直接运行的原因 |
| --- | --- | --- |
| DM-NonUniform，CVPR2024 | [论文](https://arxiv.org/abs/2402.17376)、[官方step_optim.py](https://github.com/scxue/DM-NonUniform/blob/main/step_optim.py) | 其公开优化目标使用高阶多步ODE积分系数，不查询当前UNet；虽可CPU生成候选，不能称为当前一阶随机DDPM的误差最优。暂不扩候选数 |
| Optimal Stepsize/OSS，2025 | [论文](https://arxiv.org/abs/2503.21774)、[官方适配说明](https://github.com/bebebe666/OptimalSteps#airplane-appling-for-other-models) | 官方适配要求flow-matching形式velocity与局部直线更新；不能把当前epsilon-DDPM直接喂进去，需另论VP/PF转换及随机过程，不本轮叠加 |
| Adaptive Non-Uniform Timestep Sampling，CVPR2025 | [官方实现](https://github.com/ku-dmlab/Adaptive-Timestep-Sampler) | 是模型训练时的采样策略，不是现成checkpoint的推理网格优化 |
| Tracing the Oracle，2026预印本 | [原始论文页面](https://arxiv.org/abs/2606.06236) | 3D CT/DDS问题，与本项目任务/模型不同；本次仅发现原始论文，未核实可复用的官方实现，不列可执行候选 |

较新论文也没有取消checkpoint/solver/任务对齐的必要性。不能据方法名或
生成FID宣称它会比uniform有显著PSNR优势。

## 4. 公共坐标与整数投影协议（自定义工程规范）

从实际checkpoint对应扩散表读取，不重新设beta：

\[
a_t=\sqrt{\bar\alpha_t},\quad b_t=\sqrt{1-\bar\alpha_t},\quad
\sigma_t=b_t/a_t,\quad\lambda_t=\log a_t-\log b_t=-\log\sigma_t.
\]

这里σ是VP噪声/信号比，不是measurement noise σ_y=.05。
noise-coordinate端点固定σ(999)与σ(0)>0；t=0后原核心的终端输出操作保留，
不把σ=0塞进log，不删掉t=0模型调用。当前30个模型时间点意味着：
999→0之间29个区间，加上t=0处最后一次模型评估，共30次/轮。

使用61个候选整数节点构成公共lattice（0和999必须在内）。本次提案用
在logσ坐标等距的61个连续目标，仅作为成本采样lattice，不是重扫
uniform_logsigma最终网格；随后从中选择30节点。先投影lattice再计算模型成本，
使teacher和AYS缓存使用真正整数t，不插值UNet权重/embedding。

连续目标的建议投影是最小化总logσ误差的严格有序指派：

\[
\min_{0=i_0<i_1<\dots<i_{M-1}=999}
\sum_j[\log\sigma(i_j)-\log\sigma_j^*]^2 .
\]

这是本项目投影方案，不是AYS/GITS论文定理；CPU DP即可做，0新增NFE。
记录投影误差、最小/最大t跳跨度、实际M/S、grid hash和alpha表hash。
避免逐点round后随手去重导致不足30步，或默认末端大跳。沿用旧贪心投影
时必须显式说明并比较0-NFE投影误差，不能隐藏算法更换。

选点DP只允许端点可达、严格互异且足够剩余节点的边；输出30节点后
ζ/D初始化与联合优化按该固定网格重建，**不把旧已学ζ/D按步号搬过去**。
初始值同参照，整个run网格不变；可变网格的noise-anchor插值后置。

## 5. 两个新增候选的可否定机制与离线成本

### I-AYS-proxy：模型相关去噪器冻结成本

必要性：旧Karras/power只问“哪里更密”，没有测当前UNet在各噪声段的
去噪估计变化；候选用实际模型差异确定密度，且不需要重建GT指标作成本。

用3张已参与开发的校准图（提案ID00000/00004/00007；先核实身份），
每图4个预注册的独立前向噪声路径。转换为VE归一化状态z=x_VP/a：

\[
z_{\sigma_{j+1}}=z_{\sigma_j}
+\sqrt{\sigma_{j+1}^2-\sigma_j^2}\,\xi_j,\qquad
z_{\sigma_0}=x_{clean}+\sigma_0\xi_{-1}.
\]

每个ξ独立；**不能在全部σ上只复用同一个epsilon**，那不是AYS需要的
前向条件耦合。UNet接收a_j z_j，t为实际整数；缓存该节点去噪预测
d_j=(x_VP-b_j epsilon_theta)/a_j。候选成本使用raw去噪输出（不加新裁剪），
这点独立记录；最终ZAPS核心的原Tweedie clamp不改变。

给定高噪节点j和低噪节点i，使用离散求积代理：

\[
\widehat C_{ij}=\sum_{k=i}^{j-1}
\frac{\sigma_{k+1}-\sigma_k}{\sigma_k^3}
\frac{1}{12d}\sum_{n=1}^{12}\|d_k^{(n)}-d_j^{(n)}\|^2.
\]

这是本项目的左端求积/小样本denoiser-freezing代理；近零σ权重偏大、
有限格点偏差与近似score假设都要披露，不能称严格自然图像KLUB认证。
按29区间最小和成本DP选30节点，无额外UNet。与官方AYS迭代MC优化不同，
名称固定为`ays_proxy_dp30`，不能简写为官方AYS最优。

预算：3图×4噪声路径×61节点 = **732离线sample-equivalent NFE**。
批处理并不降低此数；可另报batch-forward调用数和耗时。缓存可用于CPU
bootstrap/分半检验，不追加新的前向。若两半选点差异很大、低噪声单格
支配成本或存在非有限值，先判成本不可靠；不靠追加画质扫描掩盖。
稳定性阈值在实验卡中预注册，若未过则不解锁GPU重建对照。

### I-GITS-proxy：同模型teacher局部误差成本

必要性：这是第二种独立信息——模型生成轨迹的实际弯曲/局部误差，
不靠选c/rho猜密度，也不利用测试图测量残差决定初始网格。

在归一化VE状态上用同checkpoint构造PF-ODE teacher：

\[
z=x_{VP}/a,\qquad f_\theta(z,\sigma_t)
=\frac{z-D_\theta(a_tz,t)}{\sigma_t}.
\]

teacher从σ(999)×标准Gaussian开始，沿61个合法整数节点Euler运行60区间，
缓存z_i和f_i。raw denoiser与转换分开门控；不改变模型训练值域，
不插入classifier、不更换checkpoint、不调用官方自动下载模型逻辑。
最终重建仍使用原DDPM fixed-small eta1；teacher ODE只产生固定网格。

\[
 C_{ij}=\frac1{16}\sum_n
 \|z_i^{(n)}+(\sigma_j-\sigma_i)f_i^{(n)}-z_j^{(n)}\|_2,
 \quad V(i,m)=\min_{j>i}\{C_{ij}+V(j,m-1)\}.
\]

这里i/j按高→低噪声顺序。固定metric=l2、coeff=1、16个预注册
无条件warmup seeds，不逐图挑网格。官方256warmup在这里缩到16，
仅是资源受限迁移pilot，不能外推原论文稳定性。coeff=1意味着本提案
采用未重加权的局部成本之和，不宣称它提供全局ODE或DDPM误差保证。

预算：16×60 = **960离线sample-equivalent NFE**；成本矩阵/DP/分半
稳定性检验零新增UNet。teacher不作t=0额外denoise，故离线60而不是61。
最终ZAPS仍明确30次/轮，末端调用不能“免费”。缓存不兼容时拒绝复用；
不能用30步含噪且测量引导的旧ZAPS轨迹冒充本PF teacher。

若teacher异常或分半成本导致网格明显不稳，本轮候选标inconclusive；
换121节点teacher、换Heun、扩大warmup都属于新预算/新假设，需另确认。
暂不打开gamma/teacher/rho/样本数多维搜索。

## 6. 高效筛选：先0-NFE，再3图，最后独立10图

以下为待确认实验卡提纲，正式运行前还需填写全部绝对父run、hash、软件环境、
seed表、命令、失败停止条件与CPU/GPU门控；目前不交付运行命令。

### I0：缓存与准入，0 NFE

- 用旧记录排除已跑网格形状；按image/H/y/RNG/模型/源码/输出/指标身份复用，
  不按schedule名称或日志日期自动认定兼容。
- 验证两候选的VP↔VE公式、整数投影、节点数、端点、last t=0/NFE语义。
- 随机噪声路径与重建噪声分账；AYS前向耦合/GITS teacher不能混用。
- 所有变体状态调度关闭，ζ/D联合；noise/首步修正/方差策略保持参照不变。
- 使用61节点而不是仅30节点，避免DP只有唯一选点路径；本轮不改变999→0范围。

### I1：至多两新增候选的3图校准

FFHQ SR×4先行，ID00000/00004/00007，seed1000/1004/1007，均已经探索过，
称校准而不是测试。参考为uniform30和paper15/10/5；新候选两种。
Karras7旧结果有较大LPIPS代价，用于Pareto背景，不无理由原样重跑。

每臂固定30×10、lr=.01、ζ/D联合与上述FFHQ经验T，输出last_opt。
共享同一个y、x_T、按epoch/调用位置的噪声draw；网格变动时这只是CRN配对，
不是同一物理时间Brownian路径。若改为Brownian bridge则另立随机过程实验。

- 两新增候选重建成本：2×3×300 = **1800方法NFE**。
- 两参照均身份兼容时可复用，0新增；不兼容则最多补2×3×300=1800，
  明确补跑必要性是重建配对身份，而非分数不好重跑。
- 两离线搜索上限合计1692。I1与搜索总新增上限 **5292 NFE**，
  身份兼容全复用参照时为3492；另记必要门控的诊断NFE，不藏进免费项。
- 零状态响应/null门控与旧重复误差先确定；不能因新方法波动就放宽门槛。
- 同时报PSNR/SSIM/LPIPS、残差、耗时、actual NFE、clipping/轨迹异常。

**提案工作门槛，须先与用户确认**：对uniform配对mean dPSNR≥+.10dB，
同时mean dSSIM≥−.002、mean dLPIPS≤+.002，且3图至少2图PSNR正增量，
才作为综合候选进入确认。这些容忍度不是文献定律；若既有重复误差大于
门槛，结论标不确定，在预注册少量第二seed中核实，不事后换成功定义。
不使用加权总分覆盖感知退化；可另标PSNR/感知Pareto候选，但不称综合成功。

若两新增候选被uniform稳定支配，停止这一轮离线机制，不扩大搜索；
保留uniform为A线锚点。若都处于Pareto且难区分，先向用户说明取舍再确认
一个主候选，不自动把独立10图扩成全部候选矩阵。

### I2：独立10图确认，最多3种方法

冻结一个I1候选及所有参数；确认ID00010–00019从未参与此次选择且存在，
种子表提前固定。运行新候选、uniform与paper，共3×10×300 = **9000方法NFE**。
这些新的验证图没有旧配对缓存，预算不可声称为零。

报告逐图配对差、mean/std/median/胜率及区间，特别说明10图统计范围有限。
主要工作目标仍为上述综合非劣界与mean+.10dB，且至少7/10 PSNR胜；
若差值与GPU重复误差相近或区间跨0，只称小规模趋势，不宣称统计确认或全局最优。
若LPIPS稳定恶化（例如既有rho3状态+.0067），不因PSNR好看忽略。

三种结果后的处理：

| 类别 | 允许结论 | 动作 |
| --- | --- | --- |
| 综合收益支持 | 当前FFHQ开发基座上得到更好固定初始化 | 冻结网格，再进入A线固定30步双指标微调 |
| 不支持/被支配 | 该离线代理未改善当前重建 | 保留最强简单参照，不重扫旧rho/power |
| Pareto混合/小差值/成本不稳 | 尚无综合最好网格 | 记录边界，用户决定指标取舍；不无上限追加到显著 |
| 门控失败 | 无算法结果 | 先修具体身份/坐标/NFE问题，保留失败记录 |

## 7. 主线边界与后续步数研究

ImageNet R线仍优先，19.5987/19.9209是未调参local_reference，不是作者
完全一致认证。主线程正在验证的首步干预仅属于单图冻结路径敏感性，
不能用于本研究的paper baseline栏。等R线形成明确版本后，I线在该版本
上另作Gaussian/SR迁移；不能把FFHQ成本/网格跨checkpoint直接当认证。

即使新初始化没有综合胜出，A线仍可在固定uniform/paper/rho3参照上推进，
但必须区分“比本候选null好”与“比最强固定基线好”；保留物理一致性与余弦。
本次研究不设计或修改状态指标权重，不开展自动S。

30步初始化和固定30步状态收益建立后，才研究20/30/60的范围：
等总300NFE的20×15、30×10、60×5检验的是**步数—优化轮数分配**，
不能写成只有单轮S不同。若要隔离单轮步数，再另画同10轮的200/300/600
成本曲线。10/15步压力测试与自动20≤S≤60、总NFE≤300后置。
固定端点999→0，不调低起点、删尾段或减晚期噪声制造初始化收益。
GITS的61节点teacher不足以在S=60时形成充分候选空间，需要另卡扩lattice，
不能沿用30步搜索成本声称60步全局优化免费。

## 8. 交付状态

- 文献与官方代码：只读核查完成；链接均为论文/作者项目或官方实现。
- 新候选：两项提案；GPU运行、数值一致性、实际质量收益均未验证。
- 正式算法、默认值、FFHQ/ImageNet/batch：未修改。
- 后续执行前：用户确认候选必要性、资源与门槛后，独立实验卡/入口先行。
