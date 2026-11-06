# A95：正常噪声下的共同响应目标与联合对角求解

状态planned；A工作线。用户已同意从A94结果继续设计并实现实验。
只用原3张FFHQ00226/00443/00963×四任务，模型[-1,1] sigma=.05；固定原30点网格和10轮学习入口。
本次只改变最后一轮index15(t333->s309)一次修正，随后14次原ZAPS续推；不做高噪、新图、窗口或全30步推广。

## 已有证据与新增问题

A94完成96/96，但更换了原标量的噪声收缩目标，新增倍率N(0,I)先验，且仅两次近端更新。
其早期实际方向含W^T diag((Wq)^2)W的参数化预条件；共享标量先验又变成N*gamma²/2。
本轮去掉倍率惩罚，保留噪声校正响应目标，改在实际修正的物理坐标联合求解。
A89/A90已做线性响应的分带耦合最小二乘；本轮新增的是完整下一网络非线性响应、逐系数正负自由和共同停止标准，不重做旧分带/范数约束。

父A94 /data0/lzy/ZAPS-results/diag_joint_diagonal/A94_20261106，须96/96且0unresolved；其A92单次干预状态、原ZAPS和直接物理方向结果复用。
继承A71的原九轮学习后zeta/D、last-unroll入口/y/mask/kernel/checkpoint；所有旧源和结果保持原SHA。
SR保留历史bicubic经验反投影，物理响应/梯度使用实际H；Gaussian61/sigma3与Motion原zero边界，补洞随机70%missing。
原db4/level3/periodic、clamp[-1,1]、DDPM eta1、原15/10/5取整、uint8 PSNR/SSIM及alexLPIPS均保持。
本轮仍是小面板local_reference机制诊断，不是完整作者基准复现或新风险理论。

## 共同目标与标量退化

z_s=原DDPM下一状态(当前创新已抽取)，m_s(x)=原下一时刻clamped raw Tweedie预测，P=观测RGB选择。
b0=P H m_s(z_s)，r0=P y-b0，omega=(1-M*sigma_y²/||r0||²)_+。
原q仍包含已学习zeta/D；c=Wq，p=c*gamma，Delta=W^T p，R(p)=P H m_s(z_s+Delta)-b0。

F(p)=.5||R(p)-omega*r0||² = .5||P H m_s(z_s+Delta)-(b0+omega*r0)||²。

omega>0时，与 .5||r0-R||²+(1-omega)/(2omega)||R||² 有相同最小化解(仅正比例/常数差)。
omega=0直接使用目标R=0，不计算发散的等价惩罚系数。
正则化的是期望响应而非gamma元素；无gamma²惩罚、原向量/标量范数上限、0..2上限、beta经验权重或GT目标。
退回原A71的基准clean与线性响应R=gamma*d后，gamma=omega*max(<r,d>,0)/||d||²，严格恢复原标量公式。
实际下一基准不同于旧当前clean，故必须用NEXT_PROXY隔离基准/时间代理变化，不把所有变化归因于网络Jacobian。
omega仍是原经验噪声收缩，m(y)的数据依赖、补洞缺失域风险没有因此被校准；本轮不同时更换不确定性定义。

## 四个新臂与可解释对照

1. LEGACY_SCALAR：在同一A92状态重新计算原A71 r_t/d_t/omega/gamma，仅此一次缩放。
2. NEXT_PROXY_SCALAR：基准改成m_s(z_s)，仍采用冻结eps的Hq/sqrt(alpha_s)代理；相同原select公式。
3. REAL_SCALAR：同目标F，仅Delta=beta*q/||q||，beta>=0；初始化NEXT_PROXY的物理长度。
4. REAL_DIAG：同目标F，在所有非零c_j支持上共同优化实际系数p_j，正负自由，无子带共享。

REAL_DIAG以REAL_SCALAR的解作为可行初始化，不作为目标/范数锚点；没有“尽量接近标量”的罚项。
这保证实际数值容限内矩阵目标不高于其标量初始化；终局画质仍独立评估，不由目标改善推出。
原c_j=0处保持p_j=0；近零系数不作裁剪。gamma=p/c只在求解后作诊断，不以巨大比值生成修正。
全自由有符号系数几乎可表示任意修正，不能把自由度本身称创新；要看共同目标下是否产生真实重建增益。
原ZAPS NATIVE100、直接物理方向PHYS_DIRECT100为A92同作用位置已完成对照，直接复用，不重新抽样。
旧A71全末轮标量收益不能当本次一次干预收益，本次LEGACY必须在当前状态重算，禁止移植其他轨迹gamma。

## 联合求解、参考与资格

用实际图像长度/正交系数p作为优化坐标，避免倍率坐标普通GD隐含的c_j²缩放。
J=当前P H m_s对这些坐标的完整Jacobian，以真AD Jv及J^T v实现；不显式存巨大J，不丢交叉项。
四个固定H都是线性映射，JVP先对去噪器做精确前向AD，再应用原H和P；不要求旧Resizer/卷积的forward-AD支持，也不使用经验反投影充当伴随。
每个LM局部步解 (J^T J+mu I)delta=-J^T error；共享标量使用同一框架与非负投影。
mu只阻尼数值增量，不添加到最终F。mu初值1；实际/预测下降比>.1才接受，>.75时mu/3，<.25时mu*2；拒绝mu*4。
每次inexact CG最多12次迭代，加一次真实残差检查，门槛||true residual||/||rhs||<=.1。
最多20个外迭代，每外迭代最多4次阻尼尝试；同一projected gradient标准为1e-6+1e-3*初始范数。
满足标准才GRADIENT_TOLERANCE。达到外迭代/试探上限保留最后已接受迭代，明确OUTER_BUDGET/TRIAL_BUDGET，生成诊断图但不认证收敛/矩阵优越；不自动失败重试。
非有限值、导数/源/RNG/计数门控失败立即保留停机，无策略回退。CG不合格只增加LM阻尼，不放宽准入标准。

参考：
- 原ZAPS小波/学习结构；旧A71 noise_signal_shrink.select是噪声响应目标的直接来源。
- DPS §3式16 https://arxiv.org/html/2209.14687v4 ：通过干净预测反传物理损失；非本目标或矩阵优越性证明。
- LM/inexact Gauss-Newton成熟数值框架 https://ceres-solver.readthedocs.io/latest/nnls_solving.html#levenberg-marquardt 。本轮用mu I物理度量，不照搬Ceres整个实现或继承全局收敛。
- PyTorch2.6 torch.func.jvp https://docs.pytorch.org/docs/2.6/generated/torch.func.jvp.html ：真实前向AD，不使用正负扰动。

原custom checkpoint不支持forward AD；仅在JVP调用内临时直接执行其相同内部函数，随后恢复Python全局绑定。
不改历史文件、NN权重、precision、native/VJP/采样实现。实际权重必须通过原checkpoint VJP与直接JVP的primal<=1e-6及伴随误差门控，失败不启动正式案例。
每个案例新做一次16F原前缀回放取当前clean，检查q/uncond相对L2<=1e-6、CPU/CUDA RNG完全等于A92。
所有solver调用均s309，续推严格14点，最终RNG与A92一致；solver禁止消耗采样随机数，NN无优化更新。

## 预算、资源、评价和分支

12个共享前缀192F；48个新后缀各14F=672F。
每个非线性解最大：20*4*(12+1+1)=1120JVP，21+20*4*13=1061VJP，80额外候选值F；总2261F。
24个非线性解上限54264F；本轮总新F上限55128，VJP25464，JVP26880；原checkpoint实测准入每载入另2F/1VJP/1JVP。
实际通常低于上限；全部JVP/VJP/rejected/true-residual成本计数，LM trace不省略。对角臂部署还需计算标量初始化，不能将共享成本称免费。
每30次NN输出当前案例/臂的F/JVP/VJP，并保存运行中计数；断线重连仍跟随独立工作进程的完整日志。
GPU在用户分配0/1中选空余>=8GiB的一张；实测AD峰值+1GiB后逐项准入。不结束进程、改精度/分辨率或远程代跑。
时钟继续记录但不用于本轮筛选；只是缓存状态后缀，不提供完整300 clock或加速结论。

GT只在动作/续推完成后算PSNR/SSIM/LPIPS、下一预测全图/观测/缺失误差、观测改善的真实signal/noise分解。
主要判据：REAL_DIAG对同目标REAL_SCALAR，每任务两者均数值合格、meanPSNR>=+.1dB、>=2/3胜、SSIM不降、LPIPS不升；同时对LEGACY/原ZAPS/直接物理方向完整比较。
若真实标量未优于代理，先分析目标基准与响应选择；若标量好而矩阵弱，定位欠定自由度/目标对齐，不扩大图或频带。
若矩阵目标降低但画质退化，保留结果并定位目标—风险错配；不继续按GT调mu/omega、选择最好迭代或输出。
若求解预算未达到统一标准，明确未解答比较，不把弱结果归因矩阵表征上限。
只有共同目标及同位置矩阵收益清晰，才另立完整末轮30步实验；本脚本不自动解锁。

## 运行

bash projects/run_a95_20261106.sh --gpus 0,1
/data0/lzy/ZAPS-results/diag_response_target/A95_20261106
results/zaps-response-target-lm-20261106；断线安全，逐臂进度，终端8行主要差值，完整summary/CSV/PNG/trace自动轻量上传。
--status只读；--archive-only仅汇总上传；资源暂停可续未开始部分，失败不自动重试。
Mac无Torch/CUDA：仅NumPy/CLI/语法检查；服务器强制真实Torch和真实权重AD门控，不skip、不冒充已通过。
