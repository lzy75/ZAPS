# A50r2：当前条件得分几何与ZAPS权重的兼容性

状态implementation_ready_server_gates_pending；2026-10-26；A线。
独立driver/launcher/无标签控制冻结/完整终局评估和图片归档已实现；Mac未运行Torch或UNet。
服务器先执行无skip的Torch准入，再prepare冻结实际父身份、预算和本卡，失败保留成本并停止。

## 问题、来源和可否定假设

A48相同局部目标的rho门控改变0/20动作；SR/IP的LOCAL较LEGACY退化。
不继续扫lambda/上限/rho。本次改测当前网络的真实Jacobian作用与正则化观测回投，
区别于相邻时间余弦、二维协方差与小动作曲线外推。依据JAPS v3：
https://arxiv.org/html/2511.18471v3 。投影是已有方法，不宣称原创理论或保证终局画质。

与原q方向保持一致，将当前得分目标的位移投到span(q)，比较PROX_Q、DIRECT_Q、
ALIGN_Q三个嵌套目标。允许结论为mechanism_only；禁止先宣称四任务状态收益、
完整后验、曲率控制、NFE节省或新图泛化。

计划父数据为A44五旧图00446/00559/00608/00873/00951×四任务，seed1000、sigma=.05；
父目录 /data0/lzy/ZAPS-results/diag_frozen_state_matching/A44_20261025；
A45目录 /data0/lzy/ZAPS-results/diag_balanced_state_calibration/A45_20261025。
仅completed、签名/源码/模型/measurement/原参数兼容可复用；这些不是新图。
原mask/kernel、unit-domain、last_opt、前270调用和最后30时间点全部保持。
实际图像/模型/算子/观测/软件哈希在prepare取自父工件后冻结，不填伪造hash。

## 数学定义与消融

a=sqrt(alpha_bar_t)，b2=1−alpha_bar_t，r2=b2/a；L=.5H。
m为未clamp Tweedie均值，r=y−A(m)。修补只解释有效观测；CG使用真实L/LT。

    w=(L LT + sigma_y²/r2 I)^−1 r
    v=(a/b2) LT w
    u=J_m(x_t)^T v             # v detached;真实网络VJP
    alpha_align=dot(v,u)/dot(u,u)
    target = K(t,s) * {v, u, alpha_align*u}
    gamma=max(0,dot(q,target)/dot(q,q))

q仍来自原clamp clean与learned ζ/D；只改变幅度，不换方向。
gamma*q对q的正标量重缩放严格不变，因此这里会消除当前ζ的直接幅度作用，
不是保留原ζ的轻微校正；之前ζ仍影响历史路径与学得D。必须报告这种方法范围变化。
h的K接口仅单位契约，不作为步长算法或收敛保证。

纯数学内核仅处理有限向量，零范数标unresolved；不把NA填成余弦1。
本次数值契约：CG真实相对残差<=1e-5、最多128次迭代；失败显式记unresolved并回gamma1。
q的RMS地板32*eps(dtype)*max(1,RMS(x))；u地板32*eps*max(1,RMS(v))；
不可分辨方向回gamma1，未缺失的PROX仍独立计算。非finite/非SPD停止，保留成本。
上述是数值准入，不是画质阈值，不在GPU阴性后改动。负vTu原样报告，不暗中取绝对值。

## 精度、成本和准入

已实现零网络NumPy代数核、Torch CG/VJP与完整独立入口；目前实际运行仍为零UNet。
Mac无Torch；Torch玩具线性模型的真实VJP/有限差分测试留服务器，禁止skip冒充通过。
真实FFHQ模型还需同输入grad/no_grad读数比较、双尺度导数检查、无权重grad/RNG变化、
真实线性伴随/CG残差和原C回放，不能只凭玩具测试称模型准入。

若共享单次IEEE前向，所有新臂需要同IEEE生产参照C_IEEE；另保留C_native。
不得同时声称native生产不变、IEEE传感免费且每步只有一次前向。
forward NFE之外另列backward、CG/operator次数、walltime和显存。

所有IEEE臂采用同样graph-enabled输入前向；C_NATIVE保留原no_grad/native对照。
首张00446各任务在k8/24/28作真实VJP双尺度差分，每点6前向+1反向；
grad/no_grad重复相对误差<=1e-5，方向差分与autograd相对误差两尺度均<=5%。
这只校验导数接口，不能认证真后验；失败不解释为重建阴性。

本次完整候选：最后30步每步反馈，固定位置/参数，不再只测t319单脉冲。
C_native/C_IEEE/PROX/DIRECT/ALIGN与GT-free阶段/固定/循环错配对照；
五旧图只作开发机制，之后才确定未用图。完整设计20例×8臂×30=4800诊断前向，
另4任务×3点×6=72数值门控，共4872前向；反向最多1212；没有新增训练。
阶段/固定表排除当前图，错配为预定循环下一图；所有20条自身轨迹冻结后才生成表。
这些已由纯设计检查核对，但尚无冻结运行plan或实际调用。原300科学部署成本
不含额外反向时间；不会以forward相同宣称计算等价。

## 判定与停止

先确认向量定义/可读性和状态是否真改变动作，再看完整终局PSNR/SSIM/LPIPS。
保持用户保护SSIM均降<=.002、LPIPS均升<=.005；对强固定略低可以接受，
但需要相对无Jacobian/无对齐/时间/错配至少一项可解释操作价值。
本次仅开发工程筛查，每任务独立：每图至少24/30个ALIGN状态可数值分辨；相对C_NATIVE
和C_IEEE均PSNR平均增益>=.05dB、至少4/5图PSNR正增，满足上述均值保护线才称质量候选。
在此基础上，相对TIME和SHUFFLE均PSNR平均增益>=.02dB、至少4/5图正增且保护线通过，
才称该任务匹配候选。门槛为预先设定工程值，不是统计显著性、论文通过标准或文献保证。
五旧图不做置信推断；对PROX/DIRECT/CONSTANT同时完整报告，不以某个单项过线掩盖其余失败。
若仅部分任务支持，只进入该任务独立未用图确认，不称四任务统一有效。
SHUFFLE错配整个gamma，含ζ幅度规范化；不能单独证明alpha_align的匹配作用。
尚不包含独立的alpha-only错配，也不声称轨迹曲率或步长机制已得到验证。

失败则停止本映射，不恢复旧阈值追分。没有实际调用前不报告method_gain；
新目录独立于所有父run，旧源码/plan不改。命令入口run_a50r2_20261026.sh，前台等待最终输出，
每60秒报进度。GPU至少10GiB空闲、遵守分配；同卡则串行，不保证真实VJP不会OOM。
完成phase可重用；中断/失败phase不自动重跑或抹成本，须单独审计恢复。归档/推送失败可单独重试，
不能重新计算已完成GPU臂。记录输入反向、CG/算子调用、每臂时间和峰值显存；时间含诊断IO，
不是未经测量的端到端加速。当前仍无冻结运行plan或任何质量结论。

## r2兼容性修正：不修改已交付r1

历史基线审计和当前DPS源码共同显示：AttentionBlock.forward无条件调用checkpoint(...,True)，
其CheckpointFunction.backward对inputs+params作autograd.grad。直接把模型参数requires_grad
全部置False会触发接口失败，即使外层use_checkpoint=False。r1的小型普通线性模型测试
未覆盖此实现，不能证明真实FFHQ接口可用。该问题是新诊断实现缺陷，不是科学负结果。

本r2独立入口保持r1文件字节/目录不变，新的上下文临时保留/打开parameter requires_grad，
仅autograd.grad(mean,input)取输入导数，不创建优化器、不累计parameter.grad、不更新权重；
退出后恢复原flags。额外必跑一项直接导入仓库原CheckpointFunction的真实接口回归，
覆盖初始requires_grad True/False两种状态及权重不变/grad为空。共6项Torch预检，Mac均未执行。

r2结果目录：/data0/lzy/ZAPS-results/diag_current_score_geometry_r2/A50r2_20261026；
结果分支results/ffhq-current-score-geometry-r2-20261026。原r1如已启动，保留错误与成本，
不删除或冒充r2重复样本。r2本身仍4872前向、最多1212反向预算；跨revision总成本须另加r1实耗。
