# A11/A12共用协议：固定噪声时间下clean场对输入方向的响应

planned，2026-10-20，A研究线。用户授权并行GPU实验。A11=t14、A12=t54分别立卡与目录，
不把重复性门控当独立科学实验。三张既用图00023–25，各seed1000/2000；不作独立新图泛化声称。

## 问题及唯一干预

A7的q跨步余弦/Ω、A10的clean三点差分均未支持当前固定协议的预测增量，停止这些指标的窗口/正则搜索。
新问题：固定t、原输入x和PRE参数，clean场沿实际修正q与等RMS正交方向o的响应是否有稳定差异？
这与A2a的q场方向响应不同；此次只使用A6新组，新增clean响应张量和方向对照，不重跑A2a旧组。
测量局部有限尺度输入敏感度，不称真实Hessian、严格轨迹曲率、增加步数的因果收益或在线控制成功。
q是当前实现的实际修正，legacy_bicubic并非认证精确伴随，不将q直接称精确物理梯度。

每例每层：原状态中心重复两次；两个方向d∈{q,o}，各eta∈{.01,.02}，计算clean(x±eta*d,t)。
o由CPU float64固定高斯向量对q做一次Gram-Schmidt后归一化到RMS(q)，再转float32；
seed=611020+100*排序case_index+t。只有一个正交方向，不声称代表整个正交子空间。
实际float32输入跨度沿d投影，沿用A1b perturb_inputs容差，防止名义扰动小于舍入误差。

    J_d(eta) = [clean(x_plus,t)-clean(x_minus,t)]/effective_span
    gain_d(eta) = RMS(J_d)/RMS(d)
    phi_d(eta) = [loss(x_plus,t)-loss(x_minus,t)]/effective_span

两尺度J向量的相对差≤.20才记尺度一致；分母取两J的最大RMS。
每个clean差分须超过10*max(中心clean重复RMS,1e-10)，且实际输入表示门控通过，才记可分辨。
同时记录clean差分夹角/与d夹角、clamp比例、phi及原q RMS；phi为实际损失方向差分，不假设q是精确梯度。
若两尺度响应方向相反，即使gain相似也不记尺度一致。无GT参与方向、尺度、时间、参数或输出选择。

## 冻结对照、来源及门控

A10 /data0/lzy/ZAPS-results/diag_clean_roughness/A10_20261020，必须complete、SHA/完整统计复算一致、分类为负结果。
A6 /data0/lzy/ZAPS-results/diag_fresh_refinement_confirmation/A6_20261020，必须六例完成且源plan、run/PRE/trajectory/观测身份通过。
基于master 5b9d086，实际checkpoint/config/grid/源身份由父plan继承，冻结到新plan，配置内容完整保存。
FFHQ SR4、unit-domain sigma_y=.05、legacy_bicubic、rho3 30×10、DDPM fixed variance eta1、原clamp、原PRE ζ/D和原模型权重不变。
时刻与参数索引分别t14/k28/index1，t54/k27/index2。只加载已有输入，不训练、不采样suffix、不改正式网格。
GT原图不加载；已有A6终局PSNR/SSIM/LPIPS/loss收益随记录原样附列，仅作离线上下文，不用于本轮判定或选择。
中心clean/q对缓存相对误差≤1e-6；两中心clean/q重复≤1e-7；loss重复保存；外部CPU/可见CUDA RNG及tau/zeta/D不变。
o与q相对内积绝对值≤1e-6，RMS相对误差≤1e-6；不抽采样噪声，禁用额外randn_like。
torch安全CPU加载，UNet每次forward前计数并持久化；不是只在成功结束后计数。失败/中断保留记录和已知调用下界，不自动补跑。

## 预算与预声明结果

每层每例2+2方向×2尺度×2符号=10诊断NFE，六例60；两层120，新增方法0，无warmup/训练/隐含score。
成功全跑队列成本方法3600/诊断2004=5604，原早期单图另列。CPU检查、归档不调用网络。
每例两方向都可分辨且尺度一致时，两尺度gain_q均≥1.1*gain_o记q更敏感；反向同理；其余均保留。
单层至少2/3图两个seed同为q更敏感，标GUIDANCE_ALIGNED_RESPONSE_CANDIDATE；
反向至少2/3图一致标ORTHOGONAL_RESPONSE_LARGER_CANDIDATE；否则NO_COHERENT_DIRECTION_SPECIFICITY_ESTABLISHED。
这是事先工程读数，不是显著性或算法收益。正交更敏感也不等于物理方向无用。
未分辨或尺度不稳须单列；不扩尺度、换随机方向或改时间点追支持，不自动拟合控制器。
通过只解锁“该固定t读数与明确干预收益的关系”后续设计；跨步指标失败与固定时间机制不混为同一结论。

## 并行及归档

一段命令bash run_a11_a12_20261020.sh。默认GPU0/1各一进程、各加载一份模型，CPU线程2；
仅一张可见GPU则顺序跑；指定同一GPU也顺序跑。两张卡均需至少6GiB可用才并行，否则明确停止，不影响别人的作业。
公共父工件只在启动准备阶段审计一次，各worker不做递归历史准备；任务各持独立锁、目录和日志。
已完成严格复用，失败/中断不覆盖。父进程等待两条结束再统一打印摘要、串行归档/推送，避免Git索引冲突。
信号只终止本启动器创建的两个子进程。没有新完整重建结果，不报告本轮提升了PSNR/LPIPS。
