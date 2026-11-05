# A92：整体方向与长度、残差收益与噪声拟合的配对诊断

状态 planned；A工作线，机制诊断，不是新在线算法。用户明确参考“经过去噪器的物理下降方向”，不采用无条件score。
目标：检验在固定长度下，整体对齐该方向是否改善终局；以及残差下降是否实际在拟合测量噪声。
可否定假设：DIAG_ALIGN100相对NATIVE100，在每任务噪声条件下平均PSNR>=+.1dB、至少2/3图胜、SSIM不降LPIPS不升。
PHYS_DIRECT100检测原子带非负组合的方向限制；所有尺度完整报告，不选GT最优、不用结果重新设倍率。

## 父记录、既有证据与固定条件

父 /data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029，必须completed72/72。
沿source_record解析SR真实A70二进制SHA；同三张FFHQ00226/00443/00963、四任务、模型坐标sigma=.05/.10，共24案例，原seed/y/mask/kernel/init/九轮学习ζD全部不变。
Gaussian61/sigma3与Motion61原zero边界；补洞70%missing；SR历史bicubic经验反投影不冒称精确伴随。
原15/10/5网格30点、DDPM eta1、原clamp、原浮点/uint8指标；不增加图/seed，不改变时间间隔。
A89无约束子带拟合有巨大修正，A90标量范数预算在Gaussian有限正向但耗时高，A91独立噪声能量收缩弱于标量。
本次不重复上述连续控制；只在最后一轮固定第16次更新(zero-based15)介入一次，其余更新均原ZAPS。
不训练新参数，不更新NN；原文件/旧卡/失败记录不改。GT与未用图确认无关，本轮全是旧开发图。

## 方向与长度的完整因子设计

原步骤 uncond=DDPM(x_t,m_t)，q=原ζD修正。先将uncond记作z_s，已抽取的本步噪声固定。
真实物理参考g=-grad_z .5||y-H(clamp((z-b_s eps(z,s))/a_s))||²，在z=z_s求值，保留NN输入导数。
这是下一时刻局部物理损失下降方向，非原_detached_eps/surrogate梯度；SR的真实H反传也不使用legacy经验反投影。
raw前向与原下一步干净预测完全同值，clamp梯度保留。原custom-checkpoint参数临时requires_grad兼容，但无grad累积/权重更新。

固定原db4/level3十子带P_b，q_b=P_b q：
c_b=max(0,<q_b,g>/||q_b||²)，p=sum_b c_b q_b。
此为正交非负锥上的欧氏投影；归一化p后是在该锥中整体余弦最对齐g的方向，非按GT选分量。
三个整体方向：NATIVE=q/||q||；DIAG_ALIGN=p/||p||；PHYS_DIRECT=g/||g||。
每方向独立做长度L=.5/1/1.5乘||q||，共9臂；NATIVE100必须最先验证原输出。
原q范数仅为同长度因子诊断控制，不将标量规则/其能量设为新方法预算，不提出必须受原范数约束的在线算法。
若投影为零或真实梯度为零，标记该方向unresolved，不用其他方向替代。倍率仍为正的各分量缩放，不扫描频带/阈值。
后续14个原网格点均恢复原ZAPS修正；候选第一步网络预测同时作局部响应读数，不额外重算。
参考物理下降方向的锥投影是成熟几何计算，不声称原创。目的仅为方向/长度因果拆分，未实现SURE/风险最小化在线决策。

## 噪声分解与评价目标

所有候选完全预先指定，不使用GT决定动作。完成候选续推后，GT仅离线计算：
f0=H D_s(z_s)，f=H D_s(z_s+Delta)，d=f-f0，n=y-Hx_gt。
physical_gain=<y-f0,d>-.5||d||²；clean_gain=<Hx_gt-f0,d>-.5||d||²。
严格恒等式 physical_gain=clean_gain+<n,d>。
这直接检查“观测残差改善是否来自拟合噪声”；观测/缺失区域与全图下一干净预测误差、终局PSNR/SSIM/LPIPS同时报告。
补洞只在观测RGB统计物理收益及noise ratio，完整区域误差单列；GT只评价不参与投影/长度/输出选择。
该真实噪声分解只在有GT的离线诊断可用，不冒充未知图上的在线风险估计，不用M*sigma²代替实际噪声向量。

## 预算、门控与时钟

每案例共享原前缀16F、物理参考1F/1输入VJP，9候选各14F；24案例共3432诊断F、24输入VJP、0优化反向。
每次加载另外1F/1VJP实模型预热计费。原方法完整仍300F，本次为缓存学习前缀的单步干预，不报告独立30F或完整300时钟。
共享前缀/物理参考时间、候选14F续推时间、离线H/指标、写图、模型加载/预热分别记账。
每候选同原本步与后续创新；CPU/CUDA尾RNG一致；原NATIVE100输出相对L2<=1e-6，网格/count精确16+14；范数匹配相对误差<3e-5。
NN权重/grad/flags保留，原checkpoint真实VJP测试必过。有限值/数据SHA/旧源SHA/分解恒等式任一失败停止保留，无自动失败重试。
原三图同机器参考与新臂在同会话；固定原baseline先运行，无生产时延保证。
GPU需手工分配>=6GiB，加载后实际VJP峰值+1GiB准入；资源不足只暂停未开始部分，不杀进程/换精度/降分辨率。
Mac无Torch，只跑5项NumPy/CLI/汇总配对与语法检查；服务器强制真实Torch与custom-checkpoint检查，缺依赖不skip。

## 预期分支与停止

- 等长度DIAG_ALIGN100正向：整体方向约束值得继续；先看噪声拟合与终局是否一致，不直接升级成全程方法。
- DIRECT100正向、DIAG_ALIGN100负向：当前非负子带组合不能表达足够好的方向；不直接增加子带数量救分。
- 物理下降明显但clean_gain/终局负向：纯物理下降不够，应先解决风险校正而非扩大倍率。
- 等长度方向均不支持而某长度变化有收益：方向假设未支持，长度效应单独报告，不能将同图最优尺度算在线收益。
- 混合/幅度小：保留结果，停止该轮追加扫描；不新增图/次数。
- 门控失败：修具体接口/数值问题，不解释为方法效果。

## 命令与输出

bash projects/run_a92_20261105.sh --gpus 0,1
/data0/lzy/ZAPS-results/diag_overall_direction/A92_20261105
summary.json、comparison.csv、逐案例directions.json、每臂audit.json/图片/张量；大PT仅服务器，轻量SHA索引与结果自动push。
results/zaps-overall-direction-length-audit-20261105，不合results入master；断线安全入口持续打印到摘要。
--status只读；资源暂停后原命令续未开始部分；--archive-only只做汇总/上传。
