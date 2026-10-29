# A70：强ZAPS上的噪声校正残差投影与实时匹配检验

2026-10-29；A线；planned。独立入口，A67–A69/原ZAPS源码不修改。

## 目标、父记录与可否定假设

A69完成40/40：小波近似梯度两档均退化；真实梯度高噪声有收益但不及首次噪声交点。
首次噪声交点为经典discrepancy原则的射线实现，实时读取r/d不自动证明匹配有价值或公式原创。
本轮检验：无需额外NN/输入反向的连续残差收缩，是否改善强ZAPS，并相对冻结固定/错配倍率有增量。
父 /data0/lzy/ZAPS-results/diag_scalar_gamma_solver/A69_20261029，代码741c02c；必须40/40 complete。
复用旧四图的BASE末轮入口和首次交点trace以冻结参照，不复用确认图结果。
不是恢复A37向倍率1的正则、旧cos/ridge、Nyström或JAPS向量；新的因素是先估计应校正的残差，再投影到原ZAPS引导。

## 构造与含义

保留原ZAPS的clamped干净预测m_t、学习参数ζ/D和引导q_t。
r=y-H(m_t)，d=H(m_t+q_t/sqrt(alpha_bar_t))-H(m_t)，M为观测标量数。
d沿用A33同t identity-response代理，不是A69的小波近似梯度，也不称真实下一网络响应。
工作模型r=s+n，s~N(0,tau² I)、n~N(0,sigma_y² I)：

    tau_hat²=max(||r||²/M-sigma_y²,0)
    omega=tau_hat²/(tau_hat²+sigma_y²)=(1-M*sigma_y²/||r||²)_+
    s_hat=omega*r
    gamma=argmin_(g>=0) .5||s_hat-g*d||²
         =omega*max(<r,d>,0)/||d||²

解释：omega估计有多少残差值得当作信号校正；投影决定当前原引导对应多大倍率。
连续非负，没有固定上界2，没有倍率1候选/1%回退，没有迭代或新β/eta/lambda。
零响应/零残差取最小位移gamma0；非finite轨迹失败，不按GT换输出。
这是经典Gaussian收缩/经验方差估计加一维投影的工程组合，不宣称新统计定理。
m_t依赖y，因此噪声独立、各向同性及tau估计仅是工作模型；不是精确后验、不确定性校准或SURE。
扩散噪声通过当前预测及1/sqrt(alpha_bar_t)响应尺度进入；不另称测得实时后验不确定性。
当前ζ正幅度可能被连续重标消去；ζ学习历史、D和q方向仍保留。
原FIRST保留A67已验证的0..2、不可行时有界LS回退，单独标识，不伪称完全同约束。

## 因子、数据与固定条件

父校准图00000–00003，sigma .05/.10，seed1000+ID；只用trace，GT不进常数。
确认4张可访问case/plan历史中未用FFHQ，范围32..999，按固定seed20261029排除名称/内容重复后选取。
候选池不算实验使用，actual/failed/planned case仍排除；缺4张即停，不结果后替换。
确认每图sigma .05/.10、seed1000+ID；共8条件、4独立图簇，非跨数据集/多任务验证。

|组|目的|
|---|---|
|BASE|同场强ZAPS，倍率1|
|FIRST_NOISE|首次噪声交点实时策略，保留当前正向参照|
|SHRINK|新噪声校正残差投影|
|FIXED_FIRST / FIXED_SHRINK|分别用旧4图x30位置全部倍率的中位数，按已知sigma冻结一个常数；不是时间表|
|MISMATCH_FIRST / MISMATCH_SHRINK|分别循环替换成另一张确认图、同sigma同slot的实际实时倍率轨迹|

映射在prepare中按图ID循环偏移1冻结；所有反馈轨迹完成后只读取gamma，不读质量标签。
错配保留每档倍率集合/时间位置，破坏图像状态匹配；常数组检验普遍减弱引导是否已经足够。
这两个对照分别解释匹配价值，不能仅FIRST/SHRINK比BASE好就当作状态创新。
原9/17配置：SR4，30点15/10/5，10轮，lr=.01，ζinit=.1，Dinit=.2，db4 level3，
DDPM fixed-small eta1，原clamp/last_opt，经验bicubic引导，模型[-1,1]观测及同坐标sigma。
观测生成沿用A67函数，NN全部IEEE；前9轮原生，仅最后30步控制。不改步长/位置/总轮数。
GT仅观测合成/输出后的PSNR/SSIM/LPIPS评价，不参与常数、错配或在线决策。
不与A67六图28.186均值直接相减；本轮只作同图同sigma BASE对照。

## 预算、门控与时间

每次独立完整300前向、10次ζ/D优化反向、0输入VJP；每组从seed完整运行计warm-model时钟。
独立质量与同一BASE270参数/初始噪声/RNG的主要配对质量均报告；每次配对30前向另记diagnostic。
56次完整运行=16800方法前向；56x30=1680配对诊断；旧8条件SHRINK校准8x30=240。
总18720前向，warmup/模型加载另记；560优化反向，0输入反向。校准常数不隐藏为免费部署探测。
额外H动作、GPU峰值和完整时钟记录；算子开销不称零时间。
Mac纯标量/选择/真实子进程CLI检查，服务器强制Torch NULL/原FIRST等价/真实SR/预算/参数冻结检查。
BASE原输出与NULL相对误差<=1e-6；精度、checkpoint/图像/源码/父入口hash固定。
独立prefix差异另外记录，主要配对统一BASE入口；300-NFE完整耗时不以30步回放替代。
失败保留原attempt/成本，不自动重复；资源不足可同命令恢复完整case，已完成不重跑。

## 预先判读及后续

每档各策略相对BASE：平均PSNR>=+.05，>=3/4图正向，SSIM>=-.002、LPIPS<=+.005为质量工程信号。
对应实时策略还须相对各自FIXED和MISMATCH平均PSNR>=+.02、>=3/4图正向且满足同保护线，才记匹配信号。
全部逐图/最差/两个噪声档和对照都报告，不挑一个档或一个指标拼统一成功；4图不作总体显著性结论。
FIRST正向而新规则负向：保留FIRST机制，不扩收缩参数；固定/错配接近实时：收益主要来自调幅，不称状态独立创新。
有匹配信号再冻结第二seed及第二任务；无信号停止此构造，不扫上限/β/阈值。

## 执行

bash run_a70_20261029.sh --gpus 1；两张已分配GPU用--gpus 0,1。
依次校准/冻结常数→新图反馈→冻结错配→对照，一条断线安全队列，完成后集中摘要/归档push。
新目录 /data0/lzy/ZAPS-results/diag_noise_signal_shrink/A70_20261029。
