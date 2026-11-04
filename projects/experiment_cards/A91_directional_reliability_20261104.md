# A91：独立方向可靠度收缩，不依赖全局标量

## 用户要求、可否定目标与父记录

直接尝试方向可靠度收缩；新控制器不以全局标量为中心、预算、参考或回退。
同一固定面板：原3 FFHQ图00226/00443/00963，四任务，sigma=.05/.10；不新增图/seed/噪声档，不扫描规则。
只比较BASE_CHECK原ZAPS、SCALAR_CHECK既有规则、DIRECTIONAL新方向规则。标量仅为实验对照。
使用已完成A71同一前九轮/ζ/D/init/RNG，在最后30步改变控制器；SR沿source_record的SHA解析回A70实际二进制。
A89/A90历史源、SHA绑定结果和旧判定全部不改；新目录独立。

## 统一符号与原ZAPS增益

W是原代码db4三级analysis，W^T为synthesis；原level3/周期边界和10子带布局保留。
T为代码实际使用的测量反投影映射：Gaussian/Motion/mask沿用原映射；SR沿用历史bicubic经验映射，不冒称精确H^T。
r_t=y-H(m_t)，a_t=sqrt(bar_alpha_t)，v_t=T r_t。
q_t=ζ_t/a_t [v_t+(1-bar_alpha_t) W^T D_t W v_t]。
定义ell_t=(ζ_t/a_t)[1+(1-bar_alpha_t)diag(D_t)]，则Wq_t=ell_t ⊙ W T r_t（正交代数）。
原ζ/D已经学习，直接保留；新模块不做响应投影，不取消ζ的幅度。
D带符号不等于协方差；这里只把它作为原线性滤波器的一部分，噪声传播使用ell²，故方差非负。

## 每方向噪声能量的独立计算

局部工作模型：冻结当前m_t、ζ_t,D_t，考虑纯测量噪声epsilon，Cov(epsilon)=sigma_y² I。
其进入原q的噪声系数为ell_t ⊙ W T epsilon。
令V_j=[diag(W T T^T W^T)]_j，则噪声能量

N_{t,b}=sigma_y² sum_{j in b} ell_{t,j}² V_j。

V与数据、sigma、时间和ζ/D无关，只依赖固定T/掩码及W。
用K=16独立unit Rademacher探针xi_k，E[xi xi^T]=I，估计

Vhat=(1/K) sum_k (W T xi_k)²。

此恒等式来自线性噪声传播；是矩阵自由方差估计，不是对真实残差风险的无偏证明。
独立CPU Generator，seed=SHA256(A91|operator_key)前4字节；不消费实际采样RNG。
SR/Gaussian/Motion各共享一个bank；补洞3个固定mask各一个bank，共6bank、96次T、96次analysis，0UNet。
共享unit variance后，sigma²和ell²在线更新；保存前8/后8探针估计的差异作精度诊断，不按重建质量改变K。
每24个已缓存entry另用同一纯算子探针检查首末两个t的Wq与ell W T xi相对误差<=1e-5，共24T、72analysis、48synthesis；0UNet。

假设边界：m_t和已学习参数实际上依赖y。当前N仅是冻结这些量时的纯测量噪声贡献，
不包含这种数据依赖引起的全部噪声协方差，也不单独估计扩散去噪预测误差。
称“方向可靠度代理”，不称校准的后验不确定性、无偏完整图像风险、SURE或误差置信区间。

## 每方向信号估计与连续收缩

E_{t,b}=sum_{j in b}|(Wq_t)_j|²，Shat_{t,b}=max(E_{t,b}-N_{t,b},0)。
w_{t,b}=Shat/(Shat+N)=(1-N/E)_+；E=0时w=0。
最终Delta_t=W^T[w_{t,b(j)} ⊙ Wq_t]，x_next=x_DDPM+Delta_t。

10个权重各自从本带能量与噪声推导，RGB/同带位置共享；没有global omega、gamma_proj、scalar anchor或scalar球预算。
w是[0,1]的信号保留比例；方法是对native q的纯收缩，不再额外放大全局引导。
因此和旧SCALAR比较的是两个独立规则，不是同幅度预算的方向消融。
若ζ整体乘c>0，则E,N均乘c²，w不变而Delta乘c；原学习幅度不被投影抵消。
sigma增大时各带因N/E不同而分别变化，不是整体提出同一个omega。
不是人为将最终gamma卡在0..2，也没有beta/lambda/时间表/固定倍率回退。

## 参考与推导来源

原ZAPS Eq21/22与原算法：db4近似和学习ζ/D，未修改原实现。
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114.pdf
Wiener/LMMSE第二矩估计：理想不相关signal/noise模型下，固定w的风险
R_b(w)=(1-w)² S_b+w² N_b，最优w=S_b/(S_b+N_b)。本轮用能量矩匹配估计S，不能继承oracle最优风险结论。
https://www.epfl.ch/labs/la/wp-content/uploads/2018/08/EstimationTheory.pdf （Linear MMSE/Wiener smoothing）
随机探针只估计固定线性映射的噪声二阶矩，E[(Axi)²]=diag(AA^T)，不做网络Jacobian探针。
相关经典背景：Bekas/Kokiopoulou/Saad2007 matrix-free diagonal estimation；本轮使用矩形映射噪声传播恒等式，
不是照搬其z⊙Az估计公式，也不继承给定K的相对误差保证。
https://experts.umn.edu/en/publications/an-estimator-for-the-diagonal-of-a-matrix/

## 预算、验证与停止

24案例×3臂×30=2160新增诊断F；warmup每载入另1F；0输入VJP、0优化反向；方法完整计算仍300F。
一次性96+24=120原T映射、6次H取形状；标定与诊断transform成本/时间分开记录。
在线新控制器不加H/T或UNet，不构造10个响应向量、无QR、无NNLS；额外1次analysis(q)、最多1次synthesis。
权重全1精确使用native q、全0精确zero；其余正交重构误差有界。约束的是native q收缩，不看SCALAR能量。
额外小波代价仍需实际计时，不因0额外NFE宣称免费。
本机7项NumPy/CLI检查；服务器强制真实Torch校验：mask总噪声能量、独立RNG、噪声传播恒等式、幅度齐次性、
无全局select调用、30F/0反向/最终RNG、native范数上界。缺依赖失败不skip。
BASE和SCALAR先与原paired.pt相对L2<=1e-6，再新规则；任何非有限值/身份/门控失败保留并停，不质量回退、不自动失败重试。
GPU>=6GiB，在本机无Torch/CUDA条件下不代跑；用户手动执行；不终止其他进程。
GT只用于最终PSNR/SSIM/LPIPS、补洞observed/missing MSE；不入E/N/w或探针选择。
每任务/噪声对两个参照分别报告均值、3图胜数、SSIM/LPIPS与时间。
预注册正向：PSNRmean>=+.1dB、>=2/3图胜、SSIM不降、LPIPS不升；混合/负向完整保留，不用全局倍率救分。

## 输出与命令

/data0/lzy/ZAPS-results/diag_directional_reliability/A91_20261104
summary.json、comparison.csv、逐图输出/trace、独立bank和标定日志；大PT只留服务器SHA索引。
results/zaps-directional-reliability-20261104自动轻量上传，不合results进master。
bash projects/run_a91_20261104.sh --gpus 0,1
--status只读；资源暂停原命令续未开始部分；失败保留，--archive-only仅重试汇总/上传。
