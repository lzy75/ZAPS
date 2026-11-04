# A89：由标量倍率扩展为四频带在线对角反馈

## 动机、范围与可否定假设

全局一个倍率把低频结构与不同方向细节统一缩放，可能限制观测修正。
检验四个在线系数是否优于同图原标量，而不是用增加复杂度本身解释创新。
先用 A71 同一前九轮缓存的 ζ、D、初始状态和随机数，在最后30步比较：
BASE_CHECK（原 ZAPS） / SCALAR_CHECK（原收缩） / DIAG4（新四频带联合解）。
3张原FFHQ图00226/00443/00963 ×4任务 ×sigma=.05/.10；所有24条件保留，72个后缀。
原ZAPS完整重建仍为10×30=300F，此次不重跑270前缀，不把30步后缀时间当完整时间。

## 保留的原方法与符号

模型坐标[-1,1]，r_t=y-H(m_t)，a_t=sqrt(bar_alpha_t)，bar_alpha_t为累计信号系数。
原学习后的引导 v_t=H^T r_t，
q_t=ζ_t/a_t * [v_t+(1-bar_alpha_t) W_db4^T D_t W_db4 v_t]。
原ζ_t、逐系数db4对角D_t、DDPM网格/方差/噪声、m_t实际clamp与返回规则全部保留。
A71的原始标量使用 d_t=H(m_t+q_t/a_t)-H(m_t)，与线性H的H(q_t)/a_t等价。
补洞统计仅观测RGB，M为有效标量观测数；同一原始y和mask保留。

## 目标的延伸

omega_t=max(0,1-M*sigma_y^2/||r_t||^2)。这是原规则的工程扣噪权重，不是无偏风险估计。
原连续倍率为 gamma_t=argmin_{gamma>=0} ||omega_t*r_t-gamma*d_t||^2，
即 gamma_t=omega_t*max(0,<r_t,d_t>)/||d_t||^2，零响应沿用原select边界。

令 W_H 为固定单级正交Haar变换，按 LL /列差分 /行差分 /对角差分四频带，RGB共享倍率。
P_b=W_H^T E_b W_H，sum_b P_b=I，P_b P_c=0（b!=c）。
Gamma_t=W_H^T diag(gamma_{t,1} I_1,...,gamma_{t,4} I_4) W_H。
所以Gamma仅在Haar坐标是对角阵，在像素坐标通常非对角；它会改变原q的方向。
新状态 x_next=x_DDPM+Gamma_t q_t=x_DDPM+sum_b gamma_{t,b} P_b q_t。
若四个系数相同，则严格退化到标量修正（浮点重构误差另核对）。

B_{t,b}=H(P_b q_t)/a_t；在补洞仅取观测RGB。
解 gamma_vector>=0 的 min ||omega_t*r_t-B_t gamma_vector||^2。
G=B^T B，h=omega_t B^T r_t；保留完整4×4 Gram的交叉项，不独立除四个分母。
H掩码、zero-boundary模糊不与Haar投影交换，因此频带在观测域可能耦合。

数值上用 n_b=||P_b q_t||，z_b=n_b*gamma_b，U_b=B_b/n_b。
对最多4个变量枚举16个非负活跃面，双精度伪逆rcond=1e-12，检查KKT。
若最小化解不唯一，选 ||z-z_scalar||^2 最小者，其中z_scalar_b=n_b*gamma_scalar。
这是修正图像到原标量修正的最小L2偏离；观测不可辨认的零空间不被任意清零或补为新先验。
零q频带系数设原标量（其修正本来为0），不构造除零；其他数值失败停机，无策略回退。

## 参考与新颖性边界

依据：原规则代码noise_signal_shrink.select；原ZAPS小波近似见modules/wavelet.py与zaps_algorithm.py。
频带分解与非负最小二乘是标准数学工具，此处目标是既有标量目标的嵌套扩展，非原创性证明。
例如Cai/Zhou波段自适应阈值 https://arxiv.org/abs/0903.5147 是相关经典工具背景，
并非本模块的扩散/后验理论来源，未继承其SURE或统计风险保证。
原ZAPS的D_t本身已具各向异性；本新增Gamma是其后在线反馈的另一层，不可称首次引入小波对角阵。
拟合标量可行集包含在矩阵可行集内，只保证固定状态下代理拟合不差，不保证真实去噪器响应、PSNR或缺失区域质量。
Haar频带会把修正重新分布到邻近缺失像素；这不是估计到了不可观测内容，需单独报告区域误差。

## 成本、门控与决策

预算2160新增诊断F（72×30），0输入VJP、0优化反向；每模型载入另1次warmup。
DIAG4每步比原标量多4次H、固定Haar分析/重构和4维CPU NNLS；合计2880额外H、720个小求解。
没有正负扰动/下一去噪器探针，没有新训练参数/拟合lambda/beta/倍率上限。
结构设置仍有四频带与Haar基，数值设置有伪逆容差；不宣称完全没有设置项。
计后缀walltime、原H及新增H次数、NNLS CPU秒数、每步4个gamma及修正范数；完整300耗时后续确认才可报告。

必需门控：本机NumPy正交性/标量退化/NNLS耦合/零空间锚定/原SR缓存追溯测试；
服务器真实Torch原A71循环重放、30F、0反向与RNG测试。无Torch直接失败，不skip。
每案例先重放两参照，与原paired.pt相对L2<=1e-6，后运行DIAG4；最终CPU/CUDA RNG必须配对。
原SR缓存通过带sha的source_record解析至A70实际entry.pt/paired.pt，禁止声称A71复制了不存在的二进制。
实际FFHQ既有checkpoint严格重载、只分配>=6GiB空余GPU，不抢占/终止其他进程。
GT只进入最终离线PSNR/SSIM/LPIPS及补洞区域MSE，不用于求解/停止/挑图。

每任务/噪声，DIAG4相对SCALAR至少均值PSNR+.1dB、2/3图胜、SSIM不降且LPIPS不升，才记小面板正向；同时列对BASE差值。
混合/负向停止此固定四频带方案，不无限换基/调阈值。正向条件再做独立完整300和新图确认。
输出新目录 /data0/lzy/ZAPS-results/diag_diagonal_feedback/A89_20261104，comparison.csv、summary.json、每臂原始PNG/逐步trace。
断线安全、实时进度、自动轻量归档 results/zaps-diagonal-feedback-20261104；不合results到master、不下载PT。

## 命令

bash projects/run_a89_20261104.sh --gpus 0,1

--status只读；资源暂停后原命令续未开始部分；失败不自动重试；--archive-only仅汇总与上传。
