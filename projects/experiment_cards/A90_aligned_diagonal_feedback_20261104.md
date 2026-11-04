# A90：对齐原 db4 坐标的有界各向异性反馈

## 动机、假设、父记录

A89把原db4/三级另接单级Haar4带，且只拟合观测代理，没有修正能量约束。
Gaussian例00226/sigma.05出现高频gamma>2万、单步修正达标量13.5倍；更低代理目标没有带来更好图像。
本轮区分坐标选择和稳定性：BASE_CHECK / SCALAR_CHECK / HAAR4_TRUST / DB4_4_TRUST / DB4_10_TRUST。
HAAR4_TRUST仅测稳定化；DB4_4_TRUST在原三级db4坐标中把LL1内所有更粗尺度合成一组，另取LH1/HL1/HH1，保持4个增益以隔离换基作用。DB4_10_TRUST再展开10组，以隔离尺度分组数作用。
固定3 FFHQ旧图00226/00443/00963、四任务、sigma=.05/.10，全部24条件保留。
复用完成A71前九轮和paired入口。SR复用记录沿SHA绑定source_record解析回实际A70二进制。
原基线、原观测/mask/kernel、图像hash、checkpoint、tau、DDPM噪声、ζ/D全部保留。
目标：新db4矩阵在同修正能量上限下，是否超过标量，及是否优于同约束Haar。

## 原小波结构与矩阵大小

本地现用wave=db4、level=3、周期延拓；三级是当前实现设置，不冒称论文明确指定层数。
W表示代码analysis，W^T为synthesis；原D_t实际存[C,H,W]对角元素，不存稠密矩阵。
C=3,H=W=256时，图像向量n=196608；D_t和新Gamma_t均隐式n×n，而非4×4或10×10图像矩阵。
原每步D_t有n个可学习元素；新反馈只解10个共享系数，避免从单步观测拟合n个自由增益。
10子带为LL3及每级LH/HL/HH：LL3空间32×32；level3方向带各32×32，level2各64×64，level1各128×128。
RGB共享各带gamma，不额外改颜色比例；子带索引与原modules/wavelet.py打包完全一致。
10系数不是最优维数的结论，是直接沿用原三级结构的最低自然分组，本轮不扫描分组数/小波基。

## 统一公式与原ZAPS的对齐

bar_alpha_t累计信号系数，a_t=sqrt(bar_alpha_t)，r=y-H(m_t)，v=H^T r。
原q=ζ_t/a_t * W^T[(1+(1-bar_alpha_t)D_t) ⊙ Wv]。
原D_t带符号、未校准，不可直接当噪声方差/后验协方差，也不取D_t的逆构造置信度。
令E_b选择原打包布局中一个子带，P_b=W^T E_b W，q_b=P_b q。
新Gamma=W^T diag(g_1I_1,...,g_10I_10) W，Delta=Gamma q=sum_b g_b q_b。
在精确正交代数下，Gamma与原W^T D_t W可交换：
Delta=ζ_t/a_t * W^T[g_b ⊙ (1+(1-bar_alpha_t)D_t) ⊙ Wv]。
因此在线反馈是原学习结构的频带再分配，不另造不相容的坐标基，不重新学习D或ζ。
但共享基不等于获得真实Jacobian特征方向，也不使任意原D自动PSD。

## 噪声状态、响应与有界目标

保持原omega=(1-M*sigma_y^2/||r||²)_+及标量gamma_s；补洞仅取观测RGB，M为有效观测数。
同时间响应代理B_b=H(q_b)/a_t，原网络实际非线性响应不改称精确。
新解：min_{g>=0} ||omega*r - sum_b g_b B_b||²，
约束sum_b g_b²||q_b||² <= gamma_s² sum_b||q_b||²。
正交投影下右侧就是标量修正图像能量；实际浮点重构另验相对误差<=1e-5。
它限制整幅修正长度，不限制某个倍率在0..2。半径来源于同状态标量结果，无额外按图/任务拟合的半径因子。
此半径是预注册参考预算，不是已证明最优的半径；标量本身若有偏，约束仍会继承其局限。
只使用观测目标，无法直接估计缺失区域内容或整体重建风险。
标量gamma_s=0则球半径0，新矩阵严格为0；不以其他指标临时救分。

定义n_b=||q_b||，z_b=n_b*g_b，U_b=B_b/n_b，R=||z_scalar||。
求 min_{z>=0,||z||<=R} ||Uz-omega*r||²，是凸的带非负约束球域最小二乘。
完整跨带耦合保留，不假设H与小波投影可交换，也不假设H被db4精确对角化。

## 成熟求解方式及明确借鉴范围

1. ZAPS ECCV2024 §3.2 Eq21/22和§4.1：固定正交DWT近似Hessian，db4和原ζ/D学习。
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114.pdf
2. Moré–Sorensen, Computing a Trust Region Step, SIAM1983（作者1981技术报告§1–2）：
   二次模型的范数/椭球约束及对偶阻尼条件。这里只借用受限二次子问题框架，
   额外非负约束由成熟NNLS处理，不声称逐字复现GQTPAR或继承整条神经轨迹的收敛定理。
   https://doi.org/10.1137/0904038
   https://digital.library.unt.edu/ark:/67531/metadc283525/m2/1/high_res_d/metadc283525.pdf
3. Clason, Regularization of Inverse Problems, arXiv2001.00617v2，第5–6章：
   Tikhonov谱滤波，把1/s替换为s/(s²+mu)。仅解释弱响应放大如何被阻尼控制，
   其确定性噪声界/源条件等统计或收敛结论不适用于本模块未校准的实时残差。
   https://arxiv.org/html/2001.00617v2
4. SciPy官方optimize.nnls：标准非负最小二乘库，maxiter=100*p；p<=10。
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

先对U做float64 reduced QR，把观测域大问题压到<=10维R,c，不显式求逆(U^T U)。
若自由NNLS解已在球内则使用；否则对mu>=0解增广NNLS [R;sqrt(mu)I]z≈[c;0]。
随mu增加解范数不增，括定和二分至范数约束满足；mu是球约束对偶变量，不是预先调的质量超参数。
非负活跃集内的KKT形式：(R_F^T R_F+mu I)z_F=R_F^T c，配合非负乘子和互补性。
零响应时保留标量；一般秩亏的自由NNLS不承诺A89的“所有零空间分量最接近标量”性质。
达到64次括定/64次二分预算却未满足范数标准即失败；不冒称收敛、不输出回退策略。
KKT相对残差<=1e-6、norm预算容差1e-7；实际图像修正范数<=标量*(1+3e-5)。
SciPy版本冻结入plan，续跑必须同版本；Mac只在/tmp验证依赖，服务器不安装/升级。

## 预算、门控和正负分支

24案例×5臂×30=3600新增诊断F；每次模型载入warmup另1F；0额外UNet探针、0输入VJP、0优化反向。
矩阵臂每步额外最多4/4/10次H，总最多12960额外H，最多2160个小受约束问题；QR/NNLS耗时另记。
不把“无额外NFE”写成“无额外计算”。后缀walltime包含载入entry/y、算子、policy/QR/求解和输出CPU同步。
前270F缓存仅复用；完整方法仍300F，不能用后缀时长声称完整重建加速。
本机真实SciPy数值检查；服务器强制原Torch db4恒等/正交/标量退化、RNG/30F/0反向及实际能量约束检查。
缺依赖必须失败不skip。FFHQ既有严格checkpoint重载，>=6GiB分配GPU准入，不结束其他进程。
每案例先BASE与SCALAR重放相对L2<=1e-6，然后三个矩阵；旧结果、旧源、旧判定不改。
GT仅离线质量及补洞observed/missing MSE，不能调半径、阻尼、基、迭代数或输出选择。
每任务/噪声，相对SCALAR均值PSNR>=+.1dB、>=2/3胜、SSIM不降且LPIPS不升，记小面板正向。
依次对照：受约束Haar与A89同前缀结果比较识别稳定化；db4四组与Haar四组比较识别基；db4十组与db4四组比较识别尺度分组。没有同维数对照，不能把自由度变化归因于基对齐。
仍退化则保留原标量，不继续按任务扫lambda/子带数救分；优先查代理目标与真实响应/风险的偏差。
正向条件随后才做独立完整300和新图确认；本轮没有新图/seed/数据集。

## 输出与命令

/data0/lzy/ZAPS-results/diag_aligned_diagonal_feedback/A90_20261104
summary.json、comparison.csv、每图PNG与逐步子带g/阻尼/KKT/能量trace。
轻量分支 results/zaps-aligned-diagonal-feedback-20261104；不合results至master，不下载大PT。
bash projects/run_a90_20261104.sh --gpus 0,1
--status只读；资源暂停原命令续；失败不自动重试；--archive-only只汇总上传。
