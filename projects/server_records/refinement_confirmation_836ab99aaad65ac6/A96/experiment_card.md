# A96：A95有限迭代向量的方向/长度交叉归因与超分定位

状态planned，A工作线；用户同意先完成方向/长度归因，同时定位正常噪声超分退化。
A95R2已48/48，三个任务画质正向，超分PSNR/SSIM略降而LPIPS改善。矩阵均为20次外迭代预算终止。
本轮研究其实际产生的固定向量，不把有限迭代向量当充分收敛最优解；不重做求解、不增加图/噪声/步数。

## 公平性与作用范围

父 /data0/lzy/ZAPS-results/diag_response_target/A95R2_20261106，须revisionA95R2、completed48/48。
原3 FFHQ00226/00443/00963×四任务×模型[-1,1]sigma=.05；原A71九轮学习ζ/D、原30点15/10/5、DDPM eta1、clamp、IEEE、原H与y全部不变。
同A92单位置t333->s309，当前DDPM创新已固定，之后14次原ZAPS续推；本次标量和矩阵都是单位置，具可比性。
与历史“末轮全部30步标量”比较完整方法，后续必须让两种方法作用于相同30步。本轮不混用这两种作用范围，也不自动推广。
原ZAPS及直接物理方向对照保留A92绑定引用，不与原文大样本均值混比。

## 五臂与几何定义

q0为原ZAPS该处修正，S为A95真实响应标量修正，D为A95真实响应矩阵修正。
uS=q0/||q0||是非负标量的原射线，即使S=0仍定义；校验S确实与q0非负共线。
uD=D/||D||，LS=||S||，LD=||D||。

- SCALAR_REPLAY：S=uS*LS。
- DIAG_REPLAY：D=uD*LD。
- DIAG_DIR_SCALAR_LEN：uD*LS，只改变方向，与标量等长度。
- SCALAR_DIR_DIAG_LEN：uS*LD，只改变长度，与矩阵等长度。
- ZERO：该处修正0，之后原ZAPS续推，分离自然去噪与当前干预贡献。

S=0时短长度交叉严格为0、长长度沿原q0射线；D=0但LS>0则方向未定义，停机不替代。
所有动作仅由已付费缓存向量决定，不读GT，不重新选倍率/方向/最好图，不将某个长度设为新算法上限。
两个原臂重放输出必须相对L2<=1e-6，尾CPU/CUDA RNG必须与A95及A92一致；门控失败不解读交叉结果。

四格指标记YSS/YDS/YSD/YDD。分别报告：
direction@LS=YDS-YSS；direction@LD=YDD-YSD；length@uS=YSD-YSS；length@uD=YDD-YDS。
interaction=YDD-YDS-YSD+YSS。PSNR/SSIM/LPIPS分别计算，不做混合总分，LPIPS下降才好。
逐图与任务均值都保留，方向和长度可能有交互，不能强行分成两个可加百分比。
方向支持的工作门槛：匹配长度下PSNR均值>=+.1dB，至少2/3胜，SSIM不降、LPIPS不升；混合结果照实报告。

## 超分定位（所有GT诊断均在动作和14F完成后）

1. 长度：短矩阵方向不退化而长标量方向退化，支持幅度贡献；反向模式支持方向贡献；用四格和交互完整判断。
2. 测量风险：下一干净预测的physical_gain=clean_measurement_gain+<真实测量噪声,response>，区分真实信号改善与噪声拟合。
3. 内容风险：原db4/三级正交误差能量分解，记录LL3与9个细节带对整图MSE的贡献，sum恰为raw MSE（浮点容差1e-5）。
   高频/细节不是H的严格零空间，不能将其当不可观测成分的精确分解。
4. 时间传播：保存原全局index16(t309)、24(t119)、29(t0)的干净预测及最终输出。
   每阶段记录raw/clamped MSE、clamp比例、低频/细节误差及H预测对H(GT)的MSE；最终另报原PSNR/SSIM/LPIPS。
   检查误差在当前下一预测已出现，还是后续原ZAPS传播中放大；这不是局部物理下降必然改善终局的假设。
5. 每个案例完整五臂输出PNG与诊断记录，不按哪张好选例。旧输入、ζ/D和源字节不修改。

## 实现、预算与门控

源/父plan/summary、S/D correction/output/RNG、A92状态、GT/y/mask及实际已加载根guided_diffusion源码SHA绑定。
本轮无输入自动微分、无JVP/LM，不走此前有问题的forwardAD接口；NN权重严格重载、冻结，调用原native_parts。
12案例×5臂×14=840新增诊断UNet前向，0输入VJP/0JVP/0优化反向；每次模型载入warmup1F单列。
阶段快照从同14次前向中取，不额外调去噪器。离线H/小波/LPIPS另记；不把已有A95求解成本称免费。
Mac只做NumPy/CLI/语法检查；服务器强制真实Torch续推等价/RNG/14F/无导数、正交误差分解及两个真实父臂重放门控。
GPU手工分配范围内>=6GiB，warmup峰值+1GiB后逐项准入；不结束任务、不降精度/分辨率、不下载权重。
失败/中断记录保留，不自动失败重试；资源暂停只续未开始部分。完整14F不是独立重建或完整300时钟。

## 后续决策

方向与长度均支持：保留各自贡献和交互，再讨论共同收敛预算及相同末轮30步。
主要来自长度：优先研究标量长度目标/实际响应，不把收益全部归为矩阵；保留矩阵方向证据。
超分主要细节风险/目标错配：先修目标适配，不靠统一增加迭代保证正向。
所有任务/案例完整报告，不按GT挑某交叉臂作为本文最终方法，不由本轮启动全程控制。

bash projects/run_a96_20261106.sh --gpus 0,1
/data0/lzy/ZAPS-results/diag_direction_length/A96_20261106
results/zaps-direction-length-attribution-20261106
断线安全，逐臂进度，简明归因表，完整JSON/CSV/PNG轻量上传；snapshot大PT仅留服务器。
--status只读；--archive-only仅汇总上传。
