# A122 — 只适配对角模块，不修改已有标量与 ZAPS

状态 planned，2026-11-20，A线 / 小规模迁移机制验证。尚无真实 Torch/CUDA 结果。

## 目标、已有证据与可否定假设

DPS/ΠGDM 原标量在多数任务得到质量提升，旧对角的固定线性代理目标下降却未稳定提高重建质量。
ZAPS 上已有去模糊正向结果，本次不修改其设计，不重新选择标量规则。
假设：保持原标量和原修正向量，只将对角联合更新适配到各方法下一时刻的实际去噪器响应，能改善对角迁移。
若仍退化，响应失配不是充分解释，不增加图像、种子或任务参数扫描。

父记录：已完成 A75/A88 FFHQ、A85R1 FFHQ，以及
`/data0/lzy/ZAPS-results/diag_complete_diagonal_transfer/{dps,pigdm}_ffhq_20261120`。
读取原 plan/y/算子/权重 SHA；不修改原文件、不重造观测。

## 对照与固定项

FFHQ 原3图（00226/00443/00963）、原 seed、四任务：SR4/Gaussian/Motion/随机补洞；sigma_model=.05。
由父协议按原文件严格验证，固定模型[-1,1]域、算子/边界、原噪声与原采样网格/输出。
BASE 与旧 DIAGONAL 完成结果复用，明确历史时钟。SHRINK 原代码原样重放，先校验原输出相对L2<=1e-4。
MATCHED_DIAGONAL 是唯一方法改动。SHRINK 与 MATCHED_DIAGONAL 独立从 seed 执行完整原循环，同会话完整报时。
其余97%位置保持旧标量规则；只有进度20/50/80%介入：DPS第200/500/800步，ΠGDM第20/50/80步（1起算）。
没有修改时间步长、噪声、native q、DPS的梯度或ΠGDM的协方差方向；q已包含原拉回，不重复计算原q的拉回。

## 固定响应目标与本次对角更新

令原采样无引导下一状态为 x_s^0，各方法自己的干净预测为 C_s，物理算子为 H，P 仅取观测坐标。
DPS 使用原 sampler.p_mean_variance 的 clipped pred_xstart，ΠGDM 使用原 raw Tweedie clean。s=-1为恒等，实际三位置均不在端点。

```
h0 = P H C_s(x_s^0), r = P y - h0
omega = (1 - M sigma_y^2 / ||r||^2)_+
F(delta) = .5 || P H C_s(x_s^0 + delta) - h0 - omega r ||^2
```

保留原噪声扣除启发式；并不把它解释为无偏风险或真实噪声/信号的可识别分解。
原标量 gamma 的计算完全不变，用其候选 gamma*q 与原q作为可行起点，用同一个F选较低者；不使用GT。
此处只用于对角求解初始化，不产生一个新的标量方法臂。

```
c=Wq, u=q/||q||, P_perp=I-uu^T
M=P_perp W^T diag(c)
v = -M M^T grad_delta F / ||M M^T grad_delta F||
delta(alpha,beta) = delta0 + alpha*u + beta*v
```

W为原db4/level3，每个系数允许有符号增益，无四频带共用限制。M是归一化方向对乘性增益的雅可比。
因此 c 出现两次，不把自由小波系数的梯度当作乘性倍率梯度。
用一次真实VJP生成整体增益空间方向，两个JVP计算 Ju/Jv，float64的2x2 Gauss–Newton子问题联合解 alpha/beta。
总长度与合成方向同时变化，不先固定标量长度、不自动保留相同原方向分量而强迫长度增加。
这是全系数方向的一次联合子空间更新，**不是完整对角变量的精确最优解**。
仅用机器精度确定伪逆秩；无新可调beta、正负微扰、0/1/2网格或无限线搜索。
实际F检查原标量可行点、连续射线候选及连续联合候选，只接受不增F的候选。
同一状态下对角接受值不高于其射线候选；不保证完整轨迹PSNR/SSIM/LPIPS或缺失区域质量不降。
额外射线候选只是内嵌可行方向检查，不独立运行“新标量”实验。

## 预算、门控与报时

每次介入8个真实predictor前向：无修正下一状态、原q、旧标量、初点VJP、两个JVP、两个连续候选。
因此每图原标量DPS=1000F/1000inputVJP；新对角=1024F/1003inputVJP/6JVP。
ΠGDM原标量=100F/100inputVJP；新对角=124F/103inputVJP/6JVP。
绝不声称与旧标量等NFE；新增计算是此轮响应适配的一部分。
每方法12图任务组合，24个新完整重建；另复用24个历史结果。DPS新增24,288F；ΠGDM新增2,688F。
单测/真实权重准入/被拒候选开销单列，失败计数保留。无输入优化反向或GT调参。
主结果完整时钟含y读取、算子构造、初始化、采样、控制器、CPU输出同步和进度；模型加载、准入、指标、图片写盘另列。
新旧对角直接报完整开销差异；历史BASE时钟不作同会话加速证据。
门控：单CUDA、FFHQ加载前空闲>=6GiB，严格reload原权重，原NN参数/grad不变。
强制原样采样接口测试、解析VJP/JVP测试、乘性增益链式法则测试、原标量复现与原网格/初末RNG配对。
旧自定义checkpoint只在AD探测作用域绕过，随后恢复；真实权重JVP准入先于实验。
数值/门控失败立即停并保留记录；不自动重试、不把失败当作重建样本。

## 判定与后续

每方法/任务对原基线、原标量和旧对角分别报告PSNR/SSIM/LPIPS、逐图胜率、NFE/VJP/JVP、完整时间。
正向门槛沿用：均值PSNR>=+.1dB、至少2/3逐图胜、SSIM不降、LPIPS不升；同时展示所有取舍。
若超分/补洞仍退化，保留原标量，分析实际目标与全图风险、方向/长度接受情况，不宣称通用迁移已成立。
若某任务正向，只支持这3图/1seed/.05/三个介入位置的适配机制，再决定是否推广全程。
不把机制实验结果冒充ZAPS论文均值复现或硕士论文完整创新证明。

## 命令与输出

```
bash projects/run_matched_diagonal_transfer_20261120.sh dps --gpu 0
bash projects/run_matched_diagonal_transfer_20261120.sh pigdm --gpu 1
```

结果目录 `/data0/lzy/ZAPS-results/diag_matched_response_transfer/A122_{dps,pigdm}_ffhq_20261120`。
独立日志/plan/原卡/session/summary/每图run+trace+PNG+PT；轻量JSON归档推送 results/{方法}-matched-diagonal-ffhq-20261120，PT留服务器。
当前只完成代码/可执行入口准备；真实数值测试与重建需要服务器执行。
