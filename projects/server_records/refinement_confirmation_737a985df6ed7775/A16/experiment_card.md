# A16：一次额外预测上的末端状态修正强度

状态：planned。2026-10-20，A线，exploratory local_reference。

## 1. 目标与依据

可否定假设：一次额外网络预测提供的先验位移与物理修正，其冲突程度可用于
约束修正强度，在 PSNR 优先的保护线内优于同成本固定系数。
A14 的 24 个状态读数均可用；A15 在两层的 PSNR 增量预测有效，但原联合判据
均未通过。预测 MAE 改善不是重建 PSNR 改善。本实验是新的廉价控制假设，
不把 A15 的十调用敏感度读数直接部署，也不把本实验余弦当作相邻步余弦。

父源：已完成 A9 `/data0/lzy/ZAPS-results/diag_terminal_polish_control/A9_20261020`，
其 SHA 绑定的 A6 原始终点、PRE、测量与计划；A14/A15 仅作已完成的动机证据。
复用 00023–00025 × seeds 1000/2000，三个已见图像簇。无训练/验证独立性主张。
A9 原 S 平均 PSNR +0.131696、SSIM +0.003886、LPIPS +0.002644，按用户本次容差
可接受，但这是回顾判定。保留历史原联合判定。本次不扫描终端重复次数。
A2c 的中间步半剂量与此处原终点的一次新预测具有不同作用位置，不重跑 A2c。

## 2. 唯一因子与固定条件

从原终点 C 以原参数 index=0、t=0 额外预测一次，得到 clean 与原预条件修正 q。
令 p=clean−C，r=y−H(clean)，a=H(clean+q)−H(clean)。H 为仿射算子，禁止用 H(q)。
物理系数 L：alpha_L=clip(<a,r>/<a,a>,0,2)，分母 ≤1e−20 时用 0 并标记未分辨。
方向约束：若 <p,q><0，alpha_cap=min(2,0.5||p||²/(−<p,q>))，否则为 2。
实验臂 A=clean+min(alpha_L,alpha_cap)q，保证保留至少一半沿 p 的投影进展。
这是位移方向约束，不是 Hessian、轨迹曲率或真实梯度认证；q 沿用 legacy 预条件约定。

同一次网络输出构造 F0/F05/F1/F2=clean+{0,0.5,1,2}q；L=clean+alpha_L q 为
去掉方向约束的物理自适应消融。F1 与原 terminal_output 完全一致；另列历史 native S。
所有臂在读取 GT 前固定并存盘，GT 只用于离线指标，固定臂不会按每图 GT 选输出。
所有新臂均 IEEE FP32，同一中心；原 C 与历史 S 保留 native，不混作 IEEE 中心。

其余固定：继承 A9/A6 的 config、checkpoint SHA、30 步网格、ζ/D PRE、原测量和图像
SHA、FFHQ SR4 unit_operator+legacy_bicubic、sigma=.05、原 clamp 与 uint8 PSNR/SSIM11、
Alex LPIPS。计划显式绑定相关源码/张量/配置 SHA。Torch/CUDA/cuDNN/GPU 与 A13 一致，
记录精度策略，退出还原；不改训练、网格、共享核心、默认推理入口。

## 3. 预算与门控

新 method NFE=0；每例同输入重复两次，6例共12 diagnostic NFE（科学动作部署时每例+1）。
原方法成本另列为每例300；不把新动作算成免费。两分片默认 GPU0/1，一卡时顺序。
调用前持久化计数；不允许噪声抽样；repeat clean/q/output 相对误差 ≤1e−7，
F1 与真实 terminal_output ≤1e−7；参数/网格/RNG保持。仿射损失模型核对容差
1e−7+2e−5|loss|。C 对旧指标漂移 ≤.01dB/.001/.001；物理损失同上述容差。
每例保存精度/runtime、alpha/cos/norm、cap是否激活、各臂图像/指标/损失/成本。
失败、未完成保留，不自动覆盖或重跑。父源准备时核验一次；分片只核验所需源和代码。

## 4. 预先固定的判据和分支

用户明确选择：平均 PSNR 正向，平均 SSIM 降幅≤0.002，平均 LPIPS 升幅≤0.005。
另要求平均物理损失不超过 C（数值保护线内），逐图退化均列出。
数值保护线为10×repeat与历史C漂移的最大值，并设 PSNR 1e−6、SSIM/LPIPS 1e−7、loss1e−10 底限。

实际重建可接受：对 C 满足上述条件，PSNR 增益超过数值保护线。
状态强度候选：A 可接受，且对每个可接受固定系数臂及历史 native S 均有超数值保护线
的平均 PSNR 增益、SSIM/LPIPS 平均退化在同一容差内，且各比较至少2/3图像簇的
配对 seed 均值 PSNR 正向。固定臂是否可接受、全部差值均报告，不用 GT 在线挑选。
方向约束增量候选另要求 A 优于 L（同样 PSNR/质量及图像簇门槛）并实际激活 cap。
若仅固定系数有效，结论只记固定修正候选；若 A=L 或更差，不能宣称余弦控制有效。
若不支持，停止本系数区间/0.5投影约束规则，不扩区间/换保护比例追分。
所有结论仅限三幅已见图像探索，不解锁泛化或毕业论文创新已完成结论。

## 5. 执行与归档

`bash run_a16_20261020.sh` 一次完成必要 Torch 检查、准备、分片、等待、摘要与归档 push。
新目录 `/data0/lzy/ZAPS-results/diag_terminal_state_strength/A16_20261020`；不覆盖父实验。
完整张量留服务器，轻量记录提交 `results/refinement-confirmation-20261020`。
本机仅 NumPy/语法测试；真实 Torch/CUDA 门控待服务器执行。
