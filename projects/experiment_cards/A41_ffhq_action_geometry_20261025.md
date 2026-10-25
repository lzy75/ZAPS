# A41：FFHQ 四任务的真实引导响应与终局画质

状态：planned，A 工作线。实现/CPU 检查不代表 GPU 结果。

## 1. 问题、父实验和可否定假设

A39/A40 的随机路径余弦、残差和近似物理响应未提供可靠的终局画质增量。
A40 STATE 的 q/sqrt(alpha) 响应近似还略去了 UNet 对状态变化的响应。
本轮测试：固定同一当前状态、下一时间和同一 DDPM 噪声，实际改变一次 gamma，
测量下一 clean 估计的响应。物理响应 P 与有限动作弯曲 I 能否比时间规则、
旧 P/I 更好预测 **最终** 画质，并在未参与拟合的任务/图像上选择倍率？

父目录 /data0/lzy/ZAPS-results/diag_multitask_state_transfer/A40_20261025。
先核验其 24 个 BASELINE/MEASURE/TIME 均 completed、签名、门控、调用序列及张量 SHA。
只读复制元数据到新目录；即使旧 launcher 仍持锁，也不解锁、不写旧目录、不重跑旧工作。
任何原 GPU 阶段未完成则停止。锁持有者 PID/命令只作诊断，不自动杀进程。

## 2. 固定协议、改动与对照

- FFHQ A40 的六张图：从已签名父 plan 读取，选择前锁定，没有质量筛选。
  6 图 × SR / Gaussian / inpainting / motion =24 cases，seed1000，sigma_y=.05。
  这是复用开发图的机制与迁移小试验，不是新图确认集；24 cases 只有6图像簇。
- SR/Gaussian 完整复用 A40 BASELINE/PRE/y；补洞/运动新增原 C 300 calls。
- 模型、grid、原优化、lr=.01、D=.2、db4 level3、DDPM、原始 clamp、last_opt
  均与 A40 FFHQ 相同；补洞 zeta=.1、运动 zeta=.2。不是 paper-spec 认证。
- A(x)=H((x+1)/2)，Jacobian adjoint=.5 H^T。运动：vendored 原 Kernel(61,.5), seed0；
  Gaussian61/sigma3/zero pad；SR DPS exact resizer4；补洞固定局部生成器 Bernoulli .7 drop，
  RGB 共享 mask，mask_seed=2001000+image_id。y 保留原 full-output Gaussian noise。
- 状态物理 readout 仅统计有效观测：补洞 mask=1；其他任务全部观测。
  R 的 noise floor 是单位参照，不规定必须降至1。
- 冻结 k/t：(10,872),(19,641),(24,319),(27,54)。四阶段覆盖，不由本轮 GT 选点。
  gamma∈{0,.5,1,1.5,2} 是响应测量网格，**不声称最优控制范围或最终上限**。
- 每次只改变一个原网格位置的引导 gamma*q，随后恢复 C 至最终输出；不叠加四脉冲。
- 在相同 next_t、相同噪声下，IEEE 分别评估五个 gamma 的 clean，加一次 gamma1 重复。
  不把 IEEE 探测量混入 native 最终轨迹，不把不同精度中心作差。
- P：有效 R、真实下一 clean 的相对测量 loss 变化/响应方向/响应尺度。
- I：d-=z_next(1)-z_next(0)，d+=z_next(2)-z_next(1) 的余弦、
  ||d+-d-||/(||d+||+||d-||)，半跨度同量、相对图像位移、响应幅度。
  这是倍率方向的有限尺度非线性，**不是 ODE 曲率、Hessian、轨迹截断误差或 LPIPS 本身**。
- 旧指标 OLD=原 R/P/I/G 因果历史；对照 STAGE、OLD、P、PI；固定 ridge=1，无权重扫描。
- 目标：最终 uint8 PSNR/SSIM、alex LPIPS、full-output loss，另报 informative loss。
  GT 仅用于生成观测和离线校准/评价；不进入 probe、特征或留出动作预测。

## 3. 成本、一致性和迁移划分

- 新 C：12×300=3600 method NFE。
- 新诊断：12×153+12×198=4212 NFE；总7812。
  每 case C replay30，四点 IEEE 探测24，native 后缀144；SR/Gaussian 复用
  父实验 k19/k24 gamma .5/1.5/2 已完成的45 calls，不重复计算。
- 失败/部分尝试全部累加；方法每个单脉冲完整科学臂300 NFE，**响应探测另付费**。
  部署同样的五点传感器需要额外调用，不能称为免费300-NFE在线方案或省步。
- C replay relative_error≤1e-6，29固定噪声、外部 RNG、zeta/D/tau、native 精度无变化；
  原算子 VJP/内积误差≤2e-5；IEEE gamma1 重复 relative_error≤1e-7。
  响应分辨门：总响应RMS>max(32*float32eps,10*repeatRMS)。未分辨保留并单报，不丢弃有害点。
- 新 motion/mask 张量与源码 SHA、输入、checkpoint、测量、PRE、outputs 哈希保存。
- 两类预注册验证：TASK_LOIO 仅同任务其他5图；LEAVE_TASK_AND_IMAGE_OUT 同时排除
  整个目标任务及当前图在其他任务的所有记录，训练为其他3任务×其他5图。
  缺失填补、按(step,gamma)中心化和标准化只由训练折计算。
- 不在整个 target-task 结果上挑模型、时间点、阈值或 gamma。
  STAGE 也用相同训练折的画质标签，是强离线时间规则，校准成本不是免费。

## 4. 预注册判定与分支

- 信息候选：每个任务 PI 相比 STAGE 和 P 的 PSNR MAE 至少降低10%，≥4/6图更好，
  LPIPS MAE 不恶化超过5%，每任务每位置至少4/6响应可分辨。未分辨样本不删除。
  四任务分别报告，全部通过才叫四任务信息候选。
- 动作候选：每固定位置由留出预测 PSNR 最大且预测 SSIM≥-.002 / LPIPS≤+.005 的
  gamma（含保持1）决策；最终实际均值对 C PSNR≥+.05dB，≥4/6图正向、画质保护通过，
  并比同折 STAGE ≥+.02dB。四个位置是四个独立单脉冲实验，**不合成为一条轨迹**。
- 正向也只解锁新图、新seed、完整在线闭环及成本匹配确认，不直接宣布开题成立。
- 若有 oracle 空间而 PI 不通过，说明测量/映射不足；不继续本指标系数搜索。
- 若最佳固定动作与逐图 oracle 接近，说明此动作库缺少逐图自适应空间；不强行包装状态收益。
- 仅 loss 改善而画质失败判负。原门控失败只处理接口/精度，不解释为科学阴性。

## 5. 执行、归档

`bash run_a41_20261025.sh`：只读 A40 检查及快照、必要服务器测试、prepare、两卡/单卡串行、
定期打印进度、终局分析与固定面板图片、独立 results/ffhq-action-geometry-20261025 分支。
新根 /data0/lzy/ZAPS-results/diag_ffhq_action_geometry/A41_20261025。
不修改历史文件、共享核心、调度或采样点；不启动 ImageNet；不恢复已停止的正则路线。
