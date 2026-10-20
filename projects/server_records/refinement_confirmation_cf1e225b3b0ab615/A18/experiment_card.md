# A18：冻结F2的新图确认

状态 planned，2026-10-20，A线 local_reference；只确认固定动作，不宣称状态收益。

## 目标与依据

A16/A17在两组历史图共12例上，F2相对C平均PSNR+0.273388、SSIM+0.008154、
LPIPS+0.004060；两组分别通过平均保护线。部分图LPIPS明显退化。
A16方向cap未产生增量且已停止，不继续调系数/方向阈值。
可否定假设：冻结的 clean+2q 在新图上仍优于原C与同成本clean+q，并满足平均质量保护线。

父源 A17 `/data0/lzy/ZAPS-results/diag_fixed_terminal_replication/A17_20261020`，
全部六例completed/SHA/门控及重算摘要通过。A13提供已验证的native/IEEE策略。
本轮首次使用固定名单00026–00031，各seed1000/2000，物理测量seed=1000+image_id，
每图两seed共享测量。启动前核对可访问results/server_records历史run与plan的ID/图像SHA；
若发现重用/图像缺失则在UNet调用前停止，禁止看结果后更换名单。
“新图”只限可访问记录审计范围，不能认证未被基础模型训练使用或所有外部历史。

## 固定协议与唯一变量

从A17继承config/grid/checkpoint；原FFHQ SR4 unit_operator+legacy_bicubic、sigma=.05、
原30步×10轮学习ζ/D、original last_opt（末次Adam更新前输出）、native精度，scheduler=None。
基线C由该原协议重新生成，不加载其他图的学习参数。快照用既有无扰动SnapshotZAPS，
保留最后unroll的PRE；不改变核心、不额外回放30步或存整个轨迹。
额外调用在原C、t=0/index0、该图PRE下以IEEE精度计算clean和q。
同次输出给出F1=clean+q、F2=clean+2q；重复同输入一次做计量门控。
所有额外臂固定后保存，再进行uint8 PSNR/SSIM11与Alex LPIPS离线评价。
真值仅用于生成测量和离线评价，不进入optimize(x0_gt=None)、输出选择或停止。

## 预算/门控

12例，每例300原方法NFE+2诊断NFE，共3600 method+24 diagnostic。
科学部署成本C=300、F1/F2=301；重复额外调用是诊断成本。
默认按图像分两片、每片6例，避免共享测量并发写；一卡顺序。12例均运行，不按早期指标停。
逐次调用前持久化计数；training时间序列必须grid[::-1]×10，十轮loss有限、Snapshot审计10次、
last_opt初始noise及末端RNG吻合原一次init+290次采样噪声，UNet无梯度。
native/IEEE策略均与A13一致，退出恢复；原C/PRE有限，PRE网格正确。
额外中心clean/q与F1/F2重复、F1实际terminal core误差≤1e−7；禁止额外噪声抽样。
ζ/D/PRE网格保持；仿射损失模型误差≤1e−7+2e−5|loss|；外部RNG恢复。
原图、测量、模型、核心源码及独立入口SHA全部绑定。失败/中断保留，不自动覆盖或复训。

## 事先判据与分支

主判据在本轮12例/6图上独立计算，历史图均值不得救回失败：
F2相对C平均PSNR正向且超过数值保护线，SSIM降幅≤.002、LPIPS升幅≤.005，
平均物理损失非增（数值保护线内），至少4/6图像簇seed均值PSNR正向。
F2相对同成本F1也需PSNR超保护线且至少4/6图正向，SSIM/LPIPS退化在相同容差内。
数值保护线10×max(repeat误差，PSNR1e−6、SSIM/LPIPS1e−7、loss1e−10)。
报告每图每seed、均值、中位数、最差值、LPIPS退化>.005案例、方法/诊断NFE和耗时。
正向只支持这个名单上的固定动作确认，仍无状态控制/其他任务泛化结论；后续才讨论较大确认集。
负向停止固定F2路线，不扫3/4倍或替换失败图；门控失败只修具体协议问题。

## 执行与归档

`bash run_a18_20261020.sh`：测试、身份/历史检查、双卡、等待、集中摘要与push。
新目录 `/data0/lzy/ZAPS-results/diag_frozen_f2_confirmation/A18_20261020`。
每例存PRE/last_opt/额外输出，不保存完整30步轨迹；张量留服务器，轻量结果提交
results/refinement-confirmation-20261020。Mac仅快速CPU/语法检查，Torch/CUDA在服务器执行。
