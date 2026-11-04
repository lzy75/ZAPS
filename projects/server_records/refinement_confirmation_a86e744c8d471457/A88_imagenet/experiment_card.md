# A88：补齐 DPS 的 Gaussian 与随机补洞迁移

## 目标与可否定假设

填补 FFHQ / ImageNet × Gaussian / inpainting 四个条件。已有 A75/A77 的 SR4 / Motion 不重跑。
仅比较原 DPS 与原 A71 噪声感知连续倍率，判断质量、耗时及区域误差；不加入固定/错配倍率。
若逐数据集/任务均值 PSNR 至少 +0.1dB、至少 2/3 图 PSNR 改善、SSIM 不降且 LPIPS 不升，配对门控通过，则记小面板正向。
混合和负向完整报告，不调参救分。全部已见图、一 seed，不能认定普遍迁移或在线状态必要性。

## 父记录及固定条件

读取已完成 A75 plan / summary；物理算子沿用其 A71 根目录依赖。独立新目录、NEW shared y，两个方法同 y / mask / init / innovations。
FFHQ 00226/00443/00963；ImageNet 00067/00151/00193；原已有检查点，SHA 固定，无下载。
Gaussian 核 61、sigma3、zero boundary；随机补洞 70% 缺失，固定 mask_seed。
模型坐标 [-1,1]、观测 sigma=.05；保留官方 full-output Gaussian noise，包括缺失位置的纯噪声。
这与沿用的 ZAPS 物理算子一致；Gaussian 不是另换官方运动核或作者 benchmark 平均。

官方本地 DPS configs/gaussian_deblur_config.yaml：PS scale=.3；inpainting_config.yaml：PS scale=.5。
不把所有任务强设 .3。原1000 DDPM learned_range、clip_denoised=True、原返回 next-state，原噪声路径保留。
使用现有 FFHQ / ImageNet 对应权重与架构；不同权重和数据同时改变，不能隔离数据集效应。

## 唯一变量及插入位置

q=-scale * grad_x ||y-H(m)||；原 DPS 为 x_next=x_native+q。
新方法仅替换为 x_next=x_native+gamma*q，全部1000步在线计算，非 ZAPS 最后一轮逻辑。
r=y-H(m)，d=H(m+q/sqrt(alpha_bar_t))-H(m)。
gamma=max(0,1-M*sigma²/||r||²)*max(0,<r,d>)/||d||²；原 noise_signal_shrink.select 边界处理不变。
补洞倍率统计仅选观测 RGB，M=3*观测像素数；原 DPS 梯度的全输出范数不改。
此残差扣噪是既有启发式，不是无偏风险估计；同时间响应代理，不是真实下一网络响应。
正全局原引导幅度可被投影抵消；不声称保留其幅度意义，不改梯度方向。

## 预算与时间

每数据集 3图 ×2任务 ×2臂=12 次独立1000步；12000F +12000原输入VJP；0 optimizer。
两数据集合计24000F/24000VJP。每模型加载另一次真实1F/1VJP资源探针，分开记账。
模块每个候选步额外3次 H（残差1、有限方向线性响应2），每数据集18000 module H；无额外 NN / VJP。
完整计时含给定 y 载入、算子构造、init、1000步、CPU输出同步、进度写入；模型载入/探针、共享观测创建、离线指标和图片写入分开。
神经网络及采样完全重跑，不复用 learned cache，不计旧 A75/A77 成本为新成本。

## 门控、失败与输出

本机无Torch/CUDA：只运行协议/CLI测试。服务器入口强制真实 Torch 测试，不skip；原 checkpoint input-VJP兼容测试同时执行。
真实两任务算子、强制 gamma1 与原 DPS、原循环、RNG/网格/输入梯度与NN参数不变门控。
FFHQ至少6GiB、ImageNet至少10GiB，加载后按实测输入VJP峰值+1GiB准入；只在未开始前资源暂停，原命令续未开始部分。
不终止其他进程，不降低精度/分辨率，不自动重试失败模型案例；旧源、旧记录不改。
每50步输出进度；分数据集独立断线安全队列，末尾 summary、comparison.csv/md、全部样本原图/输出PNG以及三联图。
两臂初始/最终CPU+CUDA RNG和NN时间索引必须一致，否则该配对不接受。
保存补洞 observed/missing GT MSE 仅离线评价，GT不进入倍率决策。
轻量 results/dps-missing-tasks-20261104-{dataset} 自动归档推送；不上传大PT到master。

## 运行

bash projects/run_a88_20261104.sh --dataset ffhq --gpus 0
bash projects/run_a88_20261104.sh --dataset imagenet --gpus 1

设备按自己的分配修改；单卡先后运行亦可。--status 只读；--archive-only 只重试汇总/上传，不重跑GPU。
