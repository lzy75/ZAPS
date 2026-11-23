# A132：DPS 在 A118 的同一 30 点采样网格上的迁移

日期 2026-11-23，状态 planned，工作线 A。A124/A125 将细化插入密采样轨迹的部分位置。
A132 检验 DPS 在实际只执行 A118 30 个位置时，A118 式细化能否优于同网格的标量。

父记录是已完成的 A118 与同一观测的 A124R1 DPS。复用图 00108、00175、00431，各做
Gaussian/Motion，原观测、零边界算子、运动核、图像、checkpoint、初始种子及
sigma_model=.05。DPS 仍走原跳步 DDPM 和原 DPS 条件梯度；把官方 DPS 的时间
子集设为 A118 原 30 点，保留原 1000 步 beta 表及所选时间的 alpha。三组分别为
同网格原 DPS、全 30 点闭式标量、全 30 点标量后加原 A118 一步细化。
原 DPS 组条件幅度保持历史配置，无重新调参。原 1000 步父记录仅为历史背景。

每例三臂各 30 次 UNet 前向、30 次原有输入 VJP、无额外 JVP 或优化反向，共
6×3×30=540 NFE。核对官方采样器原 alpha 级别、实际 UNet 时刻序列、初始化与
最终 RNG。物理更新沿用 A124R1 的无上限 A118 单步。保存所有任务的配对 PSNR、
SSIM、LPIPS、原始 MSE 和用时。用时包括算子构造、采样与输出同步，观测和指标加载
不计入。真值仅用于评价。

各任务以对角相对标量 PSNR 平均 +.1 dB、至少 2/3 胜、SSIM 不降和 LPIPS 不升
作为进一步确认的门槛。正向支持稀疏 DPS 条件下可迁移；与密网格历史的差异作为
机制线索。固定面板方法对照优先，单独的 30 点结果不用于声明普遍收益。

在 /home/lzy/ZAPS/projects 运行 `/home/lzy/anaconda3/bin/python -u -m
utils.diag_a118_transfer_bridge dps`。默认保存路径
/data0/lzy/ZAPS-results/diag_a118_transfer_bridge/A132_dps。
