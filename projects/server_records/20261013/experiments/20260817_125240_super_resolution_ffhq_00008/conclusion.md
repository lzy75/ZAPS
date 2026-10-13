# 实验 20260817_125240_super_resolution_ffhq_00008

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00008.png`
- 关键参数: steps=20 schedule=(15, 10, 5) epochs=5 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.44** / SSIM=0.7822 / LPIPS=0.1494（观测基线 PSNR=27.38）
- NFE=170（优化150+采样20）
- 耗时: 优化=9.26s + 采样=1.13s = 合计 **10.39s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: NFE100 v3统一代价 omega0.5 theta0.5 四任务

## 结论 / 观察
（待填写）
