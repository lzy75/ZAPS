# 实验 20260817_000339_gaussian_deblur_ffhq_00018

- 代码版本: `8dc8655` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00018.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.68** / SSIM=0.7674 / LPIPS=0.1359（观测基线 PSNR=24.40）
- NFE=330（优化300+采样30）
- 耗时: 优化=25.19s + 采样=2.46s = 合计 **27.64s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: v3统一代价 omega0.5 theta0.5 三任务random

## 结论 / 观察
（待填写）
