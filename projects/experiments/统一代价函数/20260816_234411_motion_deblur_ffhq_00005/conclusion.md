# 实验 20260816_234411_motion_deblur_ffhq_00005

- 代码版本: `8dc8655` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00005.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.51** / SSIM=0.6916 / LPIPS=0.2510（观测基线 PSNR=20.31）
- NFE=330（优化300+采样30）
- 耗时: 优化=25.98s + 采样=2.57s = 合计 **28.55s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: v3统一代价 omega0.5 theta0.5 三任务random

## 结论 / 观察
（待填写）
