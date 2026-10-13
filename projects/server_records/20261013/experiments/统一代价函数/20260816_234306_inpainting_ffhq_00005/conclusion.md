# 实验 20260816_234306_inpainting_ffhq_00005

- 代码版本: `8dc8655` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / ffhq，图像 `00005.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=26.57** / SSIM=0.7885 / LPIPS=0.1695（观测基线 PSNR=12.29）
- NFE=330（优化300+采样30）
- 耗时: 优化=23.82s + 采样=2.48s = 合计 **26.30s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（ffhq）
- 目的: v3统一代价 omega0.5 theta0.5 三任务random

## 结论 / 观察
（待填写）
