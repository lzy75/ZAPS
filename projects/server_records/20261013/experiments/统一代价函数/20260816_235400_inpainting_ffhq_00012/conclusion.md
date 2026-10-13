# 实验 20260816_235400_inpainting_ffhq_00012

- 代码版本: `8dc8655` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / ffhq，图像 `00012.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.44** / SSIM=0.8600 / LPIPS=0.1064（观测基线 PSNR=12.70）
- NFE=330（优化300+采样30）
- 耗时: 优化=22.04s + 采样=2.35s = 合计 **24.39s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（ffhq）
- 目的: v3统一代价 omega0.5 theta0.5 三任务random

## 结论 / 观察
（待填写）
