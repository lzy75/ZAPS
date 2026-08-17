# 实验 20260817_115025_motion_deblur_ffhq_00005

- 代码版本: `97ea328` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00005.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.37** / SSIM=0.6753 / LPIPS=0.2443（观测基线 PSNR=20.32）
- NFE=330（优化300+采样30）
- 耗时: 优化=21.36s + 采样=2.12s = 合计 **23.48s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: 固定基线 三任务random 15-10-5 补齐

## 结论 / 观察
（待填写）
