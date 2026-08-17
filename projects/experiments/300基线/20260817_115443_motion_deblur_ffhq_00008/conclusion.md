# 实验 20260817_115443_motion_deblur_ffhq_00008

- 代码版本: `97ea328` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00008.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=24.19** / SSIM=0.7072 / LPIPS=0.1653（观测基线 PSNR=20.99）
- NFE=330（优化300+采样30）
- 耗时: 优化=23.06s + 采样=2.32s = 合计 **25.39s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: 固定基线 三任务random 15-10-5 补齐

## 结论 / 观察
（待填写）
