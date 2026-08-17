# 实验 20260817_120526_motion_deblur_ffhq_00015

- 代码版本: `97ea328` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00015.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.99** / SSIM=0.7376 / LPIPS=0.2175（观测基线 PSNR=22.08）
- NFE=330（优化300+采样30）
- 耗时: 优化=21.55s + 采样=2.30s = 合计 **23.86s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: 固定基线 三任务random 15-10-5 补齐

## 结论 / 观察
（待填写）
