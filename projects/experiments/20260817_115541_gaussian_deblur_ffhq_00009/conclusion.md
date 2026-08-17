# 实验 20260817_115541_gaussian_deblur_ffhq_00009

- 代码版本: `97ea328` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00009.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.96** / SSIM=0.8399 / LPIPS=0.1646（观测基线 PSNR=24.72）
- NFE=330（优化300+采样30）
- 耗时: 优化=22.63s + 采样=1.59s = 合计 **24.22s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: 固定基线 三任务random 15-10-5 补齐

## 结论 / 观察
（待填写）
