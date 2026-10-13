# 实验 20260818_011432_gaussian_deblur_ffhq_00012

- 代码版本: `d5d65fc` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00012.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.97** / SSIM=0.8664 / LPIPS=0.1290（观测基线 PSNR=23.48）
- NFE=300（优化300+采样0）
- 耗时: 优化=33.66s + 采样=0.00s = 合计 **33.66s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: 方案1 baseline 四任务 eta1 last_opt NFE300

## 结论 / 观察
（待填写）
