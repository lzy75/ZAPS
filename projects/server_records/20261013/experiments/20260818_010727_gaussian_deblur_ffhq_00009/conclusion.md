# 实验 20260818_010727_gaussian_deblur_ffhq_00009

- 代码版本: `d5d65fc` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00009.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.92** / SSIM=0.8399 / LPIPS=0.1323（观测基线 PSNR=24.73）
- NFE=300（优化300+采样0）
- 耗时: 优化=36.82s + 采样=0.00s = 合计 **36.82s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: 方案1 v3 四任务 eta1 last_opt NFE300

## 结论 / 观察
（待填写）
