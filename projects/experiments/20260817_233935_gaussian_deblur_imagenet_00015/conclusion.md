# 实验 20260817_233935_gaussian_deblur_imagenet_00015

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / imagenet，图像 `00015.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=15.83** / SSIM=0.2828 / LPIPS=0.5753（观测基线 PSNR=20.88）
- NFE=330（优化300+采样30）
- 耗时: 优化=45.85s + 采样=3.95s = 合计 **49.80s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
