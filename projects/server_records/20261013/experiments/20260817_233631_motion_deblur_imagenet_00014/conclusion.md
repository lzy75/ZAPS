# 实验 20260817_233631_motion_deblur_imagenet_00014

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / imagenet，图像 `00014.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=15.51** / SSIM=0.1893 / LPIPS=0.6488（观测基线 PSNR=21.26）
- NFE=330（优化300+采样30）
- 耗时: 优化=45.25s + 采样=3.81s = 合计 **49.06s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
