# 实验 20260817_223050_motion_deblur_imagenet_00001

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / imagenet，图像 `00001.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=15.50** / SSIM=0.1763 / LPIPS=0.6437（观测基线 PSNR=19.58）
- NFE=330（优化300+采样30）
- 耗时: 优化=55.57s + 采样=6.27s = 合计 **61.84s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
