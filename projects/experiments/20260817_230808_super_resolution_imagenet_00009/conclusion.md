# 实验 20260817_230808_super_resolution_imagenet_00009

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00009.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=20.58** / SSIM=0.4639 / LPIPS=0.5487（观测基线 PSNR=24.70）
- NFE=330（优化300+采样30）
- 耗时: 优化=86.78s + 采样=7.59s = 合计 **94.37s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
