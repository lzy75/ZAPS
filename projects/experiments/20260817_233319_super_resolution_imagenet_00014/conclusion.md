# 实验 20260817_233319_super_resolution_imagenet_00014

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00014.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=16.20** / SSIM=0.2436 / LPIPS=0.7430（观测基线 PSNR=28.04）
- NFE=330（优化300+采样30）
- 耗时: 优化=44.54s + 采样=4.30s = 合计 **48.84s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
