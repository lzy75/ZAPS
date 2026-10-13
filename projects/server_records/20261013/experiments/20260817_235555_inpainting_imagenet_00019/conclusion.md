# 实验 20260817_235555_inpainting_imagenet_00019

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / imagenet，图像 `00019.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=15.98** / SSIM=0.3691 / LPIPS=0.6299（观测基线 PSNR=13.36）
- NFE=330（优化300+采样30）
- 耗时: 优化=41.87s + 采样=4.41s = 合计 **46.28s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
