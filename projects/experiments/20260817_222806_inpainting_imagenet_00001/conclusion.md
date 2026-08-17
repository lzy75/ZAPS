# 实验 20260817_222806_inpainting_imagenet_00001

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / imagenet，图像 `00001.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=17.38** / SSIM=0.3009 / LPIPS=0.6943（观测基线 PSNR=14.88）
- NFE=330（优化300+采样30）
- 耗时: 优化=67.40s + 采样=5.98s = 合计 **73.38s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
