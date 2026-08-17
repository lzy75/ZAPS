# 实验 20260817_222029_super_resolution_imagenet_00000

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00000.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=16.63** / SSIM=0.3322 / LPIPS=0.5167（观测基线 PSNR=22.60）
- NFE=330（优化300+采样30）
- 耗时: 优化=67.68s + 采样=5.42s = 合计 **73.10s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
