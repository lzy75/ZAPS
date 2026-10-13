# 实验 20260905_102202_super_resolution_imagenet_00000

- 代码版本: `3e463bf` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=15.95** / SSIM=0.5356 / LPIPS=0.4127（观测基线 PSNR=22.60）
- NFE=330（优化300+采样30）
- 耗时: 优化=41.74s + 采样=4.04s = 合计 **45.77s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: 仅替换官方ImageNet权重，保留原ZAPS配置

## 结论 / 观察
（待填写）
