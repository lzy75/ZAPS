# 实验 20260817_224541_super_resolution_imagenet_00005

- 代码版本: `8782aa2` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00005.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=12.09** / SSIM=0.1829 / LPIPS=0.6821（观测基线 PSNR=13.66）
- NFE=330（优化300+采样30）
- 耗时: 优化=56.83s + 采样=4.60s = 合计 **61.42s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
