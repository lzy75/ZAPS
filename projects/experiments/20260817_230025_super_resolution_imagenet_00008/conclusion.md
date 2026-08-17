# 实验 20260817_230025_super_resolution_imagenet_00008

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00008.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=15.31** / SSIM=0.2433 / LPIPS=0.7097（观测基线 PSNR=24.52）
- NFE=330（优化300+采样30）
- 耗时: 优化=83.85s + 采样=9.71s = 合计 **93.56s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
