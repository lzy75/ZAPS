# 实验 20260815_203828_super_resolution_ffhq_00004

- 代码版本: `96dec19` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00004.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.87** / SSIM=0.7745 / LPIPS=0.1639（观测基线 PSNR=23.26）
- NFE=330（优化300+采样30）
- 耗时: 优化=21.87s + 采样=2.34s = 合计 **24.21s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①:x̂₀稳定版余弦 vs 含噪版对比

## 结论 / 观察
（待填写）
