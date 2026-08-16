# 实验 20260815_174028_super_resolution_ffhq_00009

- 代码版本: `105e2ca` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00009.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.02** / SSIM=0.8793 / LPIPS=0.1248（观测基线 PSNR=27.55）
- NFE=330（优化300+采样30）
- 耗时: 优化=23.92s + 采样=1.78s = 合计 **25.70s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①指标有效性诊断

## 结论 / 观察
（待填写）
