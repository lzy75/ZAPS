# 实验 20260815_174243_super_resolution_ffhq_00014

- 代码版本: `105e2ca` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00014.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.60** / SSIM=0.8101 / LPIPS=0.1213（观测基线 PSNR=27.10）
- NFE=330（优化300+采样30）
- 耗时: 优化=24.26s + 采样=2.15s = 合计 **26.41s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①指标有效性诊断

## 结论 / 观察
（待填写）
