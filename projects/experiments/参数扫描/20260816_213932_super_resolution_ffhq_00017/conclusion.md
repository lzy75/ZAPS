# 实验 20260816_213932_super_resolution_ffhq_00017

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00017.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.98** / SSIM=0.8972 / LPIPS=0.0878（观测基线 PSNR=28.30）
- NFE=330（优化300+采样30）
- 耗时: 优化=54.11s + 采样=4.94s = 合计 **59.04s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 默认 w0.3 b0.5 m1.5

## 结论 / 观察
（待填写）
