# 实验 20260816_212420_super_resolution_ffhq_00002

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00002.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.39** / SSIM=0.8485 / LPIPS=0.1382（观测基线 PSNR=29.40）
- NFE=330（优化300+采样30）
- 耗时: 优化=43.23s + 采样=4.85s = 合计 **48.08s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 默认 w0.3 b0.5 m1.5

## 结论 / 观察
（待填写）
