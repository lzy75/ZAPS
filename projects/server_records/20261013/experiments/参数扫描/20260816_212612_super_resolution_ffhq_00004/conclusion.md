# 实验 20260816_212612_super_resolution_ffhq_00004

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00004.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.44** / SSIM=0.7714 / LPIPS=0.1915（观测基线 PSNR=23.25）
- NFE=330（优化300+采样30）
- 耗时: 优化=47.70s + 采样=5.28s = 合计 **52.98s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 默认 w0.3 b0.5 m1.5

## 结论 / 观察
（待填写）
