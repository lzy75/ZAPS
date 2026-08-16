# 实验 20260816_212858_super_resolution_ffhq_00005

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00005.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.05** / SSIM=0.8081 / LPIPS=0.1748（观测基线 PSNR=26.98）
- NFE=330（优化300+采样30）
- 耗时: 优化=50.23s + 采样=5.92s = 合计 **56.15s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 纯残差 w0 b1.0 m2.0

## 结论 / 观察
（待填写）
