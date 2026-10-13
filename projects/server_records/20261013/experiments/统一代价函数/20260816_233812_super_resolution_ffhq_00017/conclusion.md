# 实验 20260816_233812_super_resolution_ffhq_00017

- 代码版本: `8dc8655` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00017.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.16** / SSIM=0.8991 / LPIPS=0.0814（观测基线 PSNR=28.28）
- NFE=330（优化300+采样30）
- 耗时: 优化=28.86s + 采样=3.01s = 合计 **31.87s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v3统一代价 omega0.5 theta0.5 SR

## 结论 / 观察
（待填写）
