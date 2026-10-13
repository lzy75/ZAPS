# 实验 20260816_233554_super_resolution_ffhq_00013

- 代码版本: `8dc8655` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00013.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=24.75** / SSIM=0.8236 / LPIPS=0.0919（观测基线 PSNR=24.80）
- NFE=330（优化300+采样30）
- 耗时: 优化=28.61s + 采样=2.90s = 合计 **31.52s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v3统一代价 omega0.5 theta0.5 SR

## 结论 / 观察
（待填写）
