# 实验 20260817_231120_super_resolution_ffhq_00017

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00017.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.51** / SSIM=0.9077 / LPIPS=0.0861（观测基线 PSNR=28.30）
- NFE=330（优化300+采样30）
- 耗时: 优化=39.54s + 采样=3.44s = 合计 **42.98s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: seedA_baseline_SR

## 结论 / 观察
（待填写）
