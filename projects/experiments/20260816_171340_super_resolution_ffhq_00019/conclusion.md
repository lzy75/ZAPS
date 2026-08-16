# 实验 20260816_171340_super_resolution_ffhq_00019

- 代码版本: `746cc58` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00019.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=24.05** / SSIM=0.7845 / LPIPS=0.1425（观测基线 PSNR=27.05）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.59s + 采样=1.89s = 合计 **20.49s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 自适应 w_cos=0.3

## 结论 / 观察
（待填写）
