# 实验 20260815_193358_super_resolution_ffhq_00016

- 代码版本: `bb1e7c4` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00016.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.41** / SSIM=0.7933 / LPIPS=0.1776（观测基线 PSNR=27.90）
- NFE=330（优化300+采样30）
- 耗时: 优化=22.47s + 采样=1.89s = 合计 **24.37s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①诊断:eta=0确定性采样对照,查cos全负成因

## 结论 / 观察
（待填写）
