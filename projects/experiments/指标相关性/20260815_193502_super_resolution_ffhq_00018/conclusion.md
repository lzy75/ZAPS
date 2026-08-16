# 实验 20260815_193502_super_resolution_ffhq_00018

- 代码版本: `bb1e7c4` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00018.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.03** / SSIM=0.7956 / LPIPS=0.2252（观测基线 PSNR=27.56）
- NFE=330（优化300+采样30）
- 耗时: 优化=23.11s + 采样=2.12s = 合计 **25.23s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①诊断:eta=0确定性采样对照,查cos全负成因

## 结论 / 观察
（待填写）
