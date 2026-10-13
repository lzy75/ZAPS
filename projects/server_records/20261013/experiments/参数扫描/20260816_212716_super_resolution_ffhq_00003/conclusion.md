# 实验 20260816_212716_super_resolution_ffhq_00003

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00003.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.88** / SSIM=0.8464 / LPIPS=0.0824（观测基线 PSNR=27.08）
- NFE=330（优化300+采样30）
- 耗时: 优化=50.20s + 采样=4.20s = 合计 **54.40s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 高余弦 w0.6 b1.0 m2.0

## 结论 / 观察
（待填写）
