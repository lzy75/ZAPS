# 实验 20260816_213235_super_resolution_ffhq_00008

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00008.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.10** / SSIM=0.8119 / LPIPS=0.1165（观测基线 PSNR=27.40）
- NFE=330（优化300+采样30）
- 耗时: 优化=52.90s + 采样=5.83s = 合计 **58.73s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 高余弦 w0.6 b1.0 m2.0

## 结论 / 观察
（待填写）
