# 实验 20260817_221508_super_resolution_ffhq_00000

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.19** / SSIM=0.8660 / LPIPS=0.0814（观测基线 PSNR=28.72）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.65s + 采样=1.48s = 合计 **20.13s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: edm_base_smoke

## 结论 / 观察
（待填写）
