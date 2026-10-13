# 实验 20260816_212816_super_resolution_ffhq_00005

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00005.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.10** / SSIM=0.8266 / LPIPS=0.1590（观测基线 PSNR=26.98）
- NFE=330（优化300+采样30）
- 耗时: 优化=53.45s + 采样=4.21s = 合计 **57.66s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 激进 w0.3 b1.0 m2.0

## 结论 / 观察
（待填写）
