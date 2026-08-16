# 实验 20260816_213738_super_resolution_ffhq_00014

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00014.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.06** / SSIM=0.7951 / LPIPS=0.1249（观测基线 PSNR=27.12）
- NFE=330（优化300+采样30）
- 耗时: 优化=45.04s + 采样=6.03s = 合计 **51.07s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 激进 w0.3 b1.0 m2.0

## 结论 / 观察
（待填写）
