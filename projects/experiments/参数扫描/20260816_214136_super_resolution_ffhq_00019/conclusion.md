# 实验 20260816_214136_super_resolution_ffhq_00019

- 代码版本: `bb0bc6b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00019.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=26.96** / SSIM=0.8577 / LPIPS=0.0844（观测基线 PSNR=27.06）
- NFE=330（优化300+采样30）
- 耗时: 优化=49.76s + 采样=5.60s = 合计 **55.36s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: v2fix 默认 w0.3 b0.5 m1.5

## 结论 / 观察
（待填写）
