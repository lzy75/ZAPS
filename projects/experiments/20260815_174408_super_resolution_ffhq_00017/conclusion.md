# 实验 20260815_174408_super_resolution_ffhq_00017

- 代码版本: `105e2ca` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00017.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.86** / SSIM=0.9127 / LPIPS=0.0793（观测基线 PSNR=28.29）
- NFE=330（优化300+采样30）
- 耗时: 优化=22.51s + 采样=2.26s = 合计 **24.77s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①指标有效性诊断

## 结论 / 观察
（待填写）
