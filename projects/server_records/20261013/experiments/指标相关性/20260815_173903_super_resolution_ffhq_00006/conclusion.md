# 实验 20260815_173903_super_resolution_ffhq_00006

- 代码版本: `105e2ca` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00006.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.26** / SSIM=0.8704 / LPIPS=0.1105（观测基线 PSNR=27.34）
- NFE=330（优化300+采样30）
- 耗时: 优化=22.77s + 采样=1.43s = 合计 **24.20s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①指标有效性诊断

## 结论 / 观察
（待填写）
