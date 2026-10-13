# 实验 20260816_170834_super_resolution_ffhq_00005

- 代码版本: `746cc58` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00005.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=22.51** / SSIM=0.6688 / LPIPS=0.2580（观测基线 PSNR=26.97）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.72s + 采样=1.82s = 合计 **20.55s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 自适应 w_cos=0.3

## 结论 / 观察
（待填写）
