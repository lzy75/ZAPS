# 实验 20260816_170812_super_resolution_ffhq_00004

- 代码版本: `746cc58` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00004.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=21.01** / SSIM=0.6869 / LPIPS=0.3125（观测基线 PSNR=23.25）
- NFE=330（优化300+采样30）
- 耗时: 优化=17.49s + 采样=1.72s = 合计 **19.21s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 自适应 w_cos=0.3

## 结论 / 观察
（待填写）
