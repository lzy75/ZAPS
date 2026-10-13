# 实验 20260818_013223_motion_deblur_ffhq_00018

- 代码版本: `d5d65fc` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00018.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.30** / SSIM=0.6633 / LPIPS=0.1734（观测基线 PSNR=20.55）
- NFE=300（优化300+采样0）
- 耗时: 优化=38.94s + 采样=0.00s = 合计 **38.94s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: 方案1 v3 四任务 eta1 last_opt NFE300

## 结论 / 观察
（待填写）
