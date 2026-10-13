# 实验 20260817_124427_gaussian_deblur_ffhq_00013

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00013.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=22.45** / SSIM=0.7424 / LPIPS=0.1363（观测基线 PSNR=22.35）
- NFE=120（优化100+采样20）
- 耗时: 优化=6.46s + 采样=1.14s = 合计 **7.60s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
