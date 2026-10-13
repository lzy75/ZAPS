# 实验 20260817_124838_gaussian_deblur_ffhq_00018

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00018.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=24.97** / SSIM=0.7453 / LPIPS=0.1625（观测基线 PSNR=24.38）
- NFE=120（优化100+采样20）
- 耗时: 优化=9.23s + 采样=1.83s = 合计 **11.07s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
