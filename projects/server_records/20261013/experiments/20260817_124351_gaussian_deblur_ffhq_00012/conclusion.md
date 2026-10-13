# 实验 20260817_124351_gaussian_deblur_ffhq_00012

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / ffhq，图像 `00012.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.46** / SSIM=0.8516 / LPIPS=0.1564（观测基线 PSNR=23.48）
- NFE=120（优化100+采样20）
- 耗时: 优化=6.42s + 采样=1.24s = 合计 **7.67s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
