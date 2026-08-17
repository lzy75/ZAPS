# 实验 20260817_124505_inpainting_ffhq_00014

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / ffhq，图像 `00014.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.08** / SSIM=0.7500 / LPIPS=0.1384（观测基线 PSNR=14.14）
- NFE=120（优化100+采样20）
- 耗时: 优化=7.51s + 采样=1.59s = 合计 **9.11s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
