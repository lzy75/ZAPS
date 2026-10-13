# 实验 20260817_234358_gaussian_deblur_imagenet_00016

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **gaussian_deblur** / imagenet，图像 `00016.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=16.60** / SSIM=0.2566 / LPIPS=0.7629（观测基线 PSNR=25.69）
- NFE=330（优化300+采样30）
- 耗时: 优化=46.80s + 采样=4.11s = 合计 **50.91s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **gaussian_deblur**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
