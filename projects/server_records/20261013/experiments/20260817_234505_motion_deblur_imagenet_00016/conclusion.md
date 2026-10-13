# 实验 20260817_234505_motion_deblur_imagenet_00016

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / imagenet，图像 `00016.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=16.49** / SSIM=0.2188 / LPIPS=0.6976（观测基线 PSNR=22.09）
- NFE=330（优化300+采样30）
- 耗时: 优化=47.41s + 采样=4.12s = 合计 **51.53s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
