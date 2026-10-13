# 实验 20260817_231613_inpainting_imagenet_00010

- 代码版本: `8111950` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / imagenet，图像 `00010.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=17.30** / SSIM=0.3659 / LPIPS=0.6321（观测基线 PSNR=12.35）
- NFE=330（优化300+采样30）
- 耗时: 优化=48.31s + 采样=6.02s = 合计 **54.34s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（imagenet）
- 目的: ImageNet 固定基线 NFE300 四任务

## 结论 / 观察
（待填写）
