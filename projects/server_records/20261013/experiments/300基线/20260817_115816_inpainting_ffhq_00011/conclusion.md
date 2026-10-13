# 实验 20260817_115816_inpainting_ffhq_00011

- 代码版本: `97ea328` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / ffhq，图像 `00011.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.77** / SSIM=0.8383 / LPIPS=0.1296（观测基线 PSNR=12.55）
- NFE=330（优化300+采样30）
- 耗时: 优化=26.92s + 采样=2.68s = 合计 **29.60s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（ffhq）
- 目的: 固定基线 三任务random 15-10-5 补齐

## 结论 / 观察
（待填写）
