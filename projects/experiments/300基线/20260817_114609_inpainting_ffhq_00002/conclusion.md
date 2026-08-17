# 实验 20260817_114609_inpainting_ffhq_00002

- 代码版本: `97ea328` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / ffhq，图像 `00002.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.36** / SSIM=0.8242 / LPIPS=0.1629（观测基线 PSNR=14.97）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.46s + 采样=1.60s = 合计 **20.06s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（ffhq）
- 目的: 固定基线 三任务random 15-10-5 补齐

## 结论 / 观察
（待填写）
