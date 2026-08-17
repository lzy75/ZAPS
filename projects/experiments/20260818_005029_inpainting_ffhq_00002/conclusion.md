# 实验 20260818_005029_inpainting_ffhq_00002

- 代码版本: `d5d65fc` (branch=master, dirty=True)
- 任务 / 数据集: **inpainting** / ffhq，图像 `00002.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.25** / SSIM=0.8072 / LPIPS=0.1321（观测基线 PSNR=14.95）
- NFE=300（优化300+采样0）
- 耗时: 优化=28.58s + 采样=0.00s = 合计 **28.58s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **inpainting**（ffhq）
- 目的: 方案1 baseline 四任务 eta1 last_opt NFE300

## 结论 / 观察
（待填写）
