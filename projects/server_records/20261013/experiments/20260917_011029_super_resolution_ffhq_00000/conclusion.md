# 实验 20260917_011029_super_resolution_ffhq_00000

- 代码版本: `3f7692c` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=22.67** / SSIM=0.7143 / LPIPS=0.2009（观测基线 PSNR=28.75）
- NFE=300（优化300+采样0）
- 耗时: 优化=19.42s + 采样=0.00s = 合计 **19.42s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: FFHQ current-code paper baseline: 15-10-5, Adam default lr=0.001

## 结论 / 观察
（待填写）
