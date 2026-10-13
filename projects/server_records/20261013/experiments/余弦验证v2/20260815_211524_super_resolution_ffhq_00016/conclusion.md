# 实验 20260815_211524_super_resolution_ffhq_00016

- 代码版本: `d1a44a3` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00016.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.24** / SSIM=0.8447 / LPIPS=0.1243（观测基线 PSNR=27.89）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.71s + 采样=1.73s = 合计 **20.44s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: x0余弦验证v2

## 结论 / 观察
（待填写）
