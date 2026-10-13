# 实验 20260815_211608_super_resolution_ffhq_00018

- 代码版本: `d1a44a3` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00018.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.02** / SSIM=0.8400 / LPIPS=0.1250（观测基线 PSNR=27.56）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.72s + 采样=1.67s = 合计 **20.38s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: x0余弦验证v2

## 结论 / 观察
（待填写）
