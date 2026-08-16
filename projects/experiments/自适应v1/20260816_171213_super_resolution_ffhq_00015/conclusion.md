# 实验 20260816_171213_super_resolution_ffhq_00015

- 代码版本: `746cc58` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00015.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.10** / SSIM=0.8144 / LPIPS=0.1594（观测基线 PSNR=28.90）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.92s + 采样=1.76s = 合计 **20.68s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 自适应 w_cos=0.3

## 结论 / 观察
（待填写）
