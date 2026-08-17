# 实验 20260817_143027_super_resolution_ffhq_00000

- 代码版本: `6b7e947` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.82** / SSIM=0.8637 / LPIPS=0.0972（观测基线 PSNR=28.73）
- NFE=330（优化300+采样30）
- 耗时: 优化=21.12s + 采样=1.76s = 合计 **22.88s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: zeta_sync_v3_SR20

## 结论 / 观察
（待填写）
