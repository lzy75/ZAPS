# 实验 20260817_143751_super_resolution_ffhq_00015

- 代码版本: `6b7e947` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00015.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.94** / SSIM=0.8790 / LPIPS=0.1143（观测基线 PSNR=28.91）
- NFE=330（优化300+采样30）
- 耗时: 优化=25.68s + 采样=1.91s = 合计 **27.59s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: zeta_sync_v3_SR20

## 结论 / 观察
（待填写）
