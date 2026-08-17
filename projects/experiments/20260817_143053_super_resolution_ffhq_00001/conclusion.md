# 实验 20260817_143053_super_resolution_ffhq_00001

- 代码版本: `6b7e947` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00001.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.89** / SSIM=0.8187 / LPIPS=0.1074（观测基线 PSNR=27.78）
- NFE=330（优化300+采样30）
- 耗时: 优化=21.18s + 采样=1.92s = 合计 **23.11s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: zeta_sync_v3_SR20

## 结论 / 观察
（待填写）
