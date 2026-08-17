# 实验 20260817_221950_super_resolution_ffhq_00007

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00007.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.63** / SSIM=0.7638 / LPIPS=0.1278（观测基线 PSNR=26.46）
- NFE=330（优化300+采样30）
- 耗时: 优化=36.19s + 采样=3.10s = 合计 **39.29s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: edm_base_v3_SR20

## 结论 / 观察
（待填写）
