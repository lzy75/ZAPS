# 实验 20260817_221801_super_resolution_ffhq_00004

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00004.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.69** / SSIM=0.7738 / LPIPS=0.1837（观测基线 PSNR=23.26）
- NFE=330（优化300+采样30）
- 耗时: 优化=24.03s + 采样=2.84s = 合计 **26.87s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: edm_base_v3_SR20

## 结论 / 观察
（待填写）
