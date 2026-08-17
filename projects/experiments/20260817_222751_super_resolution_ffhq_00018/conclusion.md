# 实验 20260817_222751_super_resolution_ffhq_00018

- 代码版本: `e80826b` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00018.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.74** / SSIM=0.8261 / LPIPS=0.1079（观测基线 PSNR=27.60）
- NFE=330（优化300+采样30）
- 耗时: 优化=38.06s + 采样=3.44s = 合计 **41.50s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: edm_base_v3_SR20

## 结论 / 观察
（待填写）
