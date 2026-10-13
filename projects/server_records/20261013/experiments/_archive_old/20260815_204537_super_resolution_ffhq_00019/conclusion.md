# 实验 20260815_204537_super_resolution_ffhq_00019

- 代码版本: `96dec19` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00019.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.12** / SSIM=0.8802 / LPIPS=0.0949（观测基线 PSNR=27.09）
- NFE=330（优化300+采样30）
- 耗时: 优化=21.32s + 采样=2.03s = 合计 **23.35s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 创新点①:x̂₀稳定版余弦 vs 含噪版对比

## 结论 / 观察
（待填写）
