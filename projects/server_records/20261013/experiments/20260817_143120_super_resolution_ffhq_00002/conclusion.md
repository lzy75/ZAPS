# 实验 20260817_143120_super_resolution_ffhq_00002

- 代码版本: `6b7e947` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00002.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.66** / SSIM=0.8520 / LPIPS=0.1236（观测基线 PSNR=29.37）
- NFE=330（优化300+采样30）
- 耗时: 优化=23.59s + 采样=1.83s = 合计 **25.42s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: zeta_sync_v3_SR20

## 结论 / 观察
（待填写）
