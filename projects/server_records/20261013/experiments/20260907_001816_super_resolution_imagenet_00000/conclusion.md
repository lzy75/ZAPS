# 实验 20260907_001816_super_resolution_imagenet_00000

- 代码版本: `6470294` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=19.06** / SSIM=0.4033 / LPIPS=0.3688（观测基线 PSNR=22.60）
- NFE=300（优化300+采样0）
- 耗时: 优化=61.36s + 采样=0.00s = 合计 **61.36s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ablation-05: paper DDPM update only

## 结论 / 观察
（待填写）
