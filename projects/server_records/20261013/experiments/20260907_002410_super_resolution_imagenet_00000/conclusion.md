# 实验 20260907_002410_super_resolution_imagenet_00000

- 代码版本: `856be41` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=18.38** / SSIM=0.3825 / LPIPS=0.3768（观测基线 PSNR=22.60）
- NFE=300（优化300+采样0）
- 耗时: 优化=60.69s + 采样=0.00s = 合计 **60.69s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: ablation-06: learned-range variance only

## 结论 / 观察
（待填写）
