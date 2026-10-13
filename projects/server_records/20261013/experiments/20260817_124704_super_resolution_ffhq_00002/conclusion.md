# 实验 20260817_124704_super_resolution_ffhq_00002

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00002.png`
- 关键参数: steps=20 schedule=(15, 10, 5) epochs=5 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.40** / SSIM=0.8375 / LPIPS=0.1626（观测基线 PSNR=29.42）
- NFE=170（优化150+采样20）
- 耗时: 优化=12.70s + 采样=1.89s = 合计 **14.60s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: NFE100 v3统一代价 omega0.5 theta0.5 四任务

## 结论 / 观察
（待填写）
