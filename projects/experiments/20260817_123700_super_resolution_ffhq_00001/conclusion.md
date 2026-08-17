# 实验 20260817_123700_super_resolution_ffhq_00001

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00001.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=27.75** / SSIM=0.8071 / LPIPS=0.1122（观测基线 PSNR=27.76）
- NFE=120（优化100+采样20）
- 耗时: 优化=5.96s + 采样=1.16s = 合计 **7.13s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
