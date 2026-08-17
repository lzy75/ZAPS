# 实验 20260817_124221_super_resolution_ffhq_00010

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00010.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=24.48** / SSIM=0.8237 / LPIPS=0.1161（观测基线 PSNR=24.16）
- NFE=120（优化100+采样20）
- 耗时: 优化=6.05s + 采样=1.07s = 合计 **7.12s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
