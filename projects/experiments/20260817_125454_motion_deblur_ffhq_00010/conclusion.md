# 实验 20260817_125454_motion_deblur_ffhq_00010

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00010.png`
- 关键参数: steps=20 schedule=(15, 10, 5) epochs=5 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=20.77** / SSIM=0.6037 / LPIPS=0.2643（观测基线 PSNR=19.30）
- NFE=170（优化150+采样20）
- 耗时: 优化=9.55s + 采样=1.26s = 合计 **10.81s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: NFE100 v3统一代价 omega0.5 theta0.5 四任务

## 结论 / 观察
（待填写）
