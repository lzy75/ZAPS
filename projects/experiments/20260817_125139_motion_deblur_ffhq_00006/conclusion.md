# 实验 20260817_125139_motion_deblur_ffhq_00006

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **motion_deblur** / ffhq，图像 `00006.png`
- 关键参数: steps=20 schedule=(15, 10, 5) epochs=5 lr=0.001 zeta_init=0.2 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=23.40** / SSIM=0.7053 / LPIPS=0.2356（观测基线 PSNR=20.79）
- NFE=170（优化150+采样20）
- 耗时: 优化=8.80s + 采样=1.00s = 合计 **9.80s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **motion_deblur**（ffhq）
- 目的: NFE100 v3统一代价 omega0.5 theta0.5 四任务

## 结论 / 观察
（待填写）
