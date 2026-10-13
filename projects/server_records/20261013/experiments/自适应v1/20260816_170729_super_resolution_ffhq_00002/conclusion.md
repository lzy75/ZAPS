# 实验 20260816_170729_super_resolution_ffhq_00002

- 代码版本: `746cc58` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00002.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=25.08** / SSIM=0.7394 / LPIPS=0.2193（观测基线 PSNR=29.42）
- NFE=330（优化300+采样30）
- 耗时: 优化=19.01s + 采样=1.83s = 合计 **20.83s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 自适应 w_cos=0.3

## 结论 / 观察
（待填写）
