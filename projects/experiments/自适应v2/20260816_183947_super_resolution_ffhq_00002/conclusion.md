# 实验 20260816_183947_super_resolution_ffhq_00002

- 代码版本: `b854db9` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00002.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=29.59** / SSIM=0.8455 / LPIPS=0.1605（观测基线 PSNR=29.38）
- NFE=330（优化300+采样30）
- 耗时: 优化=18.47s + 采样=1.78s = 合计 **20.25s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: 自适应v2 w_cos=0.3 vs 基线

## 结论 / 观察
（待填写）
