# 实验 20260905_103253_super_resolution_imagenet_00000

- 代码版本: `3e463bf` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / imagenet，图像 `00000.png`
- 关键参数: steps=30 schedule=(15, 10, 5) epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=18.13** / SSIM=0.3875 / LPIPS=0.4097（观测基线 PSNR=22.60）
- NFE=330（优化300+采样30）
- 耗时: 优化=43.60s + 采样=3.94s = 合计 **47.54s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（imagenet）
- 目的: A/B测试：官方ImageNet权重，运行时使用bicubic-antialias精确伴随H^T；其余配置保持原ZAPS默认值

## 结论 / 观察
（待填写）
