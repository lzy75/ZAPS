# 实验 20260817_123622_super_resolution_ffhq_00000

- 代码版本: `db8f248` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00000.png`
- 关键参数: steps=20 schedule=[10, 7, 3] epochs=5 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.66** / SSIM=0.8629 / LPIPS=0.1091（观测基线 PSNR=28.75）
- NFE=120（优化100+采样20）
- 耗时: 优化=6.91s + 采样=1.12s = 合计 **8.04s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **super_resolution**（ffhq）
- 目的: NFE100 固定基线 5ep20 10-7-3 四任务

## 结论 / 观察
（待填写）
