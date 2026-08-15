# 实验 20260815_162522_super_resolution_ffhq_00000

- 代码版本: `3c0b512` (branch=master, dirty=True)
- 任务 / 数据集: **super_resolution** / ffhq，图像 `00000.png`
- 关键参数: steps=30 schedule=[15, 10, 5] epochs=10 lr=0.001 zeta_init=0.1 d_init=0.2 wave=db4 level=3
- 指标: **PSNR=28.84** / SSIM=0.8695 / LPIPS=0.1081（观测基线 PSNR=28.75）
- NFE=330（优化300+采样30）  耗时=0.4 min

![recon](recon.png)

## 结论 / 观察
对比实验 B组 
实验	num_epochs	num_steps	schedule	优化 NFE	图片数
A	15	20	10 7 3	300	20
B	10	30	15 10 5	300	20
C	5	60	30 20 10	300	20
