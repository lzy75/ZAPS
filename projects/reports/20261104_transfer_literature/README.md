# 倍率模块：思路参考与直接迁移基线

2026-11-04。原文PDF已下载到本目录 `pdf/reference/`、`pdf/baseline/`；版本、来源、文件大小与SHA256见 `papers.json`。PDF约183MiB，仅在Mac本地保存，不推送Git，避免再次占满服务器根盘；服务器只同步小型源码与索引。`download_sources.py --papers-only` 可按固定版本重新下载。

## 一、解决思路参考（本轮不叠加这些机制）

|文献|版本/状态|用途|本地PDF|
|---|---|---|---|
|SAIP: A Plug-and-Play Scale-adaptive Module in Diffusion-based Inverse Problems|arXiv2509.24580v1；ICASSP2026会议记录|最直接的在线倍率竞争工作；通过先验/似然分数关系调整先验权重，不等同于当前观测空间噪声收缩|[SAIP](pdf/reference/SAIP.pdf)|
|Jacobian-Aware Posterior Sampling for Inverse Problems|2511.18471v3；PDF标注TMLR06/2026|与此前方向/投影路线的重合边界；这是AdaPS早期预印本的后续版本，不重复计数|[JAPS](pdf/reference/JAPS.pdf)|
|Getting Your Guidance Weights Right in diffusion and flow-matching posterior sampling|2610.03503v1，2026-10预印本|SIMCA离线校准先验/引导两项系数；可作强固定时间表对照|[SIMCA](pdf/reference/SIMCA.pdf)|
|Diffusion State-Guided Projected Gradient for Inverse Problems|2410.03463v5；ICLR2025|状态结构与可信方向；不是本轮新增投影模块|[DiffStateGrad](pdf/reference/DiffStateGrad.pdf)|
|Variational Control for Guidance in Diffusion Models|2502.03686v2；ICML2025|终局目标与轨迹扰动的联合代价；不直接加入昂贵内部优化|[DTM](pdf/reference/DTM.pdf)|
|Calibrated Test-Time Guidance for Bayesian Inference|2602.22428v2；ICML2026|区分重建效果与后验校准；多候选估计有额外计算|[CBG](pdf/reference/Calibrated_Bayesian_Guidance.pdf)|

## 二、直接迁移基线（本轮执行队列）

|实验|基线|配置|每run NN/输入VJP/干净空间算子梯度|本地PDF|
|---|---|---|---|---|
|A82|MPGD without projection|原x0ddim100、eta=.5；两个任务均用论文附录D.1明确的100步倍率5|100/0/100|[MPGD](pdf/baseline/MPGD.pdf)|
|A83|DMPS|原DDPM1000、learned_range、scale1.75；原生bicubicSR与Gaussian配置|1000/0/0|[DMPS](pdf/baseline/DMPS.pdf)|
|A84|ΠGDM作者团队RED-diff公开实现|原默认50点、eta0、awdTrue、引导权重1；原生平均池化SR与5tapGaussian，非MPGD的bicubic/61核|50/50/0|[ΠGDM](pdf/baseline/PiGDM.pdf)|

每个实验：FFHQ旧三图00226/00443/00963×SR4/Gaussian×BASE/SHRINK=12runs。三者共36runs、13800NN前向、600输入VJP、1200干净空间算子梯度；真实载入探针单列。候选每步额外两次H，三组共13800次额外H；没有新UNet调用或额外UNet反向。

模块完全复用 `noise_signal_shrink.select`，连续非负倍率，不增加上限、系数、调参或失败回退。仅改各宿主显式物理校正的倍率。基线与候选共用该宿主自己的H/y、图像与噪声，交替运行顺序、完整计时并核对RNG/时序。

**ΠGDM限定**：下载原文§3.1明确区分含噪协方差公式(7)和无噪伪逆公式(8)；当前作者团队 `algos/pgdm.py` 是伪逆路径，不含噪协方差求解。A84只回答“当前模块能否改善这份公开原生伪逆实现的含噪输入结果”，不冒称含噪ΠGDM原文全量复现，也不将收益全归为在线状态优势。

**MPGD限定**：下载原文附录D.1写100步倍率5，而发布的SR脚本写4、Gaussian发布CLI默认10。本轮保持原代码循环，两个任务统一采用论文明确的5，并在plan保留差异，不按质量挑选倍率。这仍是三图本地配对参考，不是论文千图均值复现。

只有每宿主每任务PSNR均值至少+.1dB、至少2/3图提升、SSIM不降、LPIPS不升且配对门控通过，才列正向开发候选；质量权衡单列。三图用于迁移可行性，不承担独立泛化或在线反馈必要性结论。

本机无Torch/CUDA：4项源码/配置/面板/CLI检查通过；服务器入口必须运行真实Torch/CUDA数值门控。当前还没有A82—A84模型结果。

## 服务器命令

先在 `/home/lzy/ZAPS` 同步本次master提交，再逐个运行，避免三套程序重复占同一GPU：

```bash
bash /home/lzy/ZAPS/projects/run_native_ray_transfer_20261104.sh --experiment A82 --gpus 0,1
```

```bash
bash /home/lzy/ZAPS/projects/run_native_ray_transfer_20261104.sh --experiment A83 --gpus 0,1
```

```bash
bash /home/lzy/ZAPS/projects/run_native_ray_transfer_20261104.sh --experiment A84 --gpus 0,1
```

每项内部可将SR/Gaussian放在两张已分配GPU；只有一张合格GPU时串行。断线后后台协调器继续，原命令重接；资源不足在未开始的run前暂停。失败不自动重试。附加`--status`只读状态，`--archive-only`只重试归档上传。

结果根：`/data0/lzy/ZAPS-results/diag_native_ray_transfer/A82_20261104`（A83/A84相同命名）。结束输出quality/full-clock/实际成本，自动push各自轻量results分支；PT/PNG仍在服务器，结果不合master。
