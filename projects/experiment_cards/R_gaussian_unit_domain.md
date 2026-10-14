# R：ImageNet Gaussian 测量域对齐候选

日期2026-10-14；状态completed（用户已回传服务器摘要，完整父记录由下项入口验真）。
本项已获用户“继续实验”确认。本机新增3项契约测试通过，3项张量测试
因无PyTorch跳过；服务器必须先执行该测试模块，不能把跳过写成数值通过。

## 目标、去重与允许结论

已完成原设置缓存确认：19.5987/.3168/.6505，新增UNet NFE=0。
原设置的观测σ=.05在[-1,1]，物理[0,1]等效σ=.025；论文§4.1写
预处理[0,1]、观测σ=.05。本次验证这个**公开协议缺口**对Gaussian的影响。
不是追分、改初值或状态收益实验，不保证复现作者未公开的内部实现。

旧 `diag_value_domain_exact.py` 是SR三图，unit比model均值低2.9768dB；
旧单纯σ翻倍探针用ζ-only、lr=.05，也不是本次Gaussian原设置。
因此本项新增Gaussian仿射物理测量，不重跑SR、不扫描噪声。

父记录：
`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/original_reference_audit_20261014_123322_742681/audit.json`。
入口验证completed、配置、源码/权重/观测/图片/输出/快照hash、旧门控。
旧生产FP32 Tweedie比较失败及参数反向未认证项**保留**。

## 改动与固定条件

仅一项成套协议改动：测量从model域改为unit物理域，UNet仍在model域。

```
A_u(x) = H((x+1)/2) = .5 H(x) + .5 H(1)
J_Au^T(r) = .5 H^T(r)
n_old = y_old - H(x_gt)
xi = n_old / .05
y_unit = A_u(x_gt) + .05 xi
```

Gaussian zero padding的H(1)不是恒为1。不得使用(y_old+1)/2，也不得
对残差加仿射偏置。不clip观测，不额外随机加噪。
同一归一化噪声场，但物理噪声从σ=.025变.05，**不是同观测、不是单纯换坐标**。
引导与loss单位也随此协议变化，不声称单独隔离了噪声或梯度尺度。

固定00001.png、seed1001、同权重、61×61 sigma3 zero-padding Gaussian；
全部ζ初值=.2，全部D=.2；联合Adam lr=.001/betas(.9,.999)/eps1e-8；
paper15/10/5原整数网格999→0、每轮30步共10轮；fixed-small DDPM eta1；
db4 level3；原Tweedie clamp、surrogate关闭、MSE mean、gradnorm clip1、
ζ>=1e-6，输出第10轮更新前last_opt。无额外采样、在线状态或GT选择。
不改共享核心、默认配置、FFHQ、batch或旧诊断。
GT只用于合成测量和预定最终输出评价；optimize传x0_gt=None。
记录当前Python/Torch/CUDA/GPU/源码身份；指标沿用uint8 PSNR/SSIM、alex LPIPS。

## 门控与预算

- 缓存原baseline不重训、不回放，0新增NFE；新候选仅300方法NFE。
- 无额外UNet门控/采样；总新增UNet上限300，任何失败保存实际已用数量。
- 0-UNet门控：仿射边界等式、实际autograd VJP、线性增量内积伴随、
  测量构造及噪声恒等；相对容差2e-5（conv FP32），无事后调容差。
- 0-UNet RNG预检恢复原x_T及第10轮噪声位置，score不得消费采样RNG；
  实训检查同网格/300调用、10轮、x_T/epoch10/结束RNG、无模型权重梯度。
- 只复用原输出，经hash与原指标/终端loss检查；原生域MSE不可跨域直接比大小。
- 非有限张量、梯度、loss、身份/门控不通过：停止，不解读画质结论。

## 事先判据及处理

相对缓存原设置，PSNR≥+.2dB且SSIM≥+.005且LPIPS≤-.005：
支持该测量协议对当前单图综合质量有实质影响；仍不自动称作者复现。
三项均未实质改善（PSNR≤+.05、SSIM≤+.001、LPIPS≥-.001）：
无综合质量支持；不再扫σ/初值，不因原分更高就称其物理协议与论文相同。
其余为取舍/不确定，三指标完整报告；不以GT挑“最佳物理域”。
所有结果先讨论再决定后续：协议确认靠文献/实现证据，不靠挑高分；
本实验不自动解锁批量或状态训练，也不能排除剩余近似反向/边界等未公开项。

## 执行与归档

`python -u utils/diag_imagenet_gaussian_unit_domain.py --reference-dir <上述父目录> --device cuda`

保存到原trace下新 `gaussian_unit_domain_时间戳`，含card/run.json/summary.csv、
新y、旧数值噪声与标准噪声、last_opt、更新前快照及hash；不覆盖旧张量。
已回传21.7256/.5387/.3958，相对原缓存+2.1269dB/+.2220/-.2547，
新增300UNet NFE。分类支持测量协议敏感性，不认证作者实现或状态收益。
完整边界及run路径见[结果摘要](../GAUSSIAN_UNIT_DOMAIN_RESULT_20261014.md)。
本卡以上运行前假设与判据保留，不据结果更改。
