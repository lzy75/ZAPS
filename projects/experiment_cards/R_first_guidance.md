# R-first-guidance：首步幅度的单因素因果验证

日期：2026-10-13；工作线R；mechanism_only。
用户在讨论唯一干预、60诊断NFE及判读后授权继续实现与运行。
状态：独立脚本实现；真实服务器结果待运行。不改变共享核心/FFHQ/batch。

## 目标、父记录与去重

可否定假设：首步近似引导幅度偏大，是当前ImageNet Gaussian画质的重要限制。
完成的终端ζ审计恢复PSNR19.598683、loss .002969034，门控通过；三个
尺度的三个ζ都未出现异号，但t999 FD跨度约29.7%，不能据此定位bug。
原轨迹下一状态RMS17.413、x0裁剪99.90%；同状态VJP近似/精确范数
约4.9963。该幅度敏感性还未干预验证。

父目录：
`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/`。
使用已完成的`guidance_clipping_20261009_142630/audit.json`和其下同级
完成的`parameter_gradient_*/run.json`；原始张量快照和观测仍在服务器。
不重复硬mask（已失败约9.3dB）、全程L2归一化、晚期降噪、lr扫描。

## 唯一变量与固定条件

两臂恢复同一第10轮采样前ζ/D快照（9次Adam更新后）：

- raw_baseline：原完整30步。
- first_norm_matched：仅t999 correction乘a，其余29步原公式不变。

`a=||g_exact||/||g_approx||`，来自同状态已完成VJP，非人工0.2、非GT
拟合、非DPS scale=.3。目标统一为`.5*sum((y-H(clamp(Tweedie)))²)`。
不替换方向、不补mask，不把这项干预当原文规定。
在线验证当前g_approx范数与父审计相同；线性B下只缩放首步H^T残差
等价于缩放完整引导，不改无条件posterior、ζ或D。

ImageNet Gaussian00001、seed1001、原H61/sigma3/zero padding、model-domain
sigma_y=.05、官方fp32权重、paper15/10/5、fixed-small DDPM/eta1、原clamp、
db4 level3、ζ/D原快照，全部锁定。源训练lr=.001、jointζ/D、30×10；本次
无训练/优化器/投影/梯度裁剪；源300方法NFE不写成新的30NFE方法。
CPU和采样GPU RNG均恢复，x_T和每一步正态抽样相同。其他GPU数量可不同。
所有身份hash、runtime、resolved配置、原始输出、逐步指纹都记录。
指标沿用原uint8 PSNR/SSIM、alex LPIPS；GT在干预完成后仅用于评价两臂。

## 预算、门控与失败条件

2×30=60新诊断NFE；复用父VJP不新增模型调用。所有实际UNet调用hook计数，
失败调用亦记录。额外一次B变换不是score NFE，计入耗时。
原路径与存档30步指纹固定容差：atol1e-6/rtol2e-5，clip_atol2e-5。
loss与存档abs<=1e-8+2e-5*abs(loss)。首步范数rtol2e-5/atol1e-6。
两臂first-input指纹必须相同，grid/NFE相同，RNG尾状态必须完全相同。
显式patch仅t999，异常自动恢复算子方法和参数。改变身份/非有限数/门控
失败立即停止，写failed JSON，不解读画质。父completed不等于作者一致。

## 预注册判读（工程门槛，不是论文复现标准）

早期缓解：下一状态RMS至少下降50%，下一步clip比例不增加超过2e-5。
综合正向：同时dPSNR>=.2dB、dSSIM>=.005、dLPIPS<=-.005。

| 结果 | 后续处理 |
| --- | --- |
| 早期缓解且综合正向 | 支持首步幅度敏感；讨论来源/匹配训练，不直接同步核心 |
| 早期缓解，但dPSNR<=.05、dSSIM<=.001、dLPIPS>=-.001 | 无最终画质支持；停止该倍率路线，不扩大扫描 |
| 指标混合或未缓解 | 不确定；不挑最好指标，不自动重训或扩大搜索 |
| gate failed | 先修身份/接口；不解读算法 |

同图单次因果敏感性不能宣称作者bug、原文基线成功、跨图泛化或状态收益。
无论结果都保留原始baseline，后续先与用户讨论。

## 运行、输出与同步

```bash
cd /home/lzy/ZAPS/projects
python -u utils/diag_zaps_first_guidance.py \
  --guidance-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/guidance_clipping_20261009_142630 \
  --device cuda
```

未给--parameter-dir时，选同source/identity的最新completed参数审计，打印
并保存所选绝对路径；显式参数可锁定具体父目录。无兼容记录立即拒绝。
新目录`first_guidance_时间戳`，保存run.json、summary.csv、trace.csv、两张
原始输出.pt和实验卡；不覆盖父记录。脚本只输出预定义两臂，不选最好图。
CPU协议/可选tensor单测与真实GPU结果分别记录；正式single/batch/FFHQ
不解锁。本次真实结果待用户服务器运行。
