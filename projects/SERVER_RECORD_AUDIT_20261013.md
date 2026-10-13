# 已上传服务器记录审计：ImageNet 去重与下一步边界

日期：2026-10-13。只读审计与文档存档；没有新 GPU 实验，没有算法修改。

## 1. 同步已核实，不只是用户口头回报

- 仓库：`lzy75/ZAPS`；结果分支：`results/server-records-20261013`。
- 上传提交：`d2c8122086db7e05ecd00a94c01d2915e6125f3c`。
- 目录：`projects/server_records/20261013/`；共 2003 个文件，含 940 个
  `run.json` 和 38 个根目录实验日志。文件清单另列原始张量/图片的位置与大小，
  但这些二进制文件未上传；不是原始张量备份。
- `source_commit.txt` 为 `3ba2a599bfa8f20697eb9938bbf2f1718d4b8966`；
  `source_modified_paths.txt` 仅列 `projects/experiments/experiments.csv`。
- **上传时 HEAD 不等于各实验运行时版本。** 两任务原始 run 记录
  `b626296`、dirty=true；后续诊断分别保存自己的 revision/源文件指纹。
  当前本地核心及退化算子 hash 与完成的 guidance/clamp 审计所记一致：
  core `e66d72ec5c9b4b973d1b59445c297248b938fc088d802511c9fc6eb67317a427`；
  H 源码 `e25293451e787ef2fe52fd904436f6c33aadfdc522d2cb7de1b557e4628be696`。
  不能据此认证未保存 hash 的全部旧运行或所有 dirty 文件。
- 旧 run 中首次余弦常为 NaN（信号尚不可算）。Python JSON 解析可读，
  不把它们误判成 747 个失败实验；但 NaN 不是标准 JSON，其他解析器可能拒绝。
  936 个 run 未显式记录 status，需结合结果、日志、门控判断，不能全部标 complete。

## 2. 当前 ImageNet 未调参参考已补齐原始指标

来源：`results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/`
下 `run.json`、`metrics.csv` 和对应日志。00001.png、seed1001，30×10，
lr=.001，ζ/D 联合，fixed-small、eta1、last_opt，无状态调度。

| 任务 | 调度 | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| Gaussian Deblur | irregular 15/10/5 | 19.5987 | .3168 | .6505 |
| Gaussian Deblur | uniform 30 | 21.5069 | .5510 | .3432 |
| SR×4，exact transpose | irregular 15/10/5 | 19.9209 | .4798 | .5472 |
| SR×4，exact transpose | uniform 30 | 20.0485 | .5275 | .5358 |

SR 的 SSIM/LPIPS 此前聊天回报缺失，本次从上传 CSV 补齐。
原文 ImageNet Gaussian/SR 均值为 22.45/23.82；单图与数据集均值不可
解释成严格配对差值，当前仍是 local_reference，非作者实现完全一致认证。

同观测 DPS 成功 run 实际目录为 `dps_same_observation_20261008_235302`：
23.1528/.6301/.3999，1000 采样 NFE，另有 1 次 preflight。
较早 `...233903` 明确 status=failed，不再把该路径标作完成基线。
DPS30 两网格为 15.5670/16.6235；ZAPS 比它高 4.0317/4.8833 dB，但
300 与 30 NFE 不等预算，不是 Jacobian 的单因素对照。

## 3. 新增 0-NFE 排除：当前基线训练 norm 裁剪未触发

源码先调用 `clip_grad_norm_([zeta,D], max_norm=1)`，Adam step 后打印
`zeta.grad.norm()` 和 `D.grad.norm()`；Adam 不将这些梯度改写成参数步长。
当前日志两参数合并欧氏范数为：

```text
G_logged = sqrt(gz^2 + gd^2)
```

| 任务 / 调度 | 已记录轮数 | 最大 G_logged |
| --- | ---: | ---: |
| Gaussian / irregular | 10 | .0041593 |
| Gaussian / uniform | 10 | .4846505 |
| SR / irregular | 10 | .3735113 |
| SR / uniform | 10 | .1140193 |

即使加上日志四位小数的舍入上界，最大也仅 .4847123。
若有限梯度被阈值 1 缩放，裁剪后的联合范数应接近 1，而不是这些数值。
因此在这 40 轮参考运行中，norm=1 裁剪未实质触发，不能解释这次差距。

全量文本检索没有找到独立的 ζ/D norm 裁剪开关实验；但**不必为了补表
而重跑这份参考的开关消融**，已有日志足以降低其优先级。
结论不外推到状态调度、高学习率、mask 等其他配置。

## 4. 去重与完成状态

| 项目 | 原始记录确认 | 决策 |
| --- | --- | --- |
| 同观测 DPS1000、DPS30 方差 | 完成 run、身份与指标存在 | 复用，不重跑 |
| sampler parity | 精确门控曾失败；首步 FP32 x0 小误差，其余记录须按分量读 | 不把它写成完整 PASS 或几 dB 根因 |
| clip/clamp VJP | 完成的 `guidance_clipping_20261009_142630` | 已做，不重复同探针 |
| 硬 mask 冻结/匹配训练 | 完成的 `clamp_ablation_20261009_151823`，重建约 9.3 dB，门控通过 | 方案失败，不加入核心 |
| 早期 guidance L2 归一化 | 当前文件清单有输出图，数值见历史聊天：raw 18.9648、L2 16.5509 | 属 SR，不能冒充 Gaussian；不重复盲扫 |
| late-noise .75 | 冻结及匹配训练已有多份结果 | 修改策略的敏感性，不是原文修复；停止扫噪声 |
| ζ/D 训练 norm clip=1 | 无独立开关记录；当前 40 轮未触发 | 当前参考可排除，不追加同设置实验 |
| 首步真实梯度范数缩放 / 倍率扫描 | 本次记录未见可确认的完成 run | 未运行，不自动安排；先论证新增信息量 |

## 5. FFHQ 不是“同设置完全没问题”的反证

上传 `logs/ffhq_regression_2x2.log` 的同观测对照确认：
exact+learned 22.8224；exact+fixed 22.8447；legacy+learned 29.5963；
legacy+fixed 29.8071。legacy 是历史 bicubic 上采样引导映射，不是 H 的
精确伴随。高分与 exact 管线不是相同设置，不能拿它排除共享流程风险。
不以追分为由回退正确伴随，也不将 legacy 高分当作者实现一致证明。

## 6. 剩余重点与下一步的限制

Gaussian irregular loss .003059→.002969，D delta RMS 约 .000646；
高噪声已见局部 Jacobian 近似方向/尺度失配与中间饱和。
这些指向“引导近似及其参数优化链需要解释”，仍未锁定唯一代码错误。

静态核心核对：前向 correction 使用 wavelet B，当前 surrogate=false，
Tweedie 的 unroll 反向则对 eps stop-gradient。冻结模型权重不等于模型
对输入的 Jacobian 为零；但采用近似也不自动构成实现 bug。
此前 SR Eq.21 surrogate 消融几乎无收益，不能将相同 SR 开关再跑一遍，
也不能直接认定 Gaussian 换开关一定有效。

下一项应先完成公式/反向路径的只读对照，并说明剩余差异的证据。
若仍需 GPU，候选是复用 Gaussian 第10轮快照与固定 RNG，在少量 ζ
方向上比较实际 terminal loss 的有限差分与当前 unroll 参数梯度，区分
“本身梯度很小”“近似反向不能反映真实损失变化”。它不是重复 x_t
局部 VJP，也不是裁剪/倍率扫描；误差也不自动证明违反作者实现。
该候选尚未写入口、尚未运行，须先与用户确认必要性、预算和判据。

保持 ImageNet 原文主要超参数参考、FFHQ 及 batch 核心不变。
初始化文献和固定 30 步状态表达优化仍保留在后续队列。
