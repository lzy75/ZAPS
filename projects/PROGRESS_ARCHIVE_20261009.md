# ZAPS 阶段进度存档：ImageNet 复现与状态感知主线

存档日期：2026-10-09（Asia/Shanghai）

代码基准：`3ba2a59`（本存档提交前；含独立 clamp 消融，未修改正式核心）

证据来源：服务器运行结果由用户回报；本地代码及历史文档核对。
最新服务器原始张量/日志未复制到本地，本文件是结论与数值快照，不是完整数据备份。

## 1. 当前结论与阶段位置

**ImageNet 的权重调用与部分基础算子已核查，但尚未完成原文水平复现；
FFHQ 状态感知主线已完成小规模配对验证，得到初步 PSNR 增益，尚未得到
PSNR/SSIM/LPIPS 全面改善。最新硬裁剪掩码方案失败，不进入正式基线。**

| 工作线 | 已完成 | 尚未完成 |
| --- | --- | --- |
| ImageNet 复现 | 正确权重验证；模型输出对齐；算子/小波核查；同观测 DPS 对照；方差与裁剪诊断 | 近似 Jacobian／高噪声引导的因果定位；原文配置与平均水平验证 |
| FFHQ 固定基线 | 学习率及初始化筛选；多种初始调度；10 图配对比较 | 精确算子约定下的正式复现认证；独立图像/种子确认 |
| 状态感知创新 | 代码门控；物理残差与余弦组合迭代；多初始调度；10 图验证 | 感知质量退化的解决；跨任务、跨数据集验证 |

不能把单图结果与论文 1000 图均值直接比较，也不能把诊断干预的收益
计入状态感知创新收益。以下所有“增益”都说明参照与实验范围。

## 2. 原文参照与比较边界

来源：[ECCV 2024 ZAPS 原文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114.pdf)。

| 数据集 / 任务 | 原文 ZAPS PSNR | SSIM | LPIPS | 参照位置 |
| --- | ---: | ---: | ---: | --- |
| ImageNet / Gaussian Deblur | 22.45 | 0.682 | 0.225 | Table 6 |
| ImageNet / 4× SR | 23.82 | 0.718 | 0.186 | Table 6 |
| FFHQ / Gaussian Deblur | 26.06 | 0.757 | 0.121 | Table 1 |
| FFHQ / Random Inpainting 70% | 27.79 | 0.813 | 0.078 | Table 1 |
| FFHQ / 4× SR，Uniform | 26.29 | 0.729 | 0.119 | Table 4，300 NFE |
| FFHQ / 4× SR，Irregular | 26.63 | 0.768 | 0.104 | Table 4，300 NFE |

原文定量主设置为 30 次模型调用/轮 × 10 轮 = 300 NFE，15/10/5
不规则网格，ζ 与 D 联合学习，db4 正交小波。原文直接比较不规则/均匀
调度的表格在 FFHQ SR 和随机修复上，不能推导每张 ImageNet 去模糊图
都必须是不规则更好。当前图像集合、值域/噪声、引导算子和评估约定仍
存在需核实的差异；本地绝对 PSNR 高于原文也不构成超过原文的证据。

## 3. ImageNet：已完成的复现排查

### 3.1 权重、模型与基础算子

- 官方上传文件：2,211,383,297 bytes；MD5 `fd9dd2335b8736d521de0aed54bd90ca`；
  SHA256 `a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`。
  该 MD5 与用户官方本地文件一致，支持上传完整性；结构严格匹配不能单独证明来源正确。
- 参数量 552,814,086，输出 6 通道；同 x_t/t/权重下，DPS 与当前模型
  在 t=999/667/333/0 的输出误差均为 0。此前坏权重造成的生成失败已处理，
  用户确认新权重生成/DPS 可运行；不能再将当前差距笼统归因于未加载权重。
- SR 历史 transpose 检查失败：范数比 16.8421、相对误差 15.8459，
  内积伴随误差 0.8903；精确伴随是一个确实需要区分的实现问题。
- db4 DWT 核查通过：重建相对误差 8.21e-7，能量比约 1，
  内积伴随误差 1.29e-7。仅证明该小波实现的正交性，不证明 Hessian 近似正确。
- 早期 SR 中 D=0、固定/重抽样噪声、Eq.21 surrogate 等消融未得到明显
  改善；这些结果只约束当时 SR 的设置，不能全局排除 Gaussian Deblur 的问题。
- ImageNet SR 单图学习率从 .001 到 .05 时，PSNR 19.5436→20.8243；
  ζ-only 为20.8344、D-only为19.3083。说明该单图收益主要来自 ζ 的适配，
  不是冻结 D 的理由。正式 ZAPS 仍保留 ζ/D 联合学习。
- 噪声值域与 unit-domain 试验没有改善当时重建；它们涉及不同物理噪声/
  目标尺度，不能据此宣布值域与原文完全一致。暂停重复值域/学习率盲扫。

### 3.2 当前可定位的任务基线

ImageNet SR 历史 10 图：PSNR mean=19.3007，SSIM=.4350，LPIPS=.4732，
observed PSNR=22.8539。不同子集与原文23.82不可作严格配对差值，
但现有结果不足以宣布 SR 复现完成。

2026-10-08 同一新图 `00001.png`、seed=1001，lr=.001，30×10、ζ/D 联合
学习，fixed-small DDPM、eta=1，无状态感知：

| 任务 | Irregular PSNR | Uniform PSNR | Irregular − Uniform |
| --- | ---: | ---: | ---: |
| Gaussian Deblur | 19.5987 | 21.5069 | −1.9082 |
| 4× SR（exact transpose） | 19.9209 | 20.0485 | −0.1275 |

Gaussian 用61×61、sigma=3核，当前 model-domain measurement sigma=.05；
ζ初值=.2、D初值=.2。SR 的ζ初值=.1。输出采用 `last_opt`：第10轮采样
输出，不是第10次 Adam 更新后重新采样。评价 PSNR/SSIM 沿用 uint8
量化约定，float PSNR另记；不可混淆日志中相差约 .02 dB 的两种口径。

### 3.3 同观测 DPS 对照：已完成

Gaussian 单图同图像、H、y、x_T，不重新加测量噪声。DPS1000 使用官方
采样器和 PS，scale=.3；H为本次存档 Gaussian 算子，不冒称 stock DPS
退化算子也逐项一致。

| 方法 / 调度 | PSNR | SSIM | LPIPS | 方法 NFE |
| --- | ---: | ---: | ---: | ---: |
| DPS official 1000，learned-range | 23.1528 | 0.6301 | 0.3999 | 1000 |
| DPS30，Irregular，fixed-small | 15.5670 | — | — | 30 |
| DPS30，Uniform，fixed-small | 16.6235 | — | — | 30 |
| ZAPS，Irregular，fixed-small | 19.5987 | 0.3168 | 0.6505 | 300 |
| ZAPS，Uniform，fixed-small | 21.5069 | 0.5510 | 0.3432 | 300 |

ZAPS 对同网格 DPS30 高4.0317/4.8833 dB，但300与30 NFE不等预算；
不足以把增益归因于近似 Jacobian 本身。Uniform 的 LPIPS 比 DPS1000
更好，但 PSNR/SSIM仍差；不能概括为所有指标下 ZAPS都更差或更好。

DPS30 learned-range − fixed-small：Irregular/Uniform 的 dPSNR
为−.0097/−.0080，dSSIM为−.0069/−.0022，dLPIPS为−.0641/−.0134。
在当前单图 DPS30 中，方差选择不是几 dB PSNR差距的主要解释；
不能外推为 ZAPS/其他步数的方差已完全排除。

### 3.4 轨迹、晚期噪声与采样公式：诊断完成，不能当作修复

- 原始优化第10轮：Irregular在t=333的晚期x0峰值21.7661，到t=0为
  19.5790（下降2.1872）；Uniform在t=344峰值22.0823，到t=0为
  21.4644（下降.6180）。峰值是GT诊断，未用于早停/挑图。
- 低噪声区保持引导、仅把随机转移噪声乘 .75，冻结回放可提高
  Irregular PSNR至24.2687；同策略重新优化为24.2327。这属于修改后的
  采样策略，不是原文基线、已确认公式修复或状态感知增益。
- 减少晚期噪声仅证明轨迹敏感性；现阶段正式 eta 保持1，不继续降噪搜索。
- 官方 sampler 对齐核查的严格总结果曾为FAIL：首步t=999的Tweedie
  x0相对误差4.09e-5、最大误差8.73e-5；时间映射、其他采样分量和后续
  步通过。支持FP32等价公式计算顺序差异，不能写成完整PASS或据此
  认定这是数 dB差距的原因。
- CUDA可见数量变化导致的存档RNG恢复报错已在独立诊断接口处理；
  保留采样卡映射、重复误差及存档轨迹指纹门控，不改算法或放宽容差。

### 3.5 裁剪／梯度核查与硬掩码：最新完成结果

固定原轨迹输入，统一目标 `L=.5*||y-H(clamp(x0_raw))||²`。当前近似
`B(v)` 与加入掩码的 `B(Mv)` 对比，M必须在非对角B之前。
高噪声、近100%裁剪的探针中，掩码提高局部方向一致性：

| 网格 / t | 裁剪比例 | cos raw→mask | 范数/真实 raw→mask |
| --- | ---: | --- | --- |
| Irregular / 916 | 99.9% | .0325→.9463 | 105.9494→2.2178 |
| Uniform / 930 | 100.0% | .0141→.9945 | 163.7512→1.7069 |

但 t=999 几乎无裁剪，余弦仍约.69、近似/真实范数比约5。
局部梯度更近只是诊断，不能预言全轨迹改善，也不能判定原文遗漏mask。

同网格/权重/y/x_T/DDPM随机数、lr=.001、eta=1、ζ/D联合学习的消融：

| 调度 | 策略 | PSNR | SSIM | LPIPS | 相对原基线 dPSNR |
| --- | --- | ---: | ---: | ---: | ---: |
| Irregular | raw baseline | 19.5987 | .3168 | .6505 | 0 |
| Irregular | frozen raw→mask | 9.3894 | .1147 | .9711 | −10.2092 |
| Irregular | matched mask training | 9.3749 | .1147 | .9736 | −10.2238 |
| Uniform | raw baseline | 21.5069 | .5510 | .3432 | 0 |
| Uniform | frozen raw→mask | 9.3176 | .1108 | .9474 | −12.1892 |
| Uniform | matched mask training | 9.2836 | .1099 | .9506 | −12.2233 |

| 调度 / 策略 | 训练首轮→末轮物理MSE | 最终未裁剪RMS | 最终超界比例 |
| --- | --- | ---: | ---: |
| Irregular / raw | .003059→.002969 | .4468 | .01% |
| Irregular / frozen mask | 不训练 | .8426 | 2.03% |
| Irregular / matched mask | .054105→.052217 | .8419 | 2.03% |
| Uniform / raw | .003240→.003136 | .4305 | .00% |
| Uniform / frozen mask | 不训练 | .8501 | 1.86% |
| Uniform / matched mask | .058043→.059500 | .8533 | 1.86% |

**决策：否定“直接补硬mask即可修复”的方案；不同步至正式核心、FFHQ或
批量代码。** 物理MSE变差约18–19倍，说明不仅是视觉/评估差异。
约2%最终超界不能解释全部9 dB误差；中间Tweedie裁剪比例与最终输出
超界比例是不同量。首末轮loss也不能证明梯度为零，随机优化轨迹与
中间饱和机制尚需区分。不得继续将mask称为已定位的根因或成功修复。

## 4. FFHQ：参数与初始调度主线

当前实验基座为FFHQ 4× SR：30×10，ζinit=.1、Dinit=.2，ζ/D联合
Adam学习，lr=.01，fixed skip variance、`legacy_bicubic`引导映射。
**legacy是历史映射，不是正确H的精确伴随，不能据此认证原文复现。**

- 同图历史回归：exact transpose约22.82 dB，legacy约29.60；历史
  legacy+fixed variance约29.81。方差效应远小于该映射效应。
  说明设置/有效引导尺度很重要，不是两个数据集使用不同ZAPS理论。
- 学习率10图均值：.001→27.8837；.005→28.1695；.01→28.2565。
  选择.01作为当前FFHQ实验参数；ImageNet上.05单图更好不能直接迁移。
- ζ/D初始化微调收益小，没有证据支持继续扩大初始化扫描。

独立的10图固定调度比较（不要与上面学习率批次混算增益）：

| 初始调度 | PSNR mean | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| paper 15/10/5 | 28.1905 | .8360 | .1265 |
| uniform 30 | 28.3043 | .8438 | .1303 |
| Karras rho7 | 28.3348 | .8315 | .1551 |

Uniform优于paper的PSNR约.1138，但LPIPS更差；Karras rho7的PSNR更高，
感知代价也更大，不能说uniform或某种不规则调度在所有指标下最好。
其他单图网格包括power2/3、uniform sigma/logsigma/logSNR、Karras
rho3/5/7。uniform sigma很差，过度集中power3退化；没有统一最佳网格。
Karras是把噪声网格映射到现有DDPM，**不是完整EDM模型/求解器复现**。

## 5. 状态感知创新：当前设计、证据及边界

### 5.1 当前保留设计

保留物理一致性与余弦两个指标；不以单指标替代研究主线。
采用“初始网格＋第1轮参考曲线＋后续状态微调”，不额外调制ζ。
每种初始网格自己的null/refined配对，同观测/初始化/按步位置共享噪声，
固定300NFE；自适应路径响应为0必须退化到固定路径，检查时间步、
ζ/D索引、单unroll输出/梯度及最终输出误差。CUDA随机性门控只验证
实现等价性，不证明微小PSNR增益显著。

令k为反向采样位置，参考量来自第1轮相同位置，不是同一个真实t的GT：

```text
r_k = ||y - H(x0_hat,k)||_2
delta_k = x0_hat,k - x0_hat,k-1
c_k = cos(delta_k, delta_k-1)
u_k = log(max(r_k, eps) / max(r_ref,k, eps))
e_r,k = tanh((u_k - m_k-1) / s_r)
m_k = beta*m_k-1 + (1-beta)*u_k
e_c,k = tanh((c_ref,k - c_k) / s_c)
q_k = 1 - gamma*max(0, -sign(e_r,k)*e_c,k)     # veto_only
modifier_k = clip(1 - lambda*e_r,k*q_k, .8, 1.2)
h_base,k = t_k * (t_nom,k - t_nom,k+1) / max(1, t_nom,k)
h_k = integer_budget_projection(h_base,k * modifier_k)
```

首次去趋势项取中性；余弦不可算时取中性。物理信号决定放大/缩小，
余弦只否决冲突，不能翻转方向；末步强制t=0，留足互异整数网格与
剩余调用预算。状态标量detach，ζ/D仍学习，不对模型权重微调。

当前候选实验参数：lambda=.20、gamma=.35、beta=.8、s_r=.15、s_c=.20，
步长边界[.8,1.2]、warmup1轮。**这些是最新候选运行参数，不是所有CLI
默认值**；复跑必须核对run.json和显式参数。

### 5.2 已完成的10图配对结果

| 初始网格 | null PSNR | refined PSNR | mean dPSNR | dPSNR std | 改善数 | mean dSSIM | mean dLPIPS |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Uniform30 | 28.3075 | 28.3396 | +.0321 | .0629 | 7/10 | −.0002 | +.0017 |
| paper15/10/5 | 28.1869 | 28.2276 | +.0407 | .0749 | 6/10 | +.0001 | +.0004 |
| Karras rho3 | 28.1686 | 28.3083 | +.1398 | .0856 | 9/10 | +.0009 | +.0067 |

最佳**状态增量**候选是Karras rho3，平均+.1398且9/10胜；但其绝对
PSNR未超过refined Uniform，且LPIPS退化。因此只是小样本PSNR证据，
不称为全面提升、稳定跨数据集泛化或超过原文。不用最优单图+.2376
代替这张10图均值表。

较早的硬残差阈值、EMA相对下降、软归一化、状态到ζ乘法、posthoc权重
没有形成可靠的综合增益；保留为机制排查，不作为最终方法成果。
不能因这些映射失败就删除两个状态指标。

### 5.3 扩展与未完成项

代码支持global/detail/multiscale余弦。拟通过db4去最粗LL的细节增量，
构造 `c_multi=(1-alpha)*c_global+alpha*c_detail`，alpha=.7作为候选，
检查是否减少LPIPS代价。**本存档没有可确认的FFHQ新配对结果，状态是
待验证，不是已获得提升。**

ImageNet换图/随机修复低步数探索（每轮步数10/15/20/30，均10轮）：

| 步数/轮 | .001固定→状态 | .05固定→状态 | .05状态增量 |
| --- | --- | --- | ---: |
| 10 | 11.5651→11.5775 | 12.2458→13.1808 | +.9349 |
| 15 | 11.3757→11.3730 | 15.1230→15.6887 | +.5657 |
| 20 | 16.8764→16.6937 | 14.8150→14.2925 | −.5224 |
| 30 | 23.0825→22.8704 | 14.6549→14.4330 | −.2219 |

这里只支持该探索设置10/15步有正增量，绝对质量仍差、20/30步负增量。
不同步数对应100/150/200/300 NFE，不是等预算比较；不是ImageNet原文
Gaussian/SR复现，不与FFHQ 70%修复27.79混用。

## 6. 下一步顺序与进入下一阶段的条件

### P0：ImageNet首步引导因果核查（提议，尚未执行）

复用同观测/权重/存档参数，原引导为对照；只在t=999将原方向的范数
对齐同输入真实梯度，其余步骤、随机数、网格、lr、eta全部不变。
既不使用GT选缩放，也不补硬mask、不重训或降晚期噪声。目的是判断
“首步约5倍幅度失配”是否影响后续恢复，不预设收益、不开新广泛搜索。
若有改善，仅支持幅度嫌疑，不排除方向/Jacobian近似/初始化交互，
更不能直接认证原文实现。当前尚没有该实验脚本或服务器结果。

### P1：基线与创新收益分开确认

基线的算子/值域/裁剪位置/引导近似及输出约定固定后，再做10–20张
固定子集及少量重复种子，不逐图选择参数。若仍无法严格恢复原文，
明确称为“本地实现基线”，保留官方DPS同观测对照，不偷换论文基线。

### P2：继续FFHQ状态感知，不扩散无方向参数扫描

保留物理＋余弦、参考曲线与veto框架；优先单变量检查高频/多尺度余弦
是否降低LPIPS退化，固定lr=.01和ζ/D联合学习。通过单图门控后才做
固定10–20图配对，报告相对各自null以及最强固定网格的绝对结果。

创新阶段的工作门槛（不是论文规定）：平均PSNR正增量超过实测重复
误差，改善占多数，SSIM/LPIPS没有稳定退化；独立图像/种子确认后再
做其他任务、数据集和低预算验证。当前Karras候选未通过综合质量门槛。

### P3：等NFE步数—轮数实验（后置，尚未执行）

先固定基线再比较20×15、30×10、60×5（均300NFE）等预算分配，
状态/null在每个设置内配对。不能预言1000步必然更好：ZAPS1000步
×10轮为10000NFE，与DPS1000不等预算；1000步×1轮的last_opt又没有
利用第一次参数更新后的输出。这不是当前优先实验。

## 7. 记录位置、脚本与恢复入口

服务器主实验：

```text
/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549
  optimized_trace_gaussian_deblur_20261008_181857/
    *_last_unroll_state.pt
    audit.json
    dps_same_observation_20261008_233903/
    dps_lowstep_variance_20261009_121908/
    guidance_clipping_20261009_142630/
    clamp_ablation_*/
```

clamp子目录确切时间戳尚未由用户回报；以status=complete的run.json核实，
不要按名称猜测完成状态。服务器原始文件是run.json/audit.json、CSV、
measurement.pt、snapshot/RNG、PNG与输出张量；不删除或覆盖。

| 用途 | 仓库入口 |
| --- | --- |
| 两任务/两调度核查 | `utils/diag_zaps_paper_task_schedule.py` |
| 原始第10轮快照与轨迹 | `utils/diag_zaps_optimized_trace.py` |
| 官方DPS同观测 | `utils/diag_dps_same_observation.py` |
| DPS30网格×方差 | `utils/diag_dps_lowstep_variance.py` |
| 裁剪真实VJP | `utils/diag_zaps_guidance_clipping.py` |
| 已失败的独立硬mask消融 | `utils/diag_zaps_clamp_ablation.py` |
| FFHQ参考曲线单图/多图 | `utils/diag_ffhq_profile_schedule.py` / `utils/diag_ffhq_profile_multi.py` |
| ImageNet低步数探索 | `utils/diag_imagenet_lowstep_state.py` |

相关历史：[基线审计](BASELINE_TASK_SCHEDULE_AUDIT.md)、
[状态实验历史](STATE_AWARE_EXPERIMENT.md)。后者保留探索过程，历史假设
及“下一步”不等于最新结论；以本日期快照为当前状态。

本次存档不运行新实验，不修改采样算法、FFHQ分支、批量入口或默认参数，
不更新PPT。下一次从P0的单因素因果问题开始，不重做已有失败扫描。
