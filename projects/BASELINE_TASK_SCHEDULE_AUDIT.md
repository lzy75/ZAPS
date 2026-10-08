# ZAPS 基线与时间步趋势核对

## 目的与已有证据

论文 ImageNet 定量表报告 Gaussian Deblur 和 4x SR，ZAPS 分别为
22.45/23.82 dB、SSIM 0.682/0.718、LPIPS 0.225/0.186。
这不是 ImageNet 随机修复的参照。已有 ImageNet SR 10 张图均值为
19.3007 dB、SSIM 0.4350、LPIPS 0.4732；图像集合不同，不能当成
同样本的严格差值，但这组结果尚不足以宣布论文基线复现成功。

论文 FFHQ SR 的 300 NFE 调度对照为：

| 调度 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Uniform | 26.29 | 0.729 | 0.119 |
| Irregular | 26.63 | 0.768 | 0.104 |

本地已报告的 10 张 FFHQ SR 对照为：

| 调度 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Uniform | 28.3043 | 0.8438 | 0.1303 |
| Irregular | 28.1905 | 0.8360 | 0.1265 |

本地对照使用 lr=0.01、legacy_bicubic 映射、固定 skip 方差和最后一轮
优化输出。均匀 PSNR 高 0.1138 dB，但 LPIPS 差 0.0038。这个排序没有
复现论文的 PSNR 趋势，不能解释成“所有指标下均匀更好”。本地子集与
论文 1000 张均值不同，绝对值更高也不能说明已经超过论文。

## 第一阶段：ImageNet 单图两个任务

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_paper_task_schedule.py \
  --dataset imagenet \
  --image /home/lzy/imagenet/256x256/00001.png \
  --tasks gaussian_deblur super_resolution \
  --learning-rates 0.001 \
  --sr-transpose exact \
  --device cuda \
  --seed 1001 \
  | tee ../imagenet_two_task_schedule_audit.log
```

各任务仅比较 15/10/5 不规则调度与全局均匀 30 点调度。固定每轮 30 次
模型调用、10 轮、300 NFE，共同学习 zeta 和 D，固定 DDPM skip 方差。
同一任务下 y、x_T 和按步位置生成的 DDPM 随机数相同。高斯任务 zeta
初值为 0.2，SR 为 0.1，D 初值都是 0.2。sigma=0.05 保持当前 [-1,1]
域约定。这是现有实现的基线审计，不宣称所有实现细节已经与原文一致。
不运行状态调度，不改变核心算法，不依据单图数值选择论文基线。

## 第二阶段：核对 FFHQ 学习率与调度交互

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_paper_task_schedule.py \
  --dataset ffhq \
  --image /home/lzy/FFHQ/00000/00000/00000.png \
  --tasks super_resolution \
  --learning-rates 0.001 0.01 \
  --sr-transpose legacy_bicubic \
  --device cuda \
  --seed 1000 \
  | tee ../ffhq_lr_schedule_audit.log
```

保留先前 FFHQ 调度比较的算子设置，只做学习率 x 调度 2x2 配对。若两个
学习率下排序不同，支持“调参改变调度排序”的解释；若均匀在两个学习率
下都更好，不能把排序差异归因于学习率，下一步需核对前向算子、引导映射、
输出策略和图像子集。legacy_bicubic 是历史引导映射，不称为精确 H^T。

## 输出和判读

脚本每完成一条轨迹就保存 run.json、metrics.csv、PNG，且保存每个任务
实际使用的 measurement.pt。run.json 包含提交号、完整 ZAPS 参数、实际
时间步、各轮 loss、最终 zeta、模型路径和任务参数。PSNR/SSIM 沿用已有
uint8 评估方式，同时记录未量化的 float_psnr，避免日志 PSNR 与最终指标
的细小差异被误判为算法变化。

最终表的差值统一为 Irregular - Uniform：PSNR/SSIM 正值、LPIPS 负值
支持不规则调度。单图用于检查流程与方向；只有在参数和算子约定确定后，
才用固定的 10--20 张图验证平均趋势。这里没有 ImageNet 不规则一定胜过
均匀的预设，因为论文展示的直接调度消融是在 FFHQ 上。

当前 ImageNet 随机修复的 10/15 步增益仅作为探索记录，不能替代这两个
正式任务的复现审计，也不能证明 ImageNet 论文基线已经复现。

## 初始轨迹审计：先过诊断一致性门控

`utils/diag_zaps_trace_audit.py` 复用上述 run.json 和 measurement.pt，仅检查
初始化时的固定采样轨迹，不重新优化，也不修改核心算法、学习率或权重。
每个调度运行诊断路径一次、核心路径两次，三次重置为同一份 CPU/CUDA RNG
状态并共用 x_T。30 步时每个调度的审计成本是 90 NFE，不是训练的 300 NFE。

诊断路径的引导修正必须与核心代码逐操作一致：先除以 sqrt(alpha_bar)，
再乘 zeta；不能以浮点下未必相等的代数重排替代。原诊断曾先乘再除，本次
服务器运行的最终相对误差约 1.78e-4；修正后仍需复测，才能确认重排解释
了多少误差。这个报错不能当成 ZAPS 基线算法错误或已确认的 GPU 非确定性
证据。

新版门控要求实际时间步及 NFE 相同，且诊断/核心相对误差不超过
min(1e-5, 2 * 核心重复相对误差 + 1e-7)。核心自身重复误差也不得超过
1e-5；保留原有硬上限，并用实测重复误差检查是否存在额外路径差异。
门控失败仍保存 *_parity.json、逐步 CSV 和 trace.json，先看具体分类，
不解读未通过一致性检查的诊断轨迹为重建机制证据。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_trace_audit.py \
  --run-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549 \
  --task gaussian_deblur --device cuda \
  | tee ../imagenet_gaussian_initial_trace.log
```

## 第10轮优化轨迹：定位后半段退化

初始化审计已在服务器上通过：两种调度的诊断/核心误差、核心重复误差
均为 0。不规则初始化轨迹在 t=333 的 x0 估计为 21.71 dB，t=0 为
19.23 dB；均匀在 t=344 为 21.28 dB，t=0 为 20.03 dB。不能仅用
初始化轨迹解释优化后的 19.5987/21.5069 dB 差距。

下一项只复用同一图、measurement.pt、seed、保存配置和时间步重新执行
两种调度的原始优化（各 30 步 x 10 轮，lr=0.001，zeta+D 都学习）。
`diag_zaps_optimized_trace.py` 通过诊断子类在最后一轮采样调用前被动保存
zeta、完整 D、x_T 和 CPU/CUDA RNG，仍调用原始 optimize/reverse 方法。
优化结束后恢复这些采样前参数，不使用第10次 Adam 更新后的参数，以
免混淆 last_opt 和新采样。诊断/核心、last_opt/核心回放分别门控；失败
保留记录并停止解读。

每个调度优化 300 NFE，诊断与两次核心回放额外 90 NFE，总成本两个调度
为 780 NFE。逐步 CSV、PNG、audit.json 和最后一轮回放状态文件保存至新
目录，不覆盖原归档。末尾直接输出并排汇总，包括最终 PSNR/SSIM/LPIPS、
t<=400 的最高 x0 PSNR、峰值时间步、t=0 x0 PSNR、两者之差和最终 MSE。
中间最高 PSNR 只用于诊断，不用 GT 早停或挑选输出。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_optimized_trace.py \
  --run-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549 \
  --task gaussian_deblur \
  --learning-rate 0.001 \
  --device cuda \
  | tee ../imagenet_gaussian_optimized_trace.log
```

若优化后不规则仍出现更大的后半段 x0 PSNR 下降，下一步才做同轨迹的
低噪声引导/随机转移单因素消融，以区分原因；不把中间峰值当作可实现
的最终重建基线。若没有复现退化趋势，则放弃以初始化晚期退化解释优化
差距，检查完整的优化轨迹及随机波动，不继续基于该假设调参。

## 已确认优化后低噪声退化：后半段噪声 x 引导分解

第10轮复测精确复现最终结果：不规则 19.5987 dB、均匀 21.5069 dB。
t<=400 的最高 x0 PSNR 分别 21.7661/22.0823 dB，t=0 为
19.5790/21.4644 dB，后半段下降 2.1872/0.6180 dB。不规则的峰值低
0.3162 dB，但到 t=0 差距扩大到 1.8854 dB，支持优先排查后半段。
这还不证明哪个公式写错，也不证明论文调度在所有 ImageNet 图上应获胜。

复用已保存的 *_last_unroll_state.pt，不重新优化，运行四种冻结参数回放：
baseline、late_noise_off、late_guidance_off、late_both_off。干预边界提前固定
为 t<=333（低噪声三分之一区域），不根据 GT 峰值选择。每个调度的前段、
zeta/D、x_T、时间步和后验噪声抽样序列不变。关闭随机噪声仍先执行原来的
噪声抽样，然后用相同 DDPM 后验均值替代带噪更新，不改变后续 RNG 序列；
关闭引导仅将该段显式 correction 置零，不冻结或改变 D 的历史训练。

baseline 先与两次核心回放门控，并核对保存的 PSNR；各干预核对 RNG 最终
状态与 baseline 相同。每个调度四条轨迹 120 NFE，门控额外 60 NFE，两种
调度总计 360 NFE，不加训练。新记录保存在 optimized_trace 目录的子目录。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_late_component.py \
  --trace-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857 \
  --late-start 333 \
  --device cuda \
  | tee ../imagenet_gaussian_late_component.log
```

判断每种调度相对自身 baseline 的变化，不把跨调度差值与干预收益混淆。
noise_off 改善支持随机转移敏感性；guidance_off 改善支持晚期引导敏感性；
仅 both_off 改善或单项效果依赖另一项，说明存在交互。均不改善则下一步
审查确定性后验/Tweedie 路径及优化目标。关闭分量是定位实验，不是最终
算法、不是论文基线，更不能仅凭该结果宣称代码有错。

## 后半段分量结果与幅度验证

固定训练参数、t<=333 的单图回放结果：

| 调度 | baseline PSNR/LPIPS | noise_off PSNR/LPIPS | guidance_off PSNR | both_off PSNR |
| --- | --- | --- | ---: | ---: |
| 不规则 | 19.5987 / 0.6505 | 23.5527 / 0.6884 | 18.8771 | 22.5121 |
| 均匀 | 21.5069 / 0.3432 | 23.6947 / 0.6575 | 20.5280 | 22.7676 |

关闭晚期噪声使 PSNR 分别 +3.9540/+2.1878，跨调度差距从 1.9082 缩小
至 0.1420 dB；关闭引导反而降低 PSNR。说明这条已训练轨迹的后半段
退化主要对随机项敏感，引导仍有正向作用。但 noise_off 的 LPIPS 分别
恶化 0.0379/0.3143，不能当作全面改善，也不能凭本图宣布 ImageNet
论文基线复现成功。

核对当前固定 DDPM 后验公式：令跳步有效 beta=1-alpha_bar_t/alpha_bar_s，
仓库 DPS SpacedDiffusion 的重新构建 beta 与 ZAPS 的 c1、c2、beta_tilde
代数一致，标量双精度两种网格检查最大误差 1.11e-16；代码使用 sqrt(beta_tilde)
乘随机项。这个检查不覆盖论文采样约定、GPU模型回放或 learned variance，
不能把轨迹敏感性直接归因为已确认的方差公式 bug。

下一步保留引导，固定边界和已训练参数，只比较噪声幅度 rho=0/0.5/0.75/1：
晚期更新为原 DDPM 均值 + rho * 原随机项 + 原 correction。rho 乘标准差，
有效方差乘 rho^2（rho=0.5 不是方差减半）。这是剂量验证，不扩展学习率、
时间步或阈值搜索。仍消耗原噪声抽样序列，baseline 门控不变。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_late_component.py \
  --trace-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857 \
  --late-start 333 \
  --noise-scales 0 0.5 0.75 1 \
  --device cuda \
  | tee ../imagenet_gaussian_late_noise_scale.log
```

关注是否存在 PSNR/SSIM 改善且 LPIPS 不明显退化的中间幅度，不能只挑
最高 PSNR。若出现稳定折中，再将优化/采样的约定同步做对照验证，不能
把训练 eta=1、冻结后采样降噪的定位结果作为新基线或状态指标收益。
当前仅扩展诊断入参，没有修改全局设置、核心采样、FFHQ或批量分支。
