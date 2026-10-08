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

## 幅度结果：固定 rho=0.75 验证优化/采样一致性

保持引导的冻结参数回放：不规则 rho=0.5/0.75 的 PSNR 为
23.9557/24.2687、LPIPS 为 0.5544/0.4985；其 rho=1 基线为
19.5987/0.6505。rho=0.75 同时改善 PSNR +4.6700、SSIM +0.3585、
LPIPS -0.1519。不再扩大幅度搜索。

均匀 rho=0.75 为 23.5971/0.6494/0.4560（PSNR/SSIM/LPIPS），相对
rho=1 仍是像素指标改善但 LPIPS 退化。不规则在共同 rho=0.75 下 PSNR
高 0.6716、SSIM 高 0.0259，但 LPIPS 差 0.0425；不能宣布全面复现论文
的不规则优势，更不能用本图 24.2687 对比论文样本均值宣布成功。

下一步 `diag_zaps_late_noise_training.py` 对每个网格做 train rho x eval
rho=1/0.75 的 2x2 对照。train=1 复用保存的第10轮参数和噪声，不重复
训练；train=0.75 执行原 optimize、lr=0.001、30步x10轮、联合学习 zeta/D，
仅在 t<=333 使用 DDPM mean+(eta*rho)*sigma*z+原 correction。两个
阶段都通过同一核心 helper 应用该设置；不改核心文件、配置默认值或
FFHQ/批量分支。helper 仅在诊断进程的调用作用域内暂时替换，异常也恢复。

回放仍取第10轮采样前参数，而非最后一次 Adam 更新后的参数；匹配
train/eval 政策的输出与原 last_opt 门控，核对原基线 PSNR、新训练 x_T、
第10轮 RNG 起点及所有评估 RNG 终点。每个网格只新增一次训练，合计
600 NFE；2x2 回放和重复门控合计 360 NFE，本次总计 960 NFE。

部分噪声采用核心 helper 的 eta*rho 直接乘标准差，保持可微计算图。
这与早先定位时 mean+rho*(带噪结果-mean) 数学等价，但浮点运算顺序
不同，因此不要求两种实现的 rho<1 轨迹逐位一致；rho=1 基线和训练
last_opt 必须分别通过当前同操作路径门控。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_late_noise_training.py \
  --trace-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857 \
  --late-start 333 \
  --noise-scale 0.75 \
  --device cuda \
  | tee ../imagenet_gaussian_late_noise_training.log
```

以 train=eval=0.75 对比 train=eval=1 判断统一设置下是否改善；
train=1/eval=0.75 只表示后处理式回放改变，不能当作一致训练基线。
两组交叉回放用于区分参数适配与采样设置效果。结果首先报告所有三个
质量指标，不只 PSNR。rho<1 是修改版固定设置，不等于恢复原文、修正
了已确认的公式 bug，也不是状态指标自身带来的创新增益。需后续验证
随机种子、少量固定图及严格原文实现约定后，再决定是否采用该候选设置。

## 一致训练结果与同图换种子

原种子一致训练结果：

| 调度 | train/eval rho | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| 不规则 | 1/1 | 19.5987 | 0.3168 | 0.6505 |
| 不规则 | 0.75/0.75 | 24.2327 | 0.6743 | 0.4914 |
| 均匀 | 1/1 | 21.5069 | 0.5510 | 0.3432 |
| 均匀 | 0.75/0.75 | 23.4764 | 0.6480 | 0.4471 |

不规则一致设置相对自身原基线 PSNR +4.6340、LPIPS -0.1590；均匀
PSNR +1.9695 但 LPIPS +0.1039。新训练参数在 eval=0.75 时相对旧
训练参数的 PSNR 变化仅为 -0.0361/-0.1207，LPIPS 略改善；换回 eval=1
后不规则回到 19.5173，均匀为 20.4530。支持本例的主导收益来自晚期
随机项幅度改变，而非 zeta/D 重新适配。不是所有图像/任务的普遍结论。

不继续调学习率、幅度或时间步。先固定同图、同 measurement.pt，仅把
优化/采样种子由 1001 改为 1000。新增 --seed 时，脚本在新种子下重新
优化 rho=1 与 0.75 两组，用新基线而非原种子的数值计算 delta。
两种调度各两次优化，共 1200 NFE，加交叉回放与门控 360 NFE，总计
1560 NFE。原观察噪声不会重新生成；x_T 与后验噪声在同种子的对照内
配对，而不是强求与不同种子的旧轨迹一致。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_late_noise_training.py \
  --trace-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857 \
  --late-start 333 \
  --noise-scale 0.75 \
  --seed 1000 \
  --device cuda \
  | tee ../imagenet_gaussian_late_noise_seed1000.log
```

每个种子均以 train=eval=0.75 对比本种子 train=eval=1，记录三种质量
指标；不同种子之间先比较配对收益，不直接混用绝对 PSNR。若同图新
种子的改善方向仍成立，再验证第二张图和另一个正式任务；若改善消失
或反转，报告轨迹依赖，不把本例当成稳定基线。即使两次方向一致，也
只支持局部稳定性，不是统计上的泛化证明或原文复现成功。FFHQ/批量
仍不改默认设置。

## 2026-10-08：结束晚期降噪探索，检查原始采样实现

同图、同观测的新种子1000结果：不规则 train/eval=1/1 为
19.0669/0.3037/0.6076（PSNR/SSIM/LPIPS），0.75/0.75 为
23.9203/0.6549/0.4797；均匀对应为21.2842/0.4463/0.5184 与
23.6131/0.6388/0.4378。主导改善仍来自评估时的随机项幅度改变。

这只定位到轨迹敏感性，没有证明代码错误、ImageNet 数据问题或原文
复现成功。停止扩大降噪/学习率/调度搜索，不再执行上一节建议的更多
降噪换图试验；保留 eta=1 的原始基线继续排查。rho<1 只归档为诊断，
不作为状态感知增益或同步到 FFHQ/批量默认配置。

新增 `utils/diag_zaps_sampler_parity.py`：复用原种子的保存参数、x_T 与
第10轮 RNG，不再优化。直接导入仓库 DPS 的原始 GaussianDiffusion、
SpacedDiffusion、DDPM 和 mean/variance processors，使用私有包名避免
与 UNet 的 guided_diffusion 包冲突；真实 utility imports 不作数值替换。
记录参考源文件 SHA256，缺失/混用来源则停止，而非悄悄换参考公式。

参考保持**源实验的方差策略**：本例 fixed_small 对 fixed_small；
不能直接与 DPS 默认 learned_range 比较而把预期差异判成 bug。
模型预测复用，每次仍仅一次 UNet 调用。检查每一步的原始时间步映射、
裁剪后的 Tweedie x0、共享同一 x0 时的后验均值、完整后验均值、有效
注噪标准差以及同一噪声下的下一步输出。末步标准差应为零，不比较其
未实际使用的 learned log variance。std 通过原核心 helper 的零输入/
单位噪声探针取得，next 通过 DPS 实际 p_sample 取得，不重写采样公式。

每个网格两次原核心回放和一次被动审计，30x3=90 NFE；两个网格合计
180 NFE，无优化、无 GT/PSNR 选优、无方差策略消融。诊断过程须通过
输出、NFE 和 RNG 终点门控。逐元素容差固定 atol=rtol=2e-5，同时记录
相对范数与最大绝对误差；两种系数计算的 float32/float64 顺序差异不可
强求逐位一致。门控不稳定或误差超限则记录 FAIL，不据此直接宣布 bug。

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_sampler_parity.py \
  --trace-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857 \
  --device cuda \
  | tee ../imagenet_gaussian_sampler_parity.log
```

PASS 仅排除这些已保存轨迹上的去噪/前向 DDPM 运算不一致，不验证引导、
反向传播或论文整体设置。FAIL 先看首个失败分量与 replay/RNG 门控；
不能调容差掩盖差异。之后才决定修正具体实现，或进入同观测官方 DPS/
ZAPS 特有引导与优化流程的排查。本地无 PyTorch：语法/CLI/导入隔离
测试通过，三个 PyTorch 数值测试显式跳过；服务器结果尚待用户运行。
