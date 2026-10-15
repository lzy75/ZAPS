# A0：固定成功参考上的单图状态轨迹验证

日期：2026-10-15（Asia/Shanghai）。状态：implemented，真实数值回放待服务器执行。
目标：检验开题报告的方向稳定性/物理一致性指标是否提供可解释的过程信息。
本阶段不训练新参数、不改变时间网格、不启用状态控制，不以PSNR提升为本实验成功条件。

## 问题、可否定假设及去重

已回传FFHQ 00000 SR unit-null为26.7013/.8202/.1608（Karras rho3、lr=.01、
经验legacy_bicubic映射）；最新公开设置四任务入口28de276尚无本地完整结果。
原状态映射在unit协议没有综合收益；这不等于两个指标没有信息。
已有indicator_analysis.py仅跨run聚合，不能回答完整去噪过程的变化。
已有optimized_trace针对ImageNet两网格，未完整分析本次FFHQ双指标。
本次新增信息：成功参考最后一轮的全部30点，clean与含噪方向对照、
噪声标度残差、离线逐步PSNR/SSIM/LPIPS及下一步变化关系。

H1：同一成功轨迹中低噪声段的残差/噪声标度中位数低于高噪声段；
允许局部上升，不要求逐步单调，也不把拟合观测噪声至0作为目标。
H2：clean余弦具有足够有效点和变化范围，并可能关联后续质量变化；
“余弦必然单调上升”不是成立前提。DDPM的clean估计依然受随机轨迹影响。
H3（后续待验证）：状态信号可以指导权重并改善质量；本被动单图实验不能证明H3。

## 父run选择与固定条件

在服务器读取原始run.json和.pt，不使用只有文本的GitHub归档重造随机状态。
预先固定FFHQ 00000/seed1000/SR×4/unit sigma=.05、模型[-1,1]。
自动选择顺序：优先已回传质量较好的状态域配对unit-null；否则已完成四任务中的第一张00000 SR。
各层按时间目录逆序，不按PSNR排序；failed/running/缺张量不复用。
显式`--source-dir`可冻结父目录，也可指定最新四任务源；`--inspect`打印准确父目录、原参数和网格（0 NFE）。
不自动运行旧实验来制造父run。若两种源都没有，停止并请求原始目录。

四任务源：paper15/10/5、lr=.001、exact伴随；
历史源：原Karras rho3、lr=.01、legacy_bicubic经验T及.5T，明确标development。
两种协议不混合、不按分数互换物理协议；结果只相对实际选中的父run解读。
两种源均：30步×10轮历史优化、joint ζ/D、db4 level3、DDPM fixed-small eta1、
score冻结、原clamp/损失/梯度裁剪/投影，读取PRE第10轮参数（不是post_params）。
复用原始y、x_T和最后一轮开始的CPU/实际采样CUDA生成器状态；同一网格和索引。
完整配置、source run绝对路径/hash、源码/图片/权重/y/参数/输出hash、git/runtime保存。
父run绝对路径在--inspect和新run.json中解析，运行前可固定，不杜撰未知目录。
没有独立验证集；这是已探索样本的机制诊断，后续验证不能复用其作为未见样本。

## 指标定义和时间语义

z_k=实际clamp后的Tweedie估计，d_k=z_k-z_(k-1)。
clean_cos=cos(d_k,d_(k-1))（前两个点NA）；latent_cos使用原含噪更新增量
x_next-x_input（第一个点NA，采样后才可知，在线使用需滞后一拍）。
范数<=1e-8的方向为NA，不补0或1。
r=||y-A(z_k)||，r_rms=r/sqrt(m)，r_noise=r/(.05 sqrt(m))。
sigma=.05是物理观测噪声，不能与模型的sigma_eff混淆。
u=(log sigma_eff(999)-log sigma_eff(t))/(log sigma_eff(999)-log sigma_eff(0))。
v_k=d_k/(u_k-u_(k-1))；turn=||v_k-v_(k-1)||/(||v_k||+||v_(k-1)||)。
正标量跨度归一化不改变余弦，只影响速度/turn；不能宣称由此修复余弦本身。
turn是轨迹代理，不是已证明的截断误差。所有状态在修正前clean上定义；
最后一步修正后的实际输出单独报告三指标及物理残差，不能与t=0 clean混用。
不加入未经校准的代价函数，不改ζ，不选最佳epoch/最佳中间图。

## 预算、身份和等价门控

新方法NFE=0，历史父方法NFE=300（不重复运行）；本次3×30=90诊断NFE，
其中带观察器回放一次、原核心重复两次。LPIPS为独立评价网络，不属于扩散NFE，
逐点与最终共31次三指标评估；总耗时包含拷贝/状态计算/指标评估。
观察器调用真实Tweedie方法并原样返回结果，不复制采样公式、不改变图或随机数。
观察器/重复轨迹的输出、30点时间/参数索引、score调用数及RNG末状态一致。
输出参数不变；原始输入/输出/参数文件hash、源码和权重hash必须匹配。
重复校准门控：repeat<=1e-5；cross<=min(1e-5,2*repeat+1e-7)，同时比较父输出。
三指标漂移限PSNR .01dB/SSIM .001/LPIPS .001，不事后放宽。
门控失败保存failed与已用NFE，不解读指标，也不自动重跑直到通过。
本机无PyTorch，契约/统计测试与CPU核心数值测试状态分别报告；服务器不应有skipped。

## 预期及结果分支

| 类别 | 预注册读数 | 解读及下一步 |
| --- | --- | --- |
| H1支持 | 低噪声t<334的r_noise中位数小于高噪声t>=667 | 只支持该轨迹宏观物理一致性趋势；看局部质量背离，不直接启动控制 |
| H2可读候选 | clean有效>=24/28，span>=.05；描述其与下一步ΔPSNR/ΔLPIPS/Δr的关系 | 只是信号有动态范围；需检查时间混杂，不能据此认证控制有效 |
| 负向 | H1不符，或余弦近常数/大量无效，或r下降而LPIPS恶化 | 否定该简单判读；只改一项表达或权重响应，先不改调度 |
| 不确定 | 关联弱/方向相互矛盾/只见全程趋势 | 保留基线；事先固定少量其他图或seed验证，不扫描直到显著 |
| 门控失败 | hash/输出/RNG/指标口径不兼容 | 停止解释，修具体一致性问题 |

报告Pearson/Spearman和去线性u趋势的偏相关，不作独立样本p值检验；
30点有自相关，去线性趋势不能去掉全部非线性混杂。所有方向的正负均完整输出。
相关性不是因果控制证据。若有潜力，下一步只做固定网格下ζ乘子的小幅有界
单因素配对，再独立小样本验证；不引入调度/步数变量。
不能预先承诺质量提高或收敛。开题方案可作为研究假设，不作为已实现创新结论。

## 运行与归档

服务器：

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_diag_state_trajectory -v
python -u utils/diag_state_trajectory.py --inspect
set -o pipefail
python -u utils/diag_state_trajectory.py --device cuda | tee ../state_trajectory_20261015.log
```

可加`--source-dir /home/lzy/ZAPS/projects/results/...`冻结inspect显示的父目录。
若无matplotlib，CSV/张量/汇总照常保存；装好后`--plot-only <新Records目录>`，不再调用UNet。
新结果写`projects/results/diag_state_trajectory/<北京时间戳>/`，不覆盖父结果。
产物：run.json、trajectory.csv、trajectory.pt、选定进度clean图、实际final图、
trajectory.png/svg（若绘图库可用）、实验卡。回贴完整30点、Gates、Final metrics、
Summary、Records。不自动扩展图数；一个阶段结束后统一上传文本和曲线，权重不入库。
