# A34：将测量噪声水平用于连续引导反馈

2026-10-21；A线；状态planned；用户授权继续设计并给出服务器命令。
目标：验证A33“测量残差越低、画质反而变差”的目标失配，是否可通过已有R/c/b
组成的噪声目标反馈改善；保持30个采样位置、ζ/D及方法300 NFE不变。

## 父证据、身份、允许结论

父 A33 /data0/lzy/ZAPS-results/diag_continuous_response_feedback/A33_20261021，
必须12case/72arm complete、SHA与门控通过；A32祖先的原起点、cache、参数、
measurement、checkpoint、runtime/native precision、unit H/legacy_bicubic transpose、
sigma_y=.05、GT图像及历史指标代码全部hash继承。
FFHQ 4×SR，00020–00025×seed1000/2000，六张重复开发图；不是独立确认。
A33三个动态组分别相对C的PSNR +1.23560/+.68439/−1.83895dB，
最终跨case RMS残差/噪声约1.0006/.4640/.0657；三个组均没有超过匹配时间表。
因此不继续扩大固定上界搜索，不把较强的阶段性收益说成状态增益。
原C/F/A32_S及A33 I_cont2/STAGE_cont2均复用，不重复UNet。

## 理论与新假设

原控制最小化 ||r−gamma*a||²，非负射线最优gamma_LS=max(0,c/b)。
新增R=||r||/(sqrt(m)*sigma_y)，目标delta=tau*sqrt(m)*sigma_y。
以“最少正向位移使代理残差进入噪声球”为原则：
- R≤tau：gamma=0，仅停当前测量引导，不停去噪。
- c≤0：gamma=0，方向不改善该代理。
- 目标可达：取方程 1−2gamma*b*c+gamma²*b²=(tau/R)² 的较小非负根。
  稳定形式 gamma=(1−(tau/R)²)/(b*(c+sqrt(c²−1+(tau/R)²)))。
- 目标不可达：取gamma_LS，达到这条射线可实现的最小代理残差。
- 无可解析响应：保留原gamma=1并标记unresolved；非有限状态显式失败，不按GT回退。

不设固定gamma上界；可解析下降方向上由状态决定范围0≤gamma≤gamma_LS。
这与Morozov discrepancy principle使用已知噪声尺度控制拟合程度的思想一致：
https://www2.imm.dtu.dk/~pcha/Regutools/RTv4manual.pdf ，§2.9。
对白噪声epsilon~N(0,sigma_y²I)，E||epsilon||²=m*sigma_y²，故R约1是物理参照。
它不是任意估计器残差必须精确等于1的定理；有效自由度、先验、算子误差、代理偏差
均可能影响最佳水平。本次只为discrepancy-inspired迁移，未证明ZAPS收敛、画质最优。
测量响应a仍用A33简化同t代理，不把它说成真实下一步Jacobian或轨迹稳定性/曲率。

## 因子、固定条件、预算

三组：N090/N100/N110，tau=.9/1/1.1；围绕噪声尺度的±10%模型失配敏感性试验，
不是高斯噪声的95%区间，不能后验挑一个就称理论最佳。
三组匹配TIME090/100/110：其他五张图的同策略真实轨迹gamma逐步取中位数，
排除当前图两个seed，全部输出冻结后生成，GT不参与。
新噪声球目标替代A33的最小残差目标及1%改善回退；不能保留会破坏噪声停止条件的
“改善不足1%改回gamma1”。除此之外响应近似、DDPM、网格、随机噪声、指标均不变。
只修改最后unroll，历史前270次不重新优化，不外推十轮全部自适应。
新增12×6×30=2160诊断NFE；部署每臂270+30=300，传感器额外UNet=0，H耗时另计。
复用不可变A33执行器（新进程显式注入A34策略对象），原脚本/核心文件不修改。
继承首中心≤1e−6、29次原噪声回放、参数/时间序列/precision/RNG/预算门控，
GT在输出保存后才评价。每arm独立恢复，失败attempt保留并记实际费用。

## 读数与可否定判定

全部方案同时报告C、I_cont2、STAGE_cont2、匹配TIME及F的终局PSNR/SSIM/LPIPS/loss，
每图两seed均值、胜率、实际倍率上下界、最终R、predicted_R_after与下一步实际R。
预测值仅是代理；不因predicted_R_after命中1就称下一步真实状态满足噪声约束。
开发候选需完整12case；PSNR相对C>0，相对STAGE_cont2及匹配TIME各≥.01dB，
至少4/6图簇优于匹配TIME；均值SSIM≥−.002、LPIPS≤+.005相对C及两个时间参照均通过。
这些是预先锁定的工程门槛，不是统计显著性或科学成功概率。
若只改善相对C但不胜匹配TIME，保留动作规律，不记状态增益；若R更合理但画质无改善，
否定“噪声目标即可解决当前反馈”假设，停止加密tau扫描。若候选通过，冻结后新图确认。
理论噪声尺度可解释并不自动证明新指标合理有效，更不代表旧方向几何已验证。

## 命令与产物

bash run_a34_20261021.sh
结果 /data0/lzy/ZAPS-results/diag_noise_target_response_feedback/A34_20261021。
双卡分两阶段运行、一次打印摘要、逐步CSV与真实统一ROI图归档到原结果分支。
Mac只做CPU逻辑/元数据检查，Torch/CUDA在服务器先检查再执行；未运行前不写结果。
