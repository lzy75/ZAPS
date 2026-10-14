# A-ImageNet-tuned-state：首步初始化调整后的固定30步状态实验

日期2026-10-14，用户同意继续。独立A线，不是原文复现认证。
本聊天只保留复现/状态主体；初始化调度文献及搜索由另一个聊天推进。

## 目标与证据

新完成first_init_joint_training：PSNR21.3650、SSIM.4417、LPIPS.5092；
相对原19.5987提升1.7663dB，冻结幅度20.7478提升.6172dB，三指标共同改善。
支持首步ζ初始化敏感，但不能认定唯一根因或推广到FFHQ/所有图像/任务。
首步初始化R线到此收束。原始manifest和19.5987参考不覆盖。

父：指定trace内兼容completed first_init_*/run.json；支持显式--init-run-dir。
锁定parent JSON、output、快照、权重、H/y、原图、核心及supporting源码hash。
当前完整父JSON尚未本机下载，入口须在服务器验真后运行。

可否定假设：在首步ζ=.0400298317787、其余ζ=.2、D=.2的同一基座上，
物理残差主导＋clean轨迹余弦置信门控，能够提供超出固定网格null的综合收益。

## 两臂与唯一方法变量

null/state均同图00001、seed1001、Gaussian61/sigma3/zero padding、原观测，
30×10、lr=.001、ζ/D联合、db4 level3、fixed-small eta1、原score近似和clamp。
仅response从0变.2；不降低噪声、不改学习率/初值/损失/采样器，不改引导权重。
首轮作为预算内pilot并严格走paper15/10/5；第2–10轮在线调整。
ζ/D仍按反向调用位置29..0使用，未引入noise-anchor参数插值。
这只是固定S微调方案，不能说已验证可变S参数语义。

## 实际状态公式（本项目设计，不冒称原论文公式）

读当前alpha_bar，sigma(t)=sqrt((1-alpha_bar)/alpha_bar)，
u=(log sigma_max-log sigma(t))/(log sigma_max-log sigma_min)，完整999→0。
pilot储存(log r_ref(u), c_ref(u))，r=||y-H(clamp(xhat0))||，
c=cos(Δxhat0_k, Δxhat0_(k-1))，global clean cosine不混高频新特征。
注意：clamp后的clean估计仍受之前随机轨迹影响，不称无噪真曲率。
pilot插值为u上的分段线性（残差在log域），不再按相同步序号假装同噪声。
cos的首两个参考无定义时保持中性，不跨无效端点补造曲率。

```
l = log r - interp_u(log r_ref)
R = tanh((l - EMA_previous)/.15)
EMA decay = .8^(Δu * 29)       # 首项建立中心，不立即响应
C = tanh((interp_u(c_ref)-c)/.2)
q = 1 - .25*max(0, -sign(R)*C)
h = remaining_scaled_nominal_h * clip(1 - lambda*R*q, .8, 1.2)
lambda_null=0; lambda_state=.2
```

余弦只能减弱冲突响应，不能反转物理方向；不是直接.8/.2线性相加。
不把正步长归一化余弦称新信息（正标量缩放不会改变余弦）。
步长经整数取整/剩余调用可达性限制，末次模型必须t0，总30次；无拒绝重算。
两臂source seed生成同x_T与按epoch/调用位置的29个正态draw；不同网格的
实际方差不同，这是CRN配对，不是同一物理时间Brownian轨迹。

## 门控、成本与输出

0新增UNet：身份检查、原训练噪声前缀和尾RNG预检。
90诊断NFE：初始参数fixed-A、fixed-B、adaptive-null各30步，无优化，
比较输出/ζ/D VJP、参数索引/网格/NFE、末端RNG；fixed-repeat估数值底噪。
输出hard cap1e-6，gradient相对误差hard cap1e-4；cross限为3×repeat，
下限output1e-7、gradient1e-6，上限hard cap。超cap直接停止，不放宽追分。
这个门控仅证明初始一轮，不冒称完整10轮逐元素一致。

新方法NFE=2×300=600，总新UNet上限690；pilot包括在每臂300内。
fresh null是新adaptive入口/坐标下的必要配对，而非因原分数低原样重跑。
GT直到两臂训练完成才加载，只评价last_opt，非最佳epoch/early stop。
后更新参数另存并标明不能生成last_opt；不额外采样。
保存每轮真实grid、最后状态轨迹、loss/指标/残差、RNG/源身份、actual NFE。
初始化参考21.3650及uniform21.5069分开记录；后者是不同初值的历史强固定参照。

## 判据和处理（预注册工作阈值，非统计显著性）

- null对父漂移需abs dPSNR≤.05、abs dSSIM≤.005、abs dLPIPS≤.005；
  超过则标INCONCLUSIVE_NULL_DRIFT，先解释漂移，不追分或宣称状态有效。
- 对同次null，dPSNR≥.10且dSSIM≥-.002、dLPIPS≤+.002：单图综合候选；
  之后讨论固定配置的小规模独立验证，单图不证明统计收益。
- 只有PSNR涨而感知超容忍界：PSNR_ONLY_TRADEOFF，不算综合成功。
- 无清楚综合收益：固定初值不动，下一步只讨论残差/余弦表达或组合的一项，
  不回到lr/倍率/降噪扫描、不取消任何一个核心状态指标。
- gate失败：只有接口/一致性诊断，没有算法结果；保留failed记录。

状态收益必须相对本次null；与uniform还要单独比较，不能把初始化的1.7663dB
记入状态创新增益。自适应/自动步数、FFHQ/batch同步均不在本次范围。
