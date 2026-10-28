# 当前方法与先行工作的关系、收益归因及质量—时间消融

日期：2026-10-28。按用户要求分析既有证据；实验暂停，没有运行新模型、修改历史判定或发布A63。

**结论：当前结果支持“物理加权与长度规则 + 当前去噪器响应方向”这一方法组合在有限样本上有效。
尚不支持“逐图实时幅度反馈优于时间规则”，也没有完成轨迹曲率稳定性、步长/权重联合控制或总体加速证明。
当前真实Jacobian方向与DPS、ΠGDM、JAPS、CLAMP存在明确先行关系，不能把它本身作为新颖性。**

## 1. 与CLAMP和ZAPS的实际关系

用户指定的CLAMP已有ICML 2026正式记录；作者预印本v1发布于2026-05-27。本次核对正式出版元数据及
全文§3.5—3.9，未运行作者代码、未取得同协议质量或时间对照。下列是公式级比较，不是性能比较。

|比较项|当前U|CLAMP|
|---|---|---|
|共性|观测信息通过当前去噪器输入Jacobian转成噪声状态方向|同样使用去噪器pullback|
|逆作用所在空间|观测空间：S=σ_y²I+cLLᵀ；S不含去噪器Jacobian|噪声状态空间：带单边曲率与方向阻尼的线性系统|
|方向修正|u=J_mᵀLᵀS⁻¹r，之后只做标量范数匹配|曲率系统求解会进一步改变梯度方向|
|稳定机制|当前没有曲率/阻尼闭环；数值准入及范数匹配|单边Gauss–Newton/Fisher近似与沿去噪残差的秩一阻尼|
|采样骨架|锁定本地ZAPS原30点DDPM及配对噪声|作者采用VP Langevin传播|

CLAMP的核心可写作（用该文符号，残差符号与本文相反）

\[
(B_t+\alpha_t H_t)\Delta_t=-c_t^{C},\quad
B_t z=\sigma_n^{-2}J_F^TJ_A^TJ_Az,\quad
c_t^{C}=\sigma_n^{-2}J_F^TJ_A^Tr_t^{C}.
\]

\[
H_t=\lambda_{\rm id}I+n_tn_t^T,\quad
n_t=(x_t-F(x_t))/\|x_t-F(x_t)\|,\quad
\alpha_t={\|x_t-F(x_t)\|\over\|J_F^TJ_A^Tr_t^{C}\|\sigma_{\rm prior}^2}.
\]

其单边曲率省略右侧J_F，通常非对称，使用GMRES；每次迭代会用网络输入VJP。
这与当前只在物理S中求逆、每步一次输入VJP不同。它的局部曲率也不能与随机相邻轨迹余弦混称。
该文的λ_id、求解器数据尺度σ_n和固定GMRES预算仍需指定；σ_n不必等于观测实际噪声，不能称为无超参数。
依据：[ICML/PMLR正式记录](https://proceedings.mlr.press/v306/shin26b.html)、
[作者全文§3.5—3.9](https://arxiv.org/html/2605.27990)。

ZAPS原方法的核心则是逐观测优化每步ζ与小波对角D，以近似先验Hessian/去噪响应，预训练网络权重固定。
本地原q近似为

\[
q_t={\zeta_{i(t)}\over a_t}[I+b_t^2 W D_{i(t)}W^T]L^T[y-\mathcal A(\operatorname{clip}m_t)].
\]

当前仍保留检查点、物理模型、原时间网格、DDPM传播、配对噪声、本地原基线与缓存优化前缀。
ζ/D仍存在于原C、Q方向及失败回退中，**但正常resolved U的目标公式不再使用学习后的ζ/D来决定方向或剂量**。
因此当前已从“保留原ZAPS自适应参数，再加状态权重”转向“在该实现骨架中替换引导向量”。
不能只凭代码来自ZAPS就把新U称为完整保留ZAPS自适应机制。

A57独立从seed生成初始化并配对末轮噪声，在5张旧FFHQ×SR/IP的10例中，30前向+30输入VJP、0优化更新，
与缓存U最大相对误差2.30e-7；该证据说明这些正常U输出可不依赖原学习前缀。
不把它扩写为新图/所有数据集/blur的完整无学习部署实验。A56/A62已消耗的历史前缀不扣回。
依据：[ZAPS原文§3与Algorithm1](https://arxiv.org/html/2407.11288v1)、本地A57摘要和源码。

更直接的先行工作是JAPS。其§3.1构造耦合的物理v与直接u，§3.3使用

\[
g_{\rm JAPS}^*= {\langle v,u\rangle\over\|u\|^2}u
=\cos(v,u){\|v\|\over\|u\|}u.
\]

当前U去掉这一余弦衰减，使用范数匹配方向。两者传播系数和实现协议仍需区分，不能据此称已经完整复现JAPS。
“归一化pullback”是明确可对照的方法变体；目前不足以构成独立统一状态控制体系。
依据：[JAPS v3，2026-06-21，§3.1—3.4](https://arxiv.org/html/2511.18471v3)。

## 2. 当前算法究竟计算什么

令abar_t为累计扩散系数，a_t=√abar_t，b_t²=1-abar_t。

\[
m_t(x_t)=(x_t-b_t\epsilon_\theta(x_t,t))/a_t,\qquad
\mathcal A(m)=H((m+1)/2)=Lm+d,\quad L=H/2.
\]

\[
r_t=y-\mathcal A(m_t),\quad c_t=b_t^2/a_t,\quad
S_t=\sigma_y^2I+c_tLL^T,\quad v_t=L^TS_t^{-1}r_t,\quad u_t=J_{m_t}(x_t)^Tv_t.
\]

\[
g_t=\|v_t\|/\|u_t\|,\quad K_{t,s}={1-\bar\alpha_t/\bar\alpha_s\over\sqrt{\bar\alpha_t/\bar\alpha_s}},
\quad T_t=K_{t,s}g_tu_t,
\quad x_s=\operatorname{DDPM}_{t\to s}(x_t,\operatorname{clip}m_t,\xi_t)+T_t.
\]

c_tI是时间相关各向同性代理，**不是本图估计的真实后验协方差**。
u不是预测噪声ε的方向；它回答“当前噪声图怎样改变，才能通过去噪器影响观测残差”。
当前是把原加性q替换成T，不是在原q之外再叠加一个同样的引导。

固定t和S时，ell_t(x)=.5r_t(x)ᵀS_t⁻¹r_t(x)满足u_t=-∇ell_t。
因此正的小位移沿u局部下降这一代理；范数等式为||T||=K||v||。
这只解释局部方向和长度，不保证有限DDPM更新后代理下降，不保证最终PSNR/LPIPS，更不是采样收敛证明。

```mermaid
flowchart LR
  X[当前 x 和固定时间 t] --> M[去噪预测 m]
  M --> R[观测残差 r]
  R --> V[观测空间加权 v]
  M --> J[当前去噪器输入 VJP]
  V --> J
  J --> U[响应方向 u]
  V --> G[范数匹配 g]
  U --> G
  G --> T[位移 K g u]
  T --> N[原 DDPM 转移加位移]
```

这里没有从状态选择h/采样位置，也没有用整段轨迹曲率决定gamma；外层gamma为1。
物理残差、Jacobian、g会随图像和时间改变，属于局部响应信息；但“输入依赖状态”不自动证明
“一个实时状态指标比纯时间/匹配控制有独立收益”。

也不能说Jacobian完全没有几何信息。写s_theta=-epsilon_theta/b，有

\[
J_m=a^{-1}[I+b^2\nabla_x s_\theta(x,t)].
\]

当s恰为真实score时，后一导数为log p_t的Hessian；理想MMSE去噪器还满足
J_m=(a/b²)Cov(x_0|x_t)。这些是带假设的局部先验几何关系。
训练网络不保证恰是理想score/条件均值，不能把它当已校准后验协方差，更不能由此声称已经度量、控制了整条轨迹曲率。
当前U确实能改变向量方向，不只是给原q加一个倍率；未成立的是其与原开题联合控制体系的等同。

## 3. 组别、超参数与自适应项

|组|每一步真正施加的加性位移|该对照能回答什么|
|---|---|---|
|C_REPLAY|原q，原精度|原本地ZAPS及严格回放一致性|
|C_IEEE|原q，与新组匹配IEEE精度和图前向|排除TF32/图前向差别；是主要新组对照|
|Q_ORIGINAL|K‖v‖ q/‖q‖|保留q方向，只改长度规则；不是原C|
|V_PHYSICAL / V_FFT|Kv|去掉网络输入Jacobian，仍有物理加权与当前残差|
|U_CURRENT / U_FFT|K‖v‖ u/‖u‖|当前响应方向与当前幅度；FFT只表示求解后端|
|TIME_GAIN|K ḡ_task(t) u_current|仅移除本图幅度g，保留当前u|
|OLD_U|旧128-CG能通过时用U，否则回到q|A62数值覆盖对照；不是完全无求解|

Q/V/U在各自当前状态使用相同范数规则，轨迹分开后数值范数不必相等。
Q与C比较包含物理范数、K及不同轨迹；U与Q/V比较支持方向组件的增量，不能把均值差当可相加的因果百分比。

TIME的ḡ是5张旧FFHQ图在每个固定step的g中位数；每任务30个数，在看新图结果前冻结。
ImageNet直接使用FFHQ的时间表。没有重新挑位置，没有限定只在t319引导，也没有重新做0—2倍率搜索。
SR时间表范围0.01150—3.03792；IP范围0.02151—4.30477。所有30点都按规则计算。
TIME每步仍用当前u；其好结果不能证明“完全不需要状态/当前方向”。

|类别|具体内容|怎样变化|
|---|---|---|
|正常U实时量|m、r、v、u、g|逐图逐步，由当前观测、当前x和模型决定；没有GT或在线拟合|
|固定物理条件|H、退化核/掩码、已知σ_y|随任务/观测定义；不是算法自己识别出来的最优参数|
|固定方法规则|c=b²/a，范数匹配，gamma=1，原30点，DDPM eta=1，clean clamp|本轮不逐任务扫描，不根据曲率改变|
|数据集相关|FFHQ/ImageNet各自预训练模型|数据与模型同时变化，不能称同一个检查点的跨域泛化|
|原ZAPS适配|ζ_t、D_t逐观测优化；10轮，D初值.2，db4/3层|仍用于C/Q/回退；正常U目标不依赖其学习值|
|原基线固定配置|A56 ζ初值.1，FFHQ lr=.01、ImageNet lr=.001；两者use_learned_var=false|是锁定本地协议，未认证作者benchmark一致|
|数值超参数|真实残差rtol=1e-5；旧CG128；A62条件数预算、上限2048/每解300秒；32eps分辨下限|决定数值准入/失败策略，不是质量状态指标|
|TIME离线量|每任务30个g中位数|旧图冻结；非当前图自适应|

原Gaussian/Motion基线ζ初值.2，其他原基线配置详见父plan。不能把这些固定设置称“无超参数”。
mask、观测噪声、seed与随机创新均按配对协议固定；输出先冻结，再用GT算指标。

## 4. 质量和时间的完整32行消融

**口径必须先读：**

- A56：6张未用FFHQ图×SR/IP；4张未用ImageNet图×SR/IP；每图seed1000，σ_y=.05，256×256。
- A62：5张旧开发FFHQ图×Gaussian/Motion；同seed/噪声。不能与A56当同一独立确认集。
- 表中C是锁定本地ZAPS参考，未认证作者论文完整复现；ImageNet IP是本地任务扩展。
  当前基线数值偏低，不能据此直接声称超过论文ZAPS或CLAMP；也不能因为提升大而自动归咎基线有bug。
- 时间为每图**末轮固定30点**的实测均值，包含现有诊断/记录和机器共享负载影响，排除加载和此前优化。
  C_IEEE减去了单独记录的VJP计量探针时间；原始值、扣除量、median及显存数据在CSV。
- 不是从完整300次优化时间中扣去另一次回放来推算前270次，也不是完整端到端U计时。
  A56仍计入已发生的300科学调用协议；A62复用历史前缀，新增4臂×30×10=1200诊断前向。
- 原300次是10轮优化展开×每轮30个时间点，不是一条含300个采样点的去噪轨迹；新臂比较的是配对末轮30点。
- 50份run.json逐一以归档manifest的SHA256、字节数以及Git blob SHA1核对；没有下载PT。
- PSNR/SSIM越高越好，LPIPS越低越好。均值是有限样本描述，没有新增显著性或置信区间声明。

| 数据集 / 任务 | 组 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 末轮30步秒/图 ↓ | 相对C_IEEE时间 |
|---|---|---:|---:|---:|---:|---:|
| FFHQ / SR | C_REPLAY | 19.790 | 0.5424 | 0.3574 | 1.92 | 0.81× |
| FFHQ / SR | C_IEEE | 19.791 | 0.5423 | 0.3573 | 2.36 | 1.00× |
| FFHQ / SR | U_CURRENT | 27.726 | 0.8072 | 0.2039 | 5.71 | 2.42× |
| FFHQ / SR | V_PHYSICAL | 26.645 | 0.7323 | 0.3082 | 3.78 | 1.60× |
| FFHQ / SR | Q_ORIGINAL | 26.541 | 0.7257 | 0.3373 | 3.98 | 1.69× |
| FFHQ / SR | TIME_GAIN | 27.758 | 0.8105 | 0.1977 | 6.60 | 2.80× |
| FFHQ / IP | C_REPLAY | 21.980 | 0.6282 | 0.3000 | 1.92 | 0.79× |
| FFHQ / IP | C_IEEE | 21.976 | 0.6281 | 0.3002 | 2.42 | 1.00× |
| FFHQ / IP | U_CURRENT | 30.369 | 0.8801 | 0.1155 | 4.86 | 2.01× |
| FFHQ / IP | V_PHYSICAL | 22.340 | 0.6224 | 0.3976 | 3.00 | 1.24× |
| FFHQ / IP | Q_ORIGINAL | 22.342 | 0.6225 | 0.3975 | 3.02 | 1.25× |
| FFHQ / IP | TIME_GAIN | 30.404 | 0.8803 | 0.1182 | 5.54 | 2.29× |
| IMAGENET / SR | C_REPLAY | 17.934 | 0.3997 | 0.5383 | 6.92 | 0.79× |
| IMAGENET / SR | C_IEEE | 17.932 | 0.3996 | 0.5378 | 8.78 | 1.00× |
| IMAGENET / SR | U_CURRENT | 24.080 | 0.6497 | 0.4311 | 19.60 | 2.23× |
| IMAGENET / SR | V_PHYSICAL | 22.282 | 0.5318 | 0.6311 | 9.78 | 1.11× |
| IMAGENET / SR | Q_ORIGINAL | 22.467 | 0.5437 | 0.6491 | 10.13 | 1.15× |
| IMAGENET / SR | TIME_GAIN | 24.188 | 0.6706 | 0.4310 | 20.81 | 2.37× |
| IMAGENET / IP | C_REPLAY | 19.365 | 0.4940 | 0.3480 | 7.02 | 0.78× |
| IMAGENET / IP | C_IEEE | 19.350 | 0.4940 | 0.3485 | 8.99 | 1.00× |
| IMAGENET / IP | U_CURRENT | 26.371 | 0.7675 | 0.2620 | 19.43 | 2.16× |
| IMAGENET / IP | V_PHYSICAL | 20.799 | 0.5405 | 0.4421 | 9.55 | 1.06× |
| IMAGENET / IP | Q_ORIGINAL | 20.799 | 0.5405 | 0.4421 | 10.17 | 1.13× |
| IMAGENET / IP | TIME_GAIN | 26.196 | 0.7623 | 0.2474 | 29.25 | 3.25× |
| FFHQ / Gaussian | C_IEEE | 24.977 | 0.7519 | 0.2214 | 2.05 | 1.00× |
| FFHQ / Gaussian | OLD_U | 28.723 | 0.8589 | 0.1790 | 14.43 | 7.05× |
| FFHQ / Gaussian | U_FFT | 28.881 | 0.8609 | 0.1770 | 217.30 | 106.19× |
| FFHQ / Gaussian | V_FFT | 28.195 | 0.8159 | 0.2186 | 213.56 | 104.36× |
| FFHQ / Motion | C_IEEE | 21.413 | 0.6357 | 0.2923 | 1.94 | 1.00× |
| FFHQ / Motion | OLD_U | 27.810 | 0.8261 | 0.1589 | 14.43 | 7.43× |
| FFHQ / Motion | U_FFT | 27.892 | 0.8348 | 0.1772 | 232.53 | 119.81× |
| FFHQ / Motion | V_FFT | 23.924 | 0.5380 | 0.3516 | 232.56 | 119.82× |

完整CSV：[ablation_quality_time.csv](ablation_quality_time.csv)。
复算：在仓库运行 `python3 projects/reports/20261028_method_ablation/aggregate.py`，只聚合轻量结果，不调用模型。

## 5. 从消融能得出哪些结论

| 数据集/任务 | Q−C_IEEE PSNR | U−Q PSNR | U−V PSNR | U−TIME PSNR |
|---|---:|---:|---:|---:|
| ffhq/super_resolution | +6.751 | +1.185 | +1.081 | -0.032 |
| ffhq/inpainting | +0.366 | +8.027 | +8.029 | -0.034 |
| imagenet/super_resolution | +4.535 | +1.613 | +1.798 | -0.108 |
| imagenet/inpainting | +1.449 | +5.572 | +5.572 | +0.175 |
| ffhq/gaussian_deblur | — | — | +0.686 | — |
| ffhq/motion_deblur | — | — | +3.968 | — |

**方向有小样本迁移证据，逐图g没有清楚的额外收益。**

A56的U相对Q/V在四个数据集—任务组合的平均PSNR均正向，原报告中也同时列SSIM/LPIPS。
但ImageNet SR的00195等单图存在U不如V/Q，不是所有图片必胜。
U相对TIME在FFHQ SR、FFHQ IP、ImageNet SR分别-0.032、-0.035、-0.108dB；ImageNet IP为+0.175dB，
同时LPIPS恶化+0.01465，超过用户既定+.005保护线。故不能说当前实时幅度比时间表更好且更均衡。
FFHQ SR的U相对TIME也有SSIM -0.00326、LPIPS +0.00619；不能只报告PSNR接近。

A62的U相对V在两blur上均5/5图PSNR正向，是旧图方向消融证据；不是独立blur迁移确认。
更完整求解相对OLD_U的Gaussian平均+0.157dB、Motion+0.082dB。
Motion同时LPIPS从.1589变为.1772（+0.01829），而00608的PSNR相对OLD_U约-.453dB。
完整数值准入并不等于整体感知质量更好。

旧CG在两blur只33/150和44/150点通过，新U/V各300/300通过；结果变化包括从q回退切换为U的动作覆盖，
不是在完全相同控制动作下单纯提高float精度。

## 6. 时间代价与目前不能下的效率结论

正常U每30点需要30次前向、30次网络输入VJP，以及30次物理逆作用；V/Q不做网络输入VJP但仍构造v。
TIME同样30次输入VJP，当前实现还保留v/g读数和闭合检查，因此其计时不是最简“查表幅度”实现。
现有V/Q对照保留图前向以匹配计算算术，也不是它们可以达到的最小部署时间。

A56末轮U相对匹配C_IEEE为2.01—2.42倍时间；ImageNet图前向的显存与输入反向也需一并考虑。
A62末轮Gaussian U约217.30秒/图、Motion约232.53秒/图，相对旧U约15.06倍和16.12倍，
两任务累计15.59倍；相对C_IEEE约106—120倍。V_FFT也同样慢，说明该实现中主要瓶颈在物理求解，
而不是单次网络输入反向。这里FFT是CPU float64卷积动作加迭代求解，含CPU/GPU传输和准入检查；
不是把原zero-padding目标改成循环卷积逐频直接求逆。

| 数据集/任务 | 原本地ZAPS完整300前向优化秒/图 | 完整U部署时间 |
|---|---:|---|
| ffhq/super_resolution | 20.87 | 未同场测量 |
| ffhq/inpainting | 20.32 | 未同场测量 |
| imagenet/super_resolution | 71.95 | 未同场测量 |
| imagenet/inpainting | 67.92 | 未同场测量 |
| ffhq/gaussian_deblur | 本轮未取回/未重测 | 未同场测量 |
| ffhq/motion_deblur | 本轮未取回/未重测 | 未同场测量 |

A57旧10例独立U平均5.50秒/图，证明该旧开发队列可以30调用输出等价；未与同场完整基线计时构成全协议benchmark。
不能把A56的新图300调用协议在事后改写成30，或把A57结论直接覆盖ImageNet/blur的完整部署成本。

## 7. 与开题的对应及创新性判断

|开题要求/论文主张|目前状态|能写的边界|
|---|---|---|
|物理一致性状态|已有r及噪声归一化读数；S为代理|局部目标有定义，但残差小不等于全图质量高|
|当前先验/去噪响应|已有真实输入VJP；U/Q/V消融正向|有限样本方向机制；与已有直接似然引导高度相关|
|实时幅度指标的增量价值|当前g相对TIME未获均衡优势；A54旧图也未独立支持|保留负结论，不把U对C的大提升当g的收益|
|轨迹曲率/稳定性|A53余弦与A59局部探针未建立终局控制证据|Jacobian是局部导数；不是轨迹曲率控制已实现|
|统一代价、步长/权重联合控制|当前30点固定，gamma=1，步长线暂停|尚未实现，不能用归一化代替这项论断|
|多任务多数据集|新FFHQ/ImageNet仅SR/IP；blur为旧FFHQ|有局部迁移迹象；ImageNet blur、原定去噪缺失|
|质量可比下降低总体NFE/时间|A57有旧队列无学习等价；A60滞后方向质量下降；blur目前昂贵|无全任务效率结论|
|独立新颖性|范数匹配是相对JAPS的具体变体；pullback和代理S已有先例|尚不能把现成组件命名为统一新体系；没有CLAMP/JAPS同协议对照|

当前可以讲成立的故事是：**在锁定实现中拆解了引导方向、物理范数与实时幅度的作用；当前响应方向有跨两数据集SR/IP的
小样本证据，而实时幅度优势和高成本求解的必要性尚未成立。**
目前不能讲成“开题完整体系已经验证”或“一个全新的曲率控制算法已实现且更快”。
这些结果是有用的研究证据，和论文创新充分性是两个判断。是否足以支撑最终硕士论文还取决于学校要求、
方法与先行工作的清晰差异以及未完成的公平对照，不能保证毕业或强迫所有任务正向。

实验继续暂停；本报告只整理了既有结果。A63草稿未发布，未把TIME缺失的blur或ImageNet blur补成虚构结果。

## 8. 原始证据和可追溯性

- A56完整20例结果：`results/frozen-direction-transfer-selection-r2-20261027`，
  commit `7e67d6a3c8872ee903a461d6aa461d610a031693`，
  bundle `projects/server_records/uncertainty_feedback_2079b3b5ace87ac9`。
- A62完整10例结果：`results/blur-fft-quality-20261027`，
  commit `cdee9ba93c0d83fa2c59beafc469574ea5b3fe4a`，
  bundle `projects/server_records/refinement_confirmation_cb6011b3363d3dbe`。
- 本地已审计摘要：`projects/reports/20261027_completed_transfer_received/`、`projects/reports/20261028_a62_received/`。
- 本次50份逐例计时的轻量提取、源SHA与校验结果：`verified_timing_records.json`、`provenance.json`。
- A57：`projects/reports/20261027_a57_a60_readout/received/A57_summary.json`。
- 详细算法推导/伪代码/开题原PDF页核对见：`projects/reports/20261028_algorithm_explained/REPORT.md`。
- 主要源码：`utils/current_score_runtime.py`、`current_score_runtime_r2.py`、`current_direction_control.py`、
  `norm_gain_control.py`、`frozen_direction_transfer.py`、`diag_frozen_direction_transfer.py`、`diag_blur_fft_quality.py`。
- A56科学调用：6000原协议+3624诊断前向；输入VJP1204。另M1资源资格72前向/32输入VJP单列。
  A62新增1200诊断前向、377输入VJP；历史前缀未重算，不把复用历史成本写成零总成本。
