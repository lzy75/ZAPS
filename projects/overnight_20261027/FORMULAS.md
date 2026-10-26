# 公式推导与可验证逻辑

核对时间：2026-10-27凌晨。来源代码已逐段核对；公式是明确假设下的数学说明，不是未运行模型的效果或后验保证。
原30点固定，当前不设计步长h或采样位置。

## 1. 符号与原ZAPS修正

alpha_t表示累计alpha_bar，a_t=sqrt(alpha_t)，b_t²=1-alpha_t，x0采用[-1,1]模型强度坐标；x_t是其含噪状态，并不逐值受[-1,1]约束。
网络的raw条件均值估计与DDPM使用的clean分开：

$$m_t(x)=(x-b_t\epsilon_\theta(x,t))/a_t,\qquad \bar m_t=\operatorname{clamp}(m_t,-1,1).$$

物理观测在[0,1]域，A(m)=H((m+1)/2)=Lm+d，L=.5H，d=.5H(1)，Lᵀ=.5Hᵀ。
不能对zero-padding blur把d直接当空间常数；也不能将unit sigma_y=.05和model域噪声混用。
补洞的公式在有效观测子空间解释；代码将非观测残差置零，RMS只统计有效观测。

原ZAPS加到下一状态的修正是q，不是余弦或单纯梯度：

$$q_t=\zeta_{i(t)}\,\frac{I+b_t^2 W D_{i(t)}W^\top}{a_t}\,L^\top[y-A(\bar m_t)].$$

W/D对应小波合成、逐系数对角、分析；zeta/D逐图优化。代码没有通过去噪器求这里的真实VJP。
D不受非负约束，不能由这条写法自动断言矩阵正定。C_IEEE是该锁定本地基线的精度匹配回放。
Q_ORIGINAL则是下述归一化方向消融，不等同完整原版C。

代码定位：modules/zaps_algorithm.py::_reverse_diffusion，utils/current_score_runtime.py::field_from_mean；
原网格参数映射为i(t)，不能混成正向循环索引k。

## 2. 物理目标v来自哪个问题

固定t，定义raw残差r=y-A(m_t(x))，代码选择各向同性代理：

$$c_t=b_t^2/a_t,\quad S_t=\sigma_y^2 I+c_t LL^\top,\quad\eta_t=\sigma_y^2/c_t.$$

$$B_t w=(LL^\top+\eta_t I)w=r,\qquad v_t=c_t^{-1}L^\top w=L^\top S_t^{-1}r.$$

S_t正定源于sigma_y>0，而非神经Jacobian正定。v_t是噪声/退化校准的物理回投方向。
代码proximal_vector按真实L/Lᵀ迭代，不用错误的周期边界替代实际算子。

另一个等价解释：若人为采用x0|x_t约等于N(m_t,c_t I)，其线性Gaussian条件均值位移为c_t v_t。
这只是代理模型的条件化。**代码没有估计真实后验协方差。**
理想MMSE均值满足Cov(x0|x_t)=c_t J_m；c_t I可作J_m近似I的粗解释，不能把有限训练网络当理想MMSE。
TMPD给出条件矩和相关近似的既有基础，非本项目新定理。[TMPD v3 §3.1-3.2](https://arxiv.org/html/2310.06721v3#S3.SS1)

代码定位：current_score_geometry.py::proximal_vector；CG_RTOL=1e-5、CG_MAXITER=128。

## 3. 当前响应方向u是哪个目标的负梯度

固定t且S_t不随x变化，定义标量代理：

$$\ell_t(x)=\tfrac12 r(x)^\top S_t^{-1}r(x),\qquad J_m=\partial m_t/\partial x.$$

由dr=-L J_m dx，

$$d\ell_t=-dx^\top J_m^\top L^\top S_t^{-1}r,\qquad -\nabla_x\ell_t=J_m^\top v_t=:u_t.$$

VJP中detach(v_t)正是链式法则的cotangent，不是丢掉一阶梯度。它避免对当前线性求解过程再求导。
若改用随x变化的协方差而仍忽略其导数，则上述等式只对“冻结协方差”的局部代理成立。

这是精确线性求解下的恒等式。真实残差1e-5门控不能在未知条件数/Jacobian范数下自动给出u的方向误差界。
数值门控、实际神经VJP检查、终局质量实验三者不可合并为一个保证。
网络参数requires_grad=True用于兼容原checkpoint内部反向；只请求输入梯度，不更新权重或累积参数.grad。

代码定位：current_score_geometry.py::input_vjp，current_score_runtime_r2.py::input_vjp_parameters，
angular_feedback_control.py::sensed_field。Jacobian转置是本报告的列向量约定，不能拿它与论文记号差异当创新。

## 4. 为什么跨步系数是K

VP Gaussian通道下，精确条件均值差与似然score g满足Delta m=(b_t²/a_t)g；
这是已有恒等式，要求前向噪声与测量在给定x0后条件独立。近似网络/近端均值不能让g自动精确。
[JAPS v3 Proposition3.1/AppendixA.1](https://arxiv.org/html/2511.18471v3#A1.SS1)

令theta=alpha_t/alpha_s，t>s，终端s=-1约定alpha_s=1。原skip-DDPM clean系数为

$$C_{t,s}=\sqrt{\alpha_s}(1-\theta)/(1-\alpha_t).$$

将上面的Delta m代入该线性clean项，得到位移单位换算：

$$K_{t,s}=C_{t,s}b_t^2/a_t=(1-\theta)/\sqrt\theta.$$

K依赖t和s，不是只看t；当前两者仍全部取原30点。这个代数不是状态最优步长或后验收敛结论。
实际生产DDPM使用clamp(m)，且引导在后验步之后相加；clamp(m+Delta m)-clamp(m)一般不等于Delta m。
因此K在当前代码中是有来源的作用单位，不构成“已经精确实现条件DDPM”的证明。

代码定位：zaps_algorithm.py::ddpm_posterior_step（c1），current_score_geometry.py::skip_coefficient。

## 5. U如何决定方向和幅度

当前候选在数值可分辨时应用

$$T_U=K\frac{\|v_t\|}{\|u_t\|}u_t,\qquad g_t=\|v_t\|/\|u_t\|.$$

它选择u的方向，将长度校准到K||v||。两条可验证性质是

$$\|T_U\|=K\|v_t\|,\qquad\nabla\ell_t^\top T_U=-K\|v_t\|\|u_t\|<0.$$

后者只说明理想数学目标的**局部纯引导方向**。若ell在线段上L-smooth，x+gamma T_U满足标准下降充分条件

$$0<\gamma K\|v_t\|/\|u_t\|<2/L.$$

gamma是位移倍率，不能误写gamma<2/L；有限方向差分也不能提供通用L上界。
DDPM基漂移、噪声、clamp及后续状态不在该纯梯度命题内，故不能推出PSNR/SSIM/LPIPS提高。
CG/向量不可分辨时用原q fallback，以上正常分支性质不自动适用。

同状态的方向参照是T_V=Kv，T_Q=K||v||q/||q||。分岔之后每臂使用其自己的当前状态，
所以是同一长度**规则**，并非各路径逐步数值长度相等。V/Q仍有状态依赖，不能称完全无状态。

JAPS的已有投影为K(<v,u>/||u||²)u=K(||v||/||u||)cos(v,u)u。
U去掉此夹角因子，采用物理范数校准；归一化、投影、条件score和K来源均有先行工作。
已有A53是旧图对夹角因子的消融，不足以证明U在新图优于正式JAPS。
[JAPS v3 §3.3](https://arxiv.org/html/2511.18471v3#S3.SS3)

代码定位：angular_feedback_control.py::controlled_target，cosine=1时对应U；current_direction_control.py对应V/Q。

## 6. 物理一致性不等于全图质量

代码R_raw=RMS_obs(r)/sigma_y，是观测偏差读数，不是完整图像风险或真实后验概率。
重建依赖观测后，不能无条件要求R趋于1，更不能要求残差趋于0。
例如单位域GT=(.5,.5)、H仅观察第一像素，m=(.6,.5)到m'=(.55,1)时，
测量残差从.1降至.05，而全图MSE从.005增至.12625。未观测方向的误差不可由测量残差识别。
这是解释性代数例子，不是模型实验。J_mᵀ可以把观测信息传入补洞区域，但也不能保证所传信息正确。
A49是**另一个局部目标J**与终局质量错位的实证，不能冒充本节ell的直接反例。

## 7. 固定t几何、随机轨迹和Hessian须分开

对固定t的光滑向量场u(x)，设d_E=u/||u||。其自治积分曲线的欧氏曲率为

$$\kappa(x)=\|(I-d_Ed_E^\top)D u(x)d_E\|/\|u(x)\|.$$

这个公式不直接适用于时间变化且有Brownian创新的DDPM轨迹；相邻步余弦同时混入t、噪声、跨度和先验漂移。
A59的d_R=u/RMS(u)=sqrt(n)d_E，eps=s max(1,RMS(x))，s=.001/.002，在同t探测x±eps d_R。

$$\Omega_\epsilon=\|\hat u_+-\hat u_-\|/(2\epsilon),\quad
S_\epsilon=\|u_+-u_-\|/(2\epsilon\|u_0\|),\quad
B_\epsilon=\|u_++u_--2u_0\|/(\epsilon^2\|u_0\|).$$

Omega是每输入RMS的方向变化率，Omega/sqrt(n)才对应每欧氏单位输入长度；S同时含方向与幅值变化，B是有限尺度非线性。
两尺度一致只说明当前尺度可读，不能认证全局Hessian范数、整个方向空间、随t变化或采样截断误差。
A59将它与最终剂量对比关联，仍需实验，不能提前称曲率控制成功。

## 8. 方向滞后与优化可移除：待验证命题

A60缓存d_bar=u_old/||u_old||、保持当前长度，则

$$\nabla\ell_{current}^\top T_{lag}=-K\|v_{current}\|\|u_{current}\|\cos(u_{current},d_{bar}).$$

夹角为正才保留局部代理下降方向。滞后步不付费计算当前u，故没有免费在线夹角证书。
旧方向也滞后了x、t、v，不能叫精确冻结Jacobian。固定1/2/4刷新只检验必要性和成本，不是自适应曲率触发。

完全resolved分支的T_U/T_V不含学习后的zeta/D，但数值q_floor、fallback和随机流仍可能造成依赖。
A57因此从seed独立构造init、跳过9×29创新、固定未学习参数并禁止optimize；输出冻结后才读取PRE审计。
通过前不能把300→30写成已证明的结果，且前向减少不等于相同比例的总计算/墙钟加速。

## 9. 与开题的逻辑接合及证据边界

原开题物理第3/7/10/11/12页已完整提取并视觉核对，原件SHA256为728d0078fac1f3f1c9b5ea75dd2e2f5f84d49ca6d74d28ebc5d3e23a38fb0499。
原目标包含双指标、统一代价、贪心选步、动态权重与NFE效率；限制变化率本身不证明收敛。
当前仅研究固定网格的状态引导子问题，原h/调度/联合体系没有完成，且用户要求暂停。

可检验链条：原C回放 → 当前状态/方向可测 → 对照分离方向与标量 → 终局质量/成本 → 未用图确认。
A55r2目前支持旧5图的方向差异；A53/A54不支持原夹角/当前g独立价值；A56—A60仅为已推待运行包。
新颖性需要匹配的现有条件score/归一化/投影方法对照；标准公式不能直接列为新贡献。
