# JAPS与现有ZAPS的兼容性：推导审查，不是新方法结果

状态：2026-10-26第二轮夜间研究。已读JAPS v3 §3.1—3.4及对应作者实现，
尚未完成附录、全任务参数、主文所有实验与官方算子实现审计，未运行GPU。
主文：https://arxiv.org/html/2511.18471v3#S3
作者实现：https://github.com/liavhen/JAPS/blob/main/guided_diffusion/condition_methods.py
已取回文件Git blob：ca3c5b823eaf5768f7ec6df3c6b2617f0b8acc73；
采样文件gaussian_diffusion.py blob：d3ee2ea3209336cb3832abfa46d47ae1f0210b52。

## 1. 先分清原版Q与候选状态

现库modules/zaps_algorithm.py约406—409行与diag_fixed_count_state_positions.field
（89—95行）一致：残差r=y-A(m)，v=L^T r，Hessian近似动作WDW^T v，

    M_t = [I+(1-alpha_bar_t) W D_t W^T] / sqrt(alpha_bar_t)
    q_t = zeta_t M_t L^T r_t

q是实际加到下一采样状态的预条件物理修正，已包含学得zeta、D、噪声系数、
算子和残差；不是原始梯度范数，不是条件协方差。当前field中eps.detach()，
因此M不等于通过真实网络VJP得到的denoiser Jacobian。
原优化器仅投影zeta>0，D不限制符号；M并无自动PSD保证，也不能由D为负
就断定M已经非PSD。具体谱界须检查1+(1-alpha_bar_t)*D，而非只看D的符号。

## 2. JAPS给出了什么已有机制

在线性Gaussian近似下，主文用同一正则化回投构造proximal参照v和direct参照u，
将v投影到span(u)，系数为内积/平方范数；这是已有方法，不是本项目的新公式。
其代码clean_likelihood先求正则化回投，再通过真实denoiser VJP求u。
conditioning中的r_t_squared=(1-alpha_bar_t)/sqrt(alpha_bar_t)，实际是累计alpha；
网页中相邻alpha/累计alpha符号不能混抄。默认系数还经_alpha_scale与skip转移系数。

因此“计算一个夹角后放大”不是完整复现，必须同时核对：同一proximal算子、
Jacobian路径、噪声单位、前向/伴随、时间跨度和投影方向，及源码scale/decay设置。
作者“相对direct方法无额外成本”不表示相对本库stop-gradient ZAPS也免费。
真实网络VJP不一定增加forward NFE，但会增加显存、反向计算和walltime。

当前原残差q与正则化回投方向也不是简单相同的向量。直接把M替代真实J，
或把q作为u再投影，都只能叫新的ZAPS-adapted近似，不能继承JAPS保证。
投影若抹掉原zeta的尺度，还可能让前面逐图优化的一部分工作失去作用，需消融。

## 3. 可精确核查的步长—引导系数联系

令a=alpha_bar_t、b=alpha_bar_s，0<a<b<1，原skip DDPM的eta=1。
令sigma_ts²=(1-b)/(1-a)*(1-a/b)。把条件似然得分写入预测噪声后，
附加位移的时间系数可写为（来自DDIM代数，而非状态最优性）：

    Gamma_ts = sqrt(b/a)*sqrt(1-a) - sqrt(1-b-sigma_ts²)
    K_ts = sqrt(1-a)*Gamma_ts
         = sqrt(b/a) - sqrt(a/b)
         = (1-a/b)/sqrt(a/b)

K依赖真实跨度(t,s)，不是只依赖t。此恒等式已经用NumPy随机合法alpha对
做零网络代数检查，记录于step_coefficient_check.json；不表示神经采样正确性。

由此可以明确一个新网格协议：在原锚点把q/K_base仅作为得分代理的表示，
变跨度时显式乘K_new；原网格必须走原算术顺序分支以保持null等价。
若对学得参数作插值，至少要说明插的是zeta本身还是zeta/K_base，以及D的坐标。
现A23仅在noise_u插值zeta/D，field不读取下一步s；所以步长与有效引导剂量
并未按这一特定条件得分解释匹配。它未证明旧实现错误，也不推翻旧负结果；
只说明后续“只改步长”的新对照需要明确选择哪种离散作用约定。

重要限制：K尺度修正不是任意粗细分割的端点半群保证，也不是A3旧冻结clean/G_t
桥的同一个null。A3通过1/(1+A_ms)分配原q以匹配特定冻结场端点；两者假设不同。
不能仅发现这一系数就宣称改网格一定变好，亦不能把h变量引入后保留总guidance
剂量偏差却全归因“状态选步”。终端t0->输出另按原语义处理，不偷删末次调用。

## 4. 单位域与实际算子

本库四任务使用A(x)=H((x+1)/2)，线性部分L=.5H。proximal系统必须以
L L^T与unit sigma_y²定义，残差继续y-A(m)，保留Gaussian padding的H(1)项。
JAPS默认代码采用model域并将unit噪声标准差乘2；两种写法可代数转换，但
不能将原sigmay=.05、model域残差和unit域伴随混搭。
作者某些FFT/伪逆基于其算子边界；我们zero-padding/精确bicubic/随机mask/motion
不保证可直接用同一闭式伪逆。必要时CG必须按实际L/transpose实现，计入walltime。

## 5. 对统一体系的启发与最低实验

有依据的统一不是将residual、cosine、rho任意相加：当前观测/先验差异决定条件
得分修正，数值局部误差决定h，K(t,s)把跨度和该修正的实际位移连接起来。
历史转向可作廉价触发/诊断，但必须另证其能识别有益加步；不等同局部截断误差。

先做四个0/小预算门槛：
1. 原网格/关闭状态的q与输出完全复现；单位域的proximal线性系统残差和伴随通过。
2. 当前q、M作用与真VJP明确分层，测投影比例、实际动作改变、耗时及显存。
3. 固定网格单因素比较原C、同公式固定倍率、当前状态投影及打乱状态，不能只赢弱C。
4. 若权重模块合理，再单独审查K新/旧跨度作用和数值误差指示器，之后才联合。

这只是候选方向的兼容性研究，尚无可执行新GPU入口/新卡或成功结论。
