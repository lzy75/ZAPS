# A126 — A118 全系数支撑与真实下一次去噪器响应的迁移

日期2026-11-22；工作线A；状态planned。服务器数值准入后运行，不代表新结果。

## 问题、可否定假设与身份

A118 的闭式完整系数细化在 ZAPS 模糊任务有正向结果。A124/A125 将其搬到
DPS/ΠGDM 后，恒等响应下增大步长/迭代不能稳定优于标量。A122/A123 使用真实
下一次响应，但只求射线和一个切向组成的二维子空间。
新增信息：使用 A118 完整系数支撑与实际下一次干净预测联合优化修正。
假设：这比闭式标量、旧代理对角和相同目标的连续射线有更好的终局质量。
不预设成立；当步损失下降不是终局 PSNR 排序定理。
父run为 /data0/lzy/ZAPS-results/diag_a118_blur_transfer/A124R1_{host}_ffhq，必须complete。
检查其A118父计划、源文件、权重、图像、核、观测和原方法记录hash。
不修改共享代码/已有计划，不覆盖历史结果。只允许小规模迁移机制结论。

## 固定条件、变量和比较臂

FFHQ已用00108/00175/00431 × Gaussian/Motion × sigma_model=.05，共6例/模型。
没有校准/测试拆分或GT选参数；不称独立测试。GT仅用于终局评价。
复用A124R1 y/H：DPS zero-boundary，ΠGDM native circular；跨模型不伪称相同H/y。
权重/采样器/物理q/原网格/输出/噪声不变；每臂独立从原seed重建。
介入时间同A125：将A118三十时间点映射到native网格，覆盖全噪声范围。
DPS原1000位置，ΠGDM原100位置，无ZAPS十轮结构。

四个新完整计时臂：SCALAR原闭式倍率；PROXY_DIAG复用A125 K1/step1；
RESPONSE_RAY相同实际响应目标从闭式标量起做1次射线优化；RESPONSE_DIAG在每个
当前位置做相同射线优化后，再做2次完整系数更新。都只在30位置介入。
BASE六例仅继承指标/历史时间。每模型24次新重建，两模型48次。

## 目标与求解

z_s为原无引导下一状态（包括原随机噪声）；C_s为该基线原干净预测。
ΠGDM用raw predictor，DPS用其原clipped pred_xstart；C_-1为恒等映射。
h0=PH C_s(z_s)，r_s=Py-h0，omega_s=(1-M sigma_y²/||r_s||²)+，b_s=h0+omega_s r_s。
F_s(delta)=.5||PH C_s(z_s+delta)-b_s||²。
同位置固定z_s/b_s/q。射线delta=gamma q, gamma>=0；对角delta=W^T p，
supp(p) subset supp(Wq)，全部非零系数联合更新，可正可负，不按频带共用倍率。
非零c=Wq上p_i=g_i c_i。继承A118自由系数结构，不采用A122的二维限制或c²预条件。

每次迭代精确VJP求梯度并投影到射线/完整支撑，一个JVP求方向响应曲率：
eta=<-grad,d>/||PH J_C d||²；最多3次实测Armijo检查，不下降逐次减半。
对角从相同状态的连续射线继续下降；不是比较不同轨迹损失。
极小梯度、线搜索预算、迭代预算分别记录；2次迭代不称收敛最优。
失败试探保留当前迭代点，是求解器停止，不是GT/质量筛选。
NaN、无效曲率或身份/RNG门控失败立即停止，不自动重试。
C_s为恒等、alpha=1时一次完整更新必须退化为A118更新（服务器门控）。
omega仍是名义噪声校正启发式，不是无偏最终风险。

## 成本、门控、判定

每位置1次h0查询，射线最多1VJP+1JVP+3次试探；对角再最多2VJP+2JVP+6次试探。
最多16个预测查询/位置；终点C_-1不调用NN；29非终点位置，故对角完整NFE
DPS<=1464、ΠGDM<=564。实际低于上限时报告实测，不填上限冒充实测。
SCALAR/PROXY仍native1000/100F和同数量原输入VJP。
新增VJP/JVP、终点仅算子导数、拒绝试探、H/小波调用单独计数；不宣称等NFE。
完整时间含输入/算子/init/采样/控制/CPU同步/进度；模型加载/门控/指标/写入另计。
不声称加速，不掩盖真实响应成本。

准入：本机协议检查；服务器真实db4退化一致、各向异性/非线性目标下降、native
scalar输出/RNG、终点成本；真实FFHQ checkpoint VJP/JVP准入。
每任务对三个参照分别报告平均PSNR>=+.1dB、>=2/3赢、SSIM不降/LPIPS不升。
正向后固定方案补其余七图；负向不扩大数据集或继续扫描步长，保留全部输出。

## 运行与归档

/home/lzy/ZAPS 中运行 projects/run_full_coefficient_response.sh {pigdm,dps} --gpu {1,0}。
默认分离进程并回传日志到结束；--status只读；--archive完成后显式上传。
目录 /data0/lzy/ZAPS-results/diag_a126_full_response/A126_{host}_ffhq。
归档临时git仓库位于DATA0，不自动删除原数据或检查点。
