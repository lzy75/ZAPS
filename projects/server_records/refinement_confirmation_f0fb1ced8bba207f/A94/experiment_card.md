# A94：逐系数正负倍率的联合非线性响应求解

状态planned；A工作线；用户明确要求据A93结果进一步设计目标与实验。先固定状态诊断，不混入窗口长度变化。
问题：A93十子带共用倍率只能到达0.34—0.60方向余弦，逐系数非负形式上限0.85—0.93。直接g_j/c_j可复现向量但不稳定且没有联合重建目标。
本次用完整下一去噪器与H的数据项联合更新所有gamma，不拟合参考g本身，不将Gram矩阵强行对角化。

## 来源、固定条件与作用范围

父A92 /data0/lzy/ZAPS-results/diag_overall_direction/A92_20261105，必须completed216/216且无unresolved。
原3 FFHQ00226/00443/00963×四任务×模型坐标sigma=.05/.10，24案例。原A71前九轮/ζD/y/mask/kernel与A92第16步状态/RNG全部SHA绑定复用。
仍原30点15/10/5、DDPM eta1、raw Tweedie后clamp、原IEEE精度；不改时间步、图像、seed或小波db4/level3。
仅在原index15(t333到s309)改变一次修正，后14次更新回到原ZAPS。A92原基线/分带对齐/直接物理方向单步输出直接复用。
不是完整300重建、独立30F算法或A93三步的同作用范围比较。SR仍历史bicubic经验反投影，真实物理数据项用原H的自动微分。
原H单位[-1,1]与noise同单位；补洞仅观测RGB进入新似然，不引入硬clamp观测值。

## 目标与先验含义

固定原q，c=Wq，gamma是与c同形状的每一个小波系数；Delta(gamma)=W^T(c*gamma)。
z为未加物理引导的DDPM下一状态，D_s为原raw mean经clamp的真实下一网络预测。

F(gamma)=||observed[y-H D_s(z+Delta(gamma))]||²/(2 sigma_y²) + a/2 sum_j gamma_j²。

SIGNED_JOINT：gamma可正可负，a=1；NONNEG_JOINT：gamma>=0，a=1；SIGNED_DATA_ONLY：gamma正负自由，a=0。
SCALAR_TIED：所有gamma_j相同，但数据项与完整sum(gamma_j²)正则完全不变，是主目标的嵌套约束对照，隔离逐系数自由度；不是换成另一个标量先验或旧A71规则。
主臂单位方差零均值gamma高斯先验是明确新增的工作模型，不是已估计出的真实不确定性，也不是无超参数理论。
等价地它给原小波修正系数c_j²一个数据依赖的工作位移方差：小c不鼓励用巨大的gamma恢复目标，但不计算g_j/c_j。
它保留原ζD对修正尺度/正则几何的影响；c_j=0处无法产生修正。D本身不解释为真实协方差。
sigma较大时数据项相对先验变弱。没有M*sigma²扣除、global omega、全局标量倍率/范数预算/单位阵中心。
gamma从0起，目标不是“最接近原gamma1”；这不是把原DPS方向最小改动当最优。
此工作MAP不是完整图像风险估计：参数基底与状态已依赖y，不继承SURE或真实后验保证。补洞缺失区仍须靠先验和区域评价。

## 联合求解与数值预算

对数据项f做真实输入VJP，grad_gamma f = c * W grad_x f；grad_x经过完整H和下一去噪器，包含分量交互。
prox更新：gamma_new=(gamma-eta*gradf)/(1+eta*a)；非负臂再投影>=0。正则可分不代表数据项可分。
所有gamma同时更新；不显式构造巨大Jacobian，不将各系数孤立算最优比例。
SCALAR_TIED先将数据梯度正交投影为其全系数均值，再用相同prox、F与回溯；共享倍率的正则是N*a*gamma²/2，N为原系数数，不暗改正则强度来偏袒任一表示。
固定最多2次接受更新，每次最多6个实际前向backtrack；初始eta=1/max(1,maxabs(gradf))，仅数值初始化，失败折半，次次检查数据项majorization与完整F不升。
首个grad复用A92真实未修正状态梯度；第二次grad才新做1F/1VJP。每次试探都是同一s309，非正负随机微扰/额外采样位置。
数值容限为64 float32 eps乘目标尺度；实际值、接受/拒绝和eta全部记录。不从候选中按GT选，不扫描先验强度。
没有在6次试探内得到合格更新：unresolved、保留成本，不生成回退图。精确零prox移动：明确working-precision stationary；2次预算耗尽不称求得全局最优。
以近端梯度与回溯为数值工具，参考Beck/Teboulle2009 §2.3式2.5—2.7、§3 ISTA回溯式3.2—3.3 https://doi.org/10.1137/080716542；网络非凸问题不继承其凸全局收敛结论。本文不启用FISTA动量。

## 预算、门控与时间

每臂最多12试探F+1额外gradientF/VJP+14原续推F=27F/1VJP。24×4臂=最多2592新增诊断F、96新增输入VJP、0优化反向。
每载入模型预热1F/1VJP单列；旧A92已付费前缀/首梯度不计为新成本，但方法自身获取首梯度仍需成本，不能宣称免费。
首缓存梯度的old源/状态/观测/NN checkpoint SHA全固定；原节点NN权重/grad/flags不变；solver不消费RNG；末尾CPU/CUDA RNG与A92原baseline匹配。
solver所有NN调用均s309，之后14次NN严格原grid[16:]；计attempts/完成/VJP/backtrack，不把拒绝探测漏记。
额外求解DWT分析/合成次数与完整H/反投影调用分别记录；原14步的固定小波开销仍计入续推时钟。
服务器强制真实Torch：链式梯度有限差分、跨系数响应、目标单调性、符号约束、RNG与原续推、原custom checkpoint兼容，缺依赖失败不skip。
GPU手工分配>=6GiB并按实际VJP峰值+1GiB准入；不终止进程、不降精度/分辨率。任意数值/身份/计数失败保留停机、不自动失败重试。
完整本轮后缀包含载入缓存状态/y/entry、算子/参数构造、求解、14F续推与CPU同步；求解和续推时间单列但已包含在后缀，不重复相加。
模型载入/预热、离线指标/GT噪声分解与图片写入另列。旧对照时钟不重测，不能给完整300或同成本加速结论。

## 判据、对照与失败分支

逐任务/噪声将主臂与原ZAPS、A92分带单步、A92直接物理单步分别比较，主判断优于直接物理方向才支持联合求解的新增价值。
预注册：meanPSNR>=+.1dB、至少2/3图胜、SSIM不降、LPIPS不升；输出objective下降与最终质量是否一致，signed/nonnegative与有无先验都保留。
主臂相对同目标SCALAR_TIED的差异才用于讨论逐系数自由度的新增作用；固定两次更新不能把目标嵌套关系误称实测必胜或已求到各自最优。
相同A71完整末轮标量质量从已绑定summary.groups直接读取（不依赖旧run有metrics字段），历史描述对比，不伪称同成本。
GT只在所有动作/续推完成后计算PSNR/SSIM/LPIPS、观测/缺失风险、physical_gain=clean_gain+<真实噪声,response>，不进入solver、eta、prior或arm选择。
主臂正向才考虑3步窗口；主臂弱/负向时先看正则是否过强、求解进度与noise拟合诊断，不立即全程推广或按图调参。
两步未收敛的不足与目标不合理区分报告；unresolved率和失败成本完整保留，不只统计成功样本。

## 运行与交付

bash projects/run_a94_20261106.sh --gpus 0,1
/data0/lzy/ZAPS-results/diag_joint_diagonal/A94_20261106
完整summary.json/comparison.csv/solver.json/audit.json、逐图输出；gamma/大PT只留服务器，轻量归档上传SHA与PNG。
results/zaps-joint-diagonal-response-20261106，不合results进master。断线安全，终端短表8行主臂相对直接物理方向，避免长摘要截断。
--status只读；--archive-only只汇总上传；资源暂停仅续未开始部分。Mac无Torch/CUDA，只做NumPy/CLI/语法检查，不冒充模型结果。
