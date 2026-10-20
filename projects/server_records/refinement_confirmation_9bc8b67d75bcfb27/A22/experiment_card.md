# A22：原轨迹五阶段的同预算动作空间检验

状态 planned，2026-10-20。A21已完成：固定、物理、完整状态及GT oracle都选原t0 gamma2，
PSNR增益0.129699 dB，状态增量与oracle余量均0。因此关闭A21的末端选择分支，
不改ridge/指标窗口/终端系数范围。这个负结果不等于其他噪声阶段也没有状态选择空间。

## 本轮问题与区别

先检验更早阶段是否确实存在值得选择的动作，再拟合状态控制器。
沿用A2c/A3预先固定的五个原检查点k8/19/24/27/28（t903/641/319/54/14），
只把当步物理修正改为gamma=.5或2，其余原步骤正常继续，评价最终终点。
A2c只评价下一时刻clean，并未验证最终重建；A4–A6改变步数的细化也不是本干预。
本次不加步、不扫描新时间点/频带或更大的gamma，不把动作空间当指标有效性证据。

## 同NFE与固定条件

同一父PRE、图/测量、原30步/10轮、参数索引、原native精度、ζ/D及共享原suffix噪声。
FFHQ SR4、unit sigma=.05、legacy_bicubic、lr=.01，保留原last_opt约定。
从当前原next_latent出发做 x_next(gamma)=original_next+(gamma−1)q_k，随后29−k个原预测。
每例每个科学候选仍270前缀训练+(k+1)原prefix预测+(29−k)suffix预测=300 NFE。
同预算强对照包括原C、原t0 gamma0/2和全部五阶段固定pulse(.5/2)，共13个固定动作。
不能把其中最佳固定的收益记为状态收益。

## 源与预算

源A21计划绑定的12个completed A2a/A6轨迹，00020–25各seed1000/2000，
00021_s2000采用completed recovery。前3图development、后3图transfer分别报告；均此前已见。
核验A21 plan/run/policy SHA与原负结论、各源PRE/trajectory/last_opt/y/image SHA及config/grid。
每例一次30调用原轨迹重演门控，加2×(21+10+5+2+1)=78次干预suffix，
共108 diagnostic NFE；12例1296 diagnostic，新增method NFE=0，不重训。
未来部署的一次候选动作300 NFE；离线多分支成本1296另记，不隐瞒。
默认GPU0/1各6例，单卡顺序；保留其他任务进程。

## 数值/随机门控

继承A3 replay_noises：原PRE恢复29次原噪声，CPU/CUDA外部RNG保持。
用真实core posterior与显式原噪声重演原30步，各latent/clean/q及final相对误差≤1e−6；
core末步无噪声，不额外采样；不允许额外UNet调用/未计warmup。
原C的PSNR/SSIM/LPIPS重算/重演漂移≤.01/.001/.001；物理loss误差1e−7+2e−5|loss|。
所有候选先固定和保存，再计算GT指标；GT不选择动作、停止或改参数。
保存每点动作前R/qRMS/turn/Omega/u，来自原因果prefix；clip来自30次门控重演的同一步，
因为A6行没有保存clip。两者均不看干预suffix，不增加调用。
任何门控失败停止解释，失败/中断目录保留不覆盖。

## 读数和预先停止条件

两个cohort分别比较全部13个固定动作，平均保护线维持SSIM delta≥−.002、
LPIPS delta≤.005、物理loss非增（10倍数值误差内），PSNR相对C非负。
用development选择最强可接受固定动作，完整报告其transfer表现；另列transfer事后最强固定作强参考。
枚举13^6个GT动作组合，计算相同平均约束下的最大PSNR，仅作选择空间上界。
该oracle不能当线上策略；3图两seed有依赖，不给p值/因果/泛化主张。
跨阶段oracle利用完整离线结果，是宽松上界；因果策略不得用未来阶段状态回选较早动作。
并逐个固定t比较原C/半量/双倍三个动作的oracle余量，区分阶段差异和同阶段样本差异。
oracle相对该cohort最强可接受固定动作余量≥.01dB才记ACTION_SELECTION_OPPORTUNITY_ONLY；
否则为NO_USEFUL_ACTION_SELECTION_HEADROOM。0.01是工程筛选线，不是文献标准。
本轮不拟合新的指标/策略。正向仅解锁“指标能否预测/指导这种选择”的另卡检验；
负向停止本五点、单pulse、gamma范围，不扩大扫描追机会。
原A21末端负结果保留；A22即使有空间也不等于状态收益已建立。

## 执行/归档

`bash run_a22_20261020.sh`，前台一段命令完成检查、准备、双卡、等待、输出和push。
新目录 `/data0/lzy/ZAPS-results/diag_equal_nfe_path_opportunity/A22_20261020`。
原张量不改；候选终点张量留服务器，CSV/JSON轻量归档results/refinement-confirmation-20261020。
Mac快速数学/预算检查，真实Torch及CUDA门控在服务器执行。
