# A80：保留宿主正则目标的DiffPIR倍率适配

状态planned；2026-11-03；A模块适用边界机制验证。

## 1. 目标与父证据
可否定假设：A79R1直接投影的退化与丢失原数据子问题正则权衡有关；在倍率求解中保留该项，比旧投影改善，且可能恢复相对基线的收益。
父run `/data0/lzy/ZAPS-results/diag_diffpir_multiplier_transfer/A79R1_20261103` 必须完整48/48；结果分支 `results/diffpir-numeric-transfer-r1-20261103`，提交b70561849d4a1a39504f4daca6490771f573c563。
A79R1四格平均均退化，稳定式与原式平均质量差异<.001dB；本次固定稳定式，不重做数值因子。
已有12条STABLE_SHRINK轨迹末20点平均gamma约.45—.60、omega约.09—.15；个别gamma>8。只能说明投影与噪声收缩有竞争，不预先认定是退化原因。

## 2. 唯一变量与公式
固定线性圆周H、干净预测m、原生近端校正q=z-m，r=y-Hm，d=Hq；M为真实观测元素数。
omega=max(0,1-M*sigma_y²/||r||²)，沿用A71、不改变噪声系数。
旧目标：min_gamma>=0 .5||gamma*d-omega*r||²。
新目标：min_gamma>=0 .5||gamma*d-omega*r||²+.5*rho*gamma²||q||²。
新解 gamma=omega*positive(<r,d>)/(||d||²+rho||q||²)。rho沿用原lambda*sigma_y²/sigma_diff(t)²，不增加系数或上限。
原近端方程给出<r,d>=||d||²+rho||q||²，因此精确求解时gamma=omega；运行时用完整式并记录恒等式误差、gamma-omega。
来源：固定DiffPIR官方数据子问题，https://arxiv.org/html/2305.08995 §3.2；新噪声校正目标是本次待验证适配，不是原文保证。
适配后属于残差噪声感知阻尼；没有新增去噪器响应预测或后验不确定性估计。

## 3. 固定条件与对照
FFHQ 00226/00443/00963；ImageNet 00067/00151/00193，各Gaussian61/std3和冻结Motion61；同A79R1的图像/核/y/checkpoint全部hash绑定。
直接复制原y，不重新生成；物理[0,1]sigma=.05（模型域.10），圆周边界。
100点原quad网格，eta0/U1/learned_range/clip_denoisedTrue，lambda/zeta原文表3同A79R1，不扫描。
DIFFPIR_REFERENCE：新运行稳定式原基线；HOST_AWARE：同场运行新倍率。按图交替臂顺序。
CACHED_OLD_PROJECTION：复用父STABLE_SHRINK完整质量，不再付NN；其时间明确标历史，不能与新臂算同场加速比。
原采样For AST、重加噪与RNG、最终x_0输出及最后一次未替换输出的NN调用不变。GT仅离线评价。
原A71—A79R1所有SHA绑定文件不修改，结果独立目录。

## 4. 预算与门控
6图×2任务×2新臂×100=2400新前向；0输入VJP/0优化反向；每次模型加载1前向探针另计。
每run99次原生解析数据解；候选额外198次H、99次q范数；无下一步探测。继承A79R1成本4800F分列，不混入本轮新成本。
纯测试核对凸二次目标最优解、kappa0恢复旧式、近端恒等式、父run完整性与新计划。
服务器强制Torch门控：gamma1与原稳定式输出逐位一致、相同init/末尾RNG/网格/计数、绑定异常恢复、无反向；真实近端轨迹恒等式数值核对。
重用相同y/核下已通过的A79R1稳定解计量，不重新豁免原float32残差。
每个新基线与父STABLE_DIFFPIR核对：相对输出L2<=1e-3、PSNR差<=.02dB、SSIM/LPIPS差<=.001。
重复门控失败保留结果并将对应收益判断置为不确定，不按输出筛除案例、不自动重跑。
完整clock包含加载y/核、初始化、FFT/NN/采样/CPU同步和周期进度；模型加载/探针/指标/父结果核对/写盘单列。没有省NFE主张。
指定GPU初始>=6GiB，加载后真实峰值+1GiB准入；资源不足在attempt前暂停；运行错误保留不自动重试。

## 5. 判据与后续
对每任务/数据集，平均PSNR>=+.1dB、>=2/3图PSNR提升、SSIM不降且LPIPS不升，且所有基线重复门控通过，才列正向开发候选。
分别比较HOST_AWARE对新基线、对旧投影；只优于旧投影而仍弱于基线，结论为减少损害，不称有效迁移。
若仍退化，停止该宿主同类系数扫描，保留DiffPIR原生校正。若正向，下一阶段再考虑噪声边界/固定倍率控制；本轮不同时扩展DAPS。
范围为两先验、两个线性去模糊任务的小型旧图开发对照，不外推非线性/相关噪声/MCMC或实时匹配必要性。

## 6. 运行与产物
`bash /home/lzy/ZAPS/projects/run_a80_20261103.sh --gpus 1`
新根 `/data0/lzy/ZAPS-results/diag_diffpir_host_aware/A80_20261103`。
断线安全并跟随到摘要，输出逐图PNG/PT、倍率/正则/恒等式trace、重复门控、配对质量/clock/cost与CSV。
轻量上传分支 `results/diffpir-host-aware-transfer-20261103`，不合入master、不上传PT。
