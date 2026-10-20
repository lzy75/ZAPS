# A2d：方向刷新独立作用的最小验证卡

日期2026-10-19，A线；状态planned/implemented，尚未运行生产实验。
设计详见[指标和决策方案](../STATE_MINIMUM_DECISION_PLAN_20261019.md)。
用户已明确不再等待Windows任务，恢复Mac主控、服务器执行；本入口不重跑任何旧实验。

## 问题、父源与假设

检验同一当前t/D下半跨度刷新得到的引导方向，是否比原方向及匹配幅度原方向有局部作用。
A2c已拒绝简单半量剂量H3；本轮不是继续扫描gamma，也不延长其未通过的后续轨迹。
唯一新构造为二阶段引导及其幅度匹配对照，不改变正式噪声时间网格。

H4：至少一个原定t903/641/319/54/14层，在至少2/3图的两个seed中，
刷新臂F相对原臂C和幅度匹配臂M均取得联合收益（画质三项不退化、至少一项改善、loss不升）。
主阈值沿用A2c的10倍重复/漂移误差及最小绝对地板，不按结果调整。
只允许mechanism_only/local_reference结论；不声称余弦控制、最终重建、更多步收益或泛化。

父源`/data0/lzy/ZAPS-results/A2a_20261018`，原别名
`/home/lzy/ZAPS/projects/results/diag_guidance_replication/A2a_20261018`；
六条必须按恢复注册表确认为complete，完整冻结文件/SHA/权重/观测/案例hash核验后复用。
A2c完整结果另核对`/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019`；
代码基准639e42f的A2c回贴证据已复算，不以此替代服务器工件核验。
新入口实施提交、完整resolved_config和源hash须在运行前绑定新plan。

## 因子与固定条件

图像00020/21/22、采样seed1000/2000、同图共用测量seed1020/21/22；均开发数据，无独立验证声明。
配置和权重继承原plan：FFHQ SR4、unit sigma_y=.05、legacy_bicubic经验T、
原clamp、Karras rho3、30×10、lr=.01、eta1、PRE joint zeta/D、last_opt。
H/T、边界、值域、Adam/loss/clip/投影、参数索引、float/uint8/SSIM/alex LPIPS均锁定父源。
不重新优化、不重建参数、不抽新噪声，固定原先验及同一次注噪，外部CPU/CUDA RNG保持。

在当前输入x、当前t和参数i，q0为缓存原修正：

    x_h = x + .5*q0
    q_h = q(x_h,t,i)
    a0 = RMS(q0), a_h = RMS(q_h)
    C = original_next
    F = original_next - .5*q0 + .5*q_h
    M = original_next - .5*q0 + .5*(a_h/a0)*q0

F/M第二段幅度相同、方向不同。名义剂量系数相同不表示实际向量和/能量与C相同。
所有臂在原下一时间s及原下一参数索引评价，t14臂在s0评价clean，不称最终输出。
B为q0/q_h单位向量弦长，chi=B/(.5*a0)，不是相邻qCos或微分曲率。
原父源已通过非零q门控；若缓存q0出现近零则身份/可读性前提矛盾，停止并保留故障，不替换点。
q_h为零时方向NA，仍保留F/M端点和全部成本。
GT只在分支构造及调用结束后评价全部结果，不参与选择或停止。

## 预算、门控和输出

每点原下一状态重复2次、当前t中点重复2次、F/M下一状态各1次，共6。
每例30诊断，六例180诊断；首例30，其余150。新增方法0，历史成本另列。
所有实际调用包括失败计数，无额外warmup、训练、拒绝重算或最终采样。
若未来在线采用，中点一次评估属方法成本，不能继续声称原300 NFE。

必须先验证冻结来源、原下一状态clean/q/loss/指标及当前参数索引；
沿用A2c原门槛：clean/q相对误差<=1e-6，重复clean<=1e-7，
loss差<=1e-7+2e-5*abs(L)，PSNR/SSIM/LPIPS漂移<=.01/.001/.001。
中点重复q相对误差<=1e-6；B/chi信号不得小于10倍重复弦长地板，NA不填造。
两臂输入位移实际表示误差<=1%；预期位移<=1e-6*a0而不分辨时保留为不确定，不放大差商。
幅度匹配及RMS(F-M)=.5*a_h*B做数值恒等式检查；q_h零方向不计算弦长恒等式，直接核对F=M。
恒等式不是H4的证据。原q_h=q0时F/M退化为C，先做合成数值检查。
用于方向收益判定的F/M预期位移差还需超过10倍两输入舍入误差和幅度单位地板；
弦长可分辨但实际端点方向干预不可分辨时，不提升为候选，保留全部原读数。
参数/网格/RNG和源SHA不变，逐次调用时间点/上下文匹配，每例实际30次。

全部输出含t/u/原跨度、clip、R、旧qCos/Omega、a0/a_h、B/chi、
F-C/M-C/F-M三组loss和PSNR/SSIM/LPIPS差值、实际NFE、耗时、异常、status和门控。
不新增窗口/尺度或复杂预测器，不用恒等式相关性宣称指标有效。

## 结果分支与停止

- H4支持：只解锁另卡的固定候选后续轨迹验证，不同步正式算法或调度。
- F只胜C不胜M：方向独立作用未建立；F只胜M不胜C：原方法保留。
- H4不支持/混合/不可分辨：停止该半跨度刷新候选，不改阈值/窗口/跨度追结果。
- 技术门控失败：停止运行，保留失败工件和实际成本，修具体问题，不自动重训。

新结果目录拟为`/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019`，与父源完全分离。
独立脚本`utils/diag_direction_refresh.py`与测试`utils/test_diag_direction_refresh.py`已实现。
运行入口：在服务器仓库执行`bash projects/run_a2d_20261019.sh`，严格单测无skip后inspect，
自动串行运行首例及其余五例；只加载一次模型，科学负结果不删除后续案例。
新plan绑定来源、当前脚本、运行入口、卡及依赖SHA；不覆盖已有失败/运行中案例。
每例保存branches.pt的中点q_h和F/M实际输入，SHA绑定run.json；复用时核对，
后续无需为了获得分支状态重新做同一实验。张量留/data0，不作为GitHub轻量归档。
启动器和Python分别持有运行锁，防止断线后重复提交；nohup日志追加，已完成案例校验复用。
运行中每快照保存调用计数；异常终止未落盘的调用可能未知，summary明确标记成本为下界，
不把running当complete或自动重跑。失败成本保留。
只读状态：`python utils/diag_direction_refresh.py --summary-only`，不会加载模型或启动实验。
生产门控与GPU结果尚未取得，本机检查与服务器结果分开报告。
原A2a/恢复/A2b0/A2c的脚本、卡与工件均不修改。
