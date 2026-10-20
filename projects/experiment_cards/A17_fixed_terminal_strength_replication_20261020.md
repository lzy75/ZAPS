# A17：固定末端系数2的跨队列复核

状态 planned；2026-10-20，A线 local_reference。父源 A16 已完成6例、12 diagnostic NFE，
结论 FIXED_STRENGTH_CANDIDATE_ONLY。A16 同次输出的 F2 比 C 平均 PSNR +0.248058、
SSIM +0.005199、LPIPS +0.004521；A 比 F2 PSNR −0.003490。方向 cap 只激活2例且
未产生增量收益，按 A16 停止条款停止该投影比例/系数区间的状态规则，不扫新阈值。

## 目标与可否定假设

冻结 F2=clean+2q，在另一组三幅历史图00020/00021/00022、两个seed1000/2000上，
实际 PSNR 收益及平均质量保护线仍成立，并优于同成本 F1=clean+q。
这些图及其原重建此前已用于 A2a/A14/A15，不能称未见图像或独立确认。
本轮只增加该队列首次末端 F1/F2 实际输出，复用历史300-NFE终点，不重训、不调度。
不测试更大系数，不复活 A16方向 cap，也不运行十调用有限差分。

## 源与固定协议

源 A16 `/data0/lzy/ZAPS-results/diag_terminal_state_strength/A16_20261020` 六例 SHA绑定。
目标队列从 A14 已冻结计划 `case_sources` 的 development 目录读取，核对原 run SHA、
completed、所有gates、PRE/last_opt/measurement/image SHA。00021_s2000 必须使用该计划
指向的 completed recovery，而不是旧 running 目录。不依赖原 run 有 run.sha256 文件。
其 config/grid 必须等于 A16；checkpoint、unit_operator+legacy_bicubic、sigma=.05、
ζ/D PRE、t=0/index0、Tweedie clamp、uint8 PSNR/SSIM11、Alex LPIPS 均继承。
新 F1/F2 使用同一 IEEE FP32 中心；runtime/policy 与 A16 一致，退出恢复。
原 C 保留 native。GT 用于历史测量生成，本轮只在候选冻结存盘后用于评价。

## 预算与门控

新 method NFE=0；每例原C预测两次作 repeat，共12 diagnostic NFE，科学动作每例+1。
部署总成本300+1，不把缓存训练成本清零。默认双卡各3例，单卡顺序。
调用前记录；禁噪声抽样；clean/q/output repeat 与真实末端F1的相对误差≤1e−7。
参数/网格/RNG/源SHA保持；C指标相对原记录漂移≤.01dB/.001/.001；记录漂移保护线。
F2物理损失和仿射模型核对容差1e−7+2e−5|loss|；失败/中断原样保留，不自动覆盖。

## 事先判据与处理

目标队列 F2 相对 C 平均 PSNR 超过数值保护线，SSIM delta≥−.002、LPIPS delta≤+.005，
平均物理损失非增（数值保护线内）；至少2/3图像簇的seed平均PSNR正向。
F2 比同成本 F1 的平均 PSNR 也须超过保护线且≥2/3图像簇正向，质量差仍在同一保护线内。
数值保护线继承 A16 limits：10×max(repeat/C漂移，PSNR1e−6，SSIM/LPIPS1e−7，loss1e−10)。
完整报告每图每seed、各队列均值及合并12例均值；主结论只按目标队列判据，
不能用合并均值掩盖目标队列失败。原A16队列为选择集，合并结果仅描述。
每例LPIPS>.005仍明确报告，不将平均保护线改写成逐例全通过。

正向：仅解锁固定动作在新的独立图像名单上的确认，不宣称状态收益。
负向：记录固定F2不可迁移/质量代价，不扩系数，不追逐均值换名单。
不确定/门控失败：保留结果与成本，只修明确接口/数值问题。

## 执行

`bash run_a17_20261020.sh`：一次同步后的命令完成测试、准备、双卡、等待、汇总和push。
新目录 `/data0/lzy/ZAPS-results/diag_fixed_terminal_replication/A17_20261020`。
原张量留服务器，轻量记录归档 results/refinement-confirmation-20261020。
本机仅公式/判定/语法检查；真实 Torch/CUDA 门控由服务器执行。
