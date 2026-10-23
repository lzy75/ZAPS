# A35b：已完成动作的可选择空间审计

状态planned；2026-10-23；A线，零UNet。A35用户回传12/12 complete、1296诊断NFE、
NO_CONFIRMED_GEOMETRY_INCREMENT。CURVE预测PSNR收益的MAE比STAGE大65.6%/70.4%；
状态动作审计在t319/54/14均选择增强，没有形成图像差异化动作。
尚不能仅据此区分：指标无信息，还是当前动作集合几乎没有可选择空间。

父run：/data0/lzy/ZAPS-results/diag_trajectory_geometry_value/A35_20261021；
原六张开发图00020–25，每图seed1000/2000，原网格和I_cont2基线。无新采样、无调参。
唯一新增是对已冻结的keep/x.5/x1.5终局结果做后验机会分析，不更改A35指标/门槛。

每个固定位置分别计算：各动作平均效果和保护线；全体图上回看最优固定动作；
每图两个seed共用一个动作的GT oracle；允许每seed独立选动作的更宽松oracle；
整图留出的纯时间选择（用其余五图同位置平均PSNR收益选动作）。
报告oracle减去回看最优固定动作的PSNR空间、动作分布及逐图结果。
GT oracle只作离线上界，永远不作实际方法；oracle不施加SSIM/LPIPS选择约束，
仅对选后结果报告用户均值保护线（SSIM降幅≤.002，LPIPS升幅≤.005）。
若同位置所有图的最优动作相同，该集合内的个性化PSNR收益为零，说明下一步应重审
动作范围/应用场景，不能继续通过增加状态特征声称能取得额外PSNR收益。
若空间非零但状态方法无法利用，才支持进一步定位状态指标或映射的问题。
不设新显著性门槛；这是看到A35后的描述性诊断，不是事先确认实验。

预算：方法0、诊断0。无需加载Torch张量/模型；验证原完成门控、文件SHA、
108调用记录及摘要对账。原tensor仍在data0，轻量A35/A35b共同归档，不覆盖源。
输出：/data0/lzy/ZAPS-results/diag_trajectory_weight_opportunity/A35b_20261023。
固定时间点的平均代表多个独立单脉冲实验，不能当作同轨迹五次调权的收益。
小样本开发图、有限动作及噪声场景的上界，不外推状态感知总体不可行。

运行：python -u utils/audit_trajectory_weight_opportunity.py --archive
