# A101R3：严格300 NFE数值门控与反向桥接

A101R2确认两个问题：独立前缀不满足逐位 `1e-6` 门控；闭式对角前向没有第10轮优化所需的计算图。本修复使用独立结果根。

候选仍独立完成270次前缀前向和9次反向；要求 `tau/initial` 精确一致、独立学习的 `zeta/D` 相对误差不超过 `1e-5`，实际误差写入 `prefix_audit.json`。随后只配对末轮RNG。对角前向严格保持闭式解，反向用 `delta_closed + q - stopgrad(q)` 沿原ZAPS修正传播；第10次更新不影响已经形成的评价输出，但其时间与反向次数计入。

其余任务、噪声、300 NFE、判据及定性协议不变。结果根：`/data0/lzy/ZAPS-results/diag_equal_nfe_feedback/A101R3_20261110`；结果分支：`results/zaps-equal-nfe-feedback-r3-20261110`。
