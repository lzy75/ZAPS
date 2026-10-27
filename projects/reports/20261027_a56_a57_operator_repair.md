# A56/A57 算子接口故障修复

用户回传明确栈：A57 sample→A56 make_operator→case['mask_seed'] KeyError。
A57沿用旧A55r2病例schema，不能使用要求新mask_seed字段的工厂。SR本来不使用mask_seed。
该栈定位A57的共享工厂，不足以证明A56独立失败原因也相同。

新增独立修复适配层：A57 IP读取NORM_ONLY归档中逐字节哈希校验的mask.pt，不重新生成掩码、不读取GT像素构造算子；SR不要求掩码种子。
A56新IP保持原显式mask_seed规则；新样本、原plan及所有旧科学源文件不变。
新session/日志独立，旧失败状态和日志保留；共享旧launcher/shard互斥。
只对明确mask_seed KeyError且记录前向/输入反向均0的失败attempt作可审计保留后恢复；非零或未知失败停止并打印原日志，不重置预算。
已完成案例按原门控复用，全部完成则直接展示旧摘要，不覆盖结果。
修复源和原计划SHA绑定在各原根的operator_repair_20261027/manifest.json，并进入轻量归档。

Mac：5项回归通过（旧SR缺字段、新IP缺种子拒绝、非零/未知失败拒绝、保留字节/idempotency、A57共享工厂实际适配/恢复），AST/shell/导入通过。
Mac无Torch/CUDA：两项真实Torch算子/原mask一致性回归尚未执行；服务器入口强制执行，无skip。
保持A56/A57原预算和终局判据；本修复本身没有任何画质、等价或迁移成功结论。

修复入口：
- python -u utils/a57_operator_repair_session.py
- python -u utils/a56_operator_repair_session.py

先A57，再A56。A56若原日志对应另一种错误，入口会打印其完整尾部并停止；回传该输出后另行定位，不伪称已解决未知故障。
