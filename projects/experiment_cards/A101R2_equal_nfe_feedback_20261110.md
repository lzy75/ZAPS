# A101R2：严格300 NFE调度修复

A101R1因wrapper递归在0次方法前向处失败；其模型、算子和方法均未执行。A101R2只修复调度：在monkeypatch前保存原始 `measure` 引用，并由单测验证不会递归。科学协议、末轮RNG配对、真实伴随修复、360次完整重建和全部预注册判据均沿用A101R1。使用独立结果根与结果分支，不覆盖A101/A101R1。

结果根：`/data0/lzy/ZAPS-results/diag_equal_nfe_feedback/A101R2_20261110`。

结果分支：`results/zaps-equal-nfe-feedback-r2-20261110`。
