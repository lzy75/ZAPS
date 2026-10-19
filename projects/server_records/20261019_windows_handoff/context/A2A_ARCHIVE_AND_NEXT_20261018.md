# A2a完成后的零NFE归档与分析

六条生产结果已回传；不要再次prepare、repair或remaining。下一步仅做完整记录核验与一次性轻量归档。
[阶段结论](GUIDANCE_REPLICATION_RESULT_20261018.md)。

新入口utils/archive_guidance_replication_stage.py不加载Torch/模型，不执行优化或采样。
它使用原冻结身份检查和恢复注册表，核对六条有效run及其工件hash、原中断组件，
重算六条科学摘要及所有尝试成本，并要求与现有recovery_summary一致。
不以H1正结果作为归档过滤条件；负/不确定分类也完整保留。

同步本次归档代码后，在服务器projects目录运行：

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_archive_guidance_replication_stage utils.test_archive_state_indicator_stage -v
```

应9项通过、无skip。随后先inspect（0 UNet，仅读取）：

```bash
python -u utils/archive_guidance_replication_stage.py --plan-dir results/diag_guidance_replication/A2a_20261018 --inspect
```

应AUDIT PASS，completed=6，method/diagnostic=1800/840。
它同时打印原先未回贴的40组同t关联及六条分段方向统计，请将这两项回贴。

通过后统一归档一次，并创建独立本地结果分支：

```bash
python -u utils/archive_guidance_replication_stage.py --plan-dir results/diag_guidance_replication/A2a_20261018 --results-branch results/guidance-replication-20261018
```

输出projects/server_records/20261018_guidance_replication_<timestamp>，包含：

- source/：原六条目录的JSON/CSV/实验卡/PNG、原中断JSON和空tmp、恢复run/注册表及原摘要，逐字节复制。
- sidecars/：现有迁移manifest与两个阶段日志。
- derived/：从六条有效run重算的完整成本摘要、180步轨迹表、原40组同t关联CSV、完整探针JSON和分段方向中位数。
- manifest.json：源路径、文件SHA256/大小、全部排除二进制的完整SHA256、统计和成本范围。
- .gitattributes：保持CRLF等原始字节；写入Git对象后再次核验。

原中断run不被改成complete；科学分析用注册恢复run，成本同时计原中断尝试。
原summary可能仍是中断时状态，source/保留原件，derived/recomputed_recovery_summary.json是显式有效摘要。
所有.pt/权重等留服务器；Git中只有清单和轻量文件，不构成二进制备份。
脚本不切当前分支、不动用户Git索引、不自动push。若导出途中源变化，则拒绝发布归档。

服务器网络恢复后，只推本阶段结果分支一次：

```bash
git -C /home/lzy/ZAPS push origin results/guidance-replication-20261018
```

然后告知结果commit，Mac统一fetch审计。当前连接故障未解决时，不反复重试GitHub，
先保留本地结果分支与/data0原始张量。导出文件、源码实现与生产结果归档须分开说明。
这一步不承诺修复SSH/VPN，也不宣称GitHub已经收到服务器结果。
