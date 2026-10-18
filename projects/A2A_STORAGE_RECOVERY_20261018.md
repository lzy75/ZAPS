# A2a磁盘故障：迁移与恢复命令

服务器回传：根分区容量耗尽（可用189M），inode充足；/data0约12T可用。
三条已记录complete：00020_s1000、00020_s2000、00021_s1000，文件hash待inspect核验。
第四条00021_s2000训练300及诊断120次调用已记下，五点已保存，但最终probe门控未落盘；
trajectory.pt截断为41,947,136字节。该条尚未完成，不能改status冒充完成。

恢复采用独立入口，不改变被原plan.json绑定的旧脚本和旧实验卡。
[恢复附卡](experiment_cards/A2a_storage_recovery_20261018.md)登记组件复用、原阈值和额外预算。
当前仅实现/本机检查，未代替服务器执行迁移或GPU恢复。本机12项检查中11项通过，
1项CPU张量检查因缺Torch待服务器；服务器应12项全过且无skip。

## 1. 同步和测试

```bash
cd /home/lzy/ZAPS
git pull --ff-only origin master
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_guidance_replication_storage_recovery -v
```

## 2. 迁移这一个A2a目录（0 UNet NFE）

目标为/data0/lzy/ZAPS-results/A2a_20261018。脚本复制全部文件（含原故障证据），
逐文件SHA核对，检查源没有变化；切换原路径为软链接并再次核对后，才移除已验证的
根分区重复副本，释放约411M。原结果全部保存在/data0，原绝对路径仍能访问。
目标已存在或权限不足则停止，不覆盖、不使用sudo。无需迁移整个仓库、模型或其他历史run。

```bash
python -u utils/migrate_guidance_replication_storage.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --destination /data0/lzy/ZAPS-results/A2a_20261018

df -h . results/diag_guidance_replication/A2a_20261018
```

若已经迁移到相同目标，脚本只确认现有链接，不重复复制。

## 3. 检查前三条及第四条可复用组件（0 UNet NFE）

```bash
python -u utils/recover_guidance_replication_storage.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --inspect
```

须出现INSPECT PASS：原计划及输入/源码hash、前三条完整工件hash、第四条观测hash，
PRE/output的完整ZIP CRC与CPU张量shape/dtype/finite/grid检查通过。
这不把第四条缺失的最终门控视为通过；只确认已完成训练组件可进入独立回放验证。
原第四条组件在中断前未另存hash，恢复入口记录本次读取的hash，不声称有丢失前逐字节证明。

## 4. 恢复第四条（新增0方法＋120诊断NFE）

```bash
set -o pipefail
python -u utils/recover_guidance_replication_storage.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --repair-case 00021_s2000 --device cuda \
  | tee /data0/lzy/ZAPS-results/A2a_repair_20261018.log
```

使用原观测、PRE参数/RNG和已保存last_opt，不训练。重做90次门控回放与30次原五点探针，
保存到recovery_runs/新时间戳。旧第四条及截断文件保留；只有所有原阈值门控通过，才登记
recovery_registry.json。原计划、阈值和完整六条样本保持不变。

## 5. 补最后两条（新增600方法＋240诊断NFE）

第四条恢复为complete、所有门控通过后：

```bash
python -u utils/recover_guidance_replication_storage.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --remaining --device cuda \
  | tee /data0/lzy/ZAPS-results/A2a_remaining_after_repair_20261018.log
```

后续统一使用新恢复入口，它识别注册表；旧入口仍按原约定拒绝原running条目。
最后两条直接调用原冻结run_case。前三条和恢复结果只核验复用；失败不自动重试。
新增检查要求结果盘至少2GiB余量，所有本轮新输出/日志均在/data0。

## 成本和回传

当前已经消耗1200方法＋480诊断=1680。修复120、最后两条840，顺利完成后全阶段
1800方法＋840诊断=2640 NFE，比原2520增加120诊断；旧第四条的120不作废或隐藏。
新recovery_summary.json同时保留原始中断成本及恢复尝试，六条有效数据只各计一次科学样本。
任何失败尝试继续单列，不将磁盘故障归为指标或控制失败。

回贴MIGRATED、INSPECT PASS、修复后的CASE/Gates和最终A2a Recovery Summary及六条五点表。
同t关联仍在新汇总的same_t_descriptive_associations字段。阶段结束后统一归档原run、
中断run、恢复注册表、恢复run与成本；不删除故障记录，不直接改原status。
