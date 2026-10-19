# Windows 到学校服务器的执行与收集

2026-10-19。入口为 `scripts/experiment_remote.ps1`，需要 PowerShell 7.4+、OpenSSH 和本机 Python。
SSH 使用现有 `server` 别名及 `ClearAllForwardings=yes`，借用已有反向代理；保持 Clash 和提供转发的 SSH 会话。
脚本不会切换原服务器工作区、使用其索引或自动合并分支。所有新任务与代码缓存写入 `/data0/lzy`。

## 已完成结果

本次 A2a/A2b0/A2c 的归档及核验报告在 `projects/WINDOWS_CHANNEL_ARCHIVE_AUDIT_20261019.md`。
首次只读归档可以执行：

```powershell
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action AuditExisting -OutputDirectory D:\ZAPS\tmp\new_archive_directory
```

输出目录必须不存在。归档只读取文本、原有图表和身份信息，对留在服务器的张量/权重分块计算 SHA256。
下载成功后，自动验证 tar 内路径、完整文件清单、大小及 SHA256，再提取到 `verified`。
失败的 `.partial` 文件保留，既不覆盖结果，也不启动任何实验。
原始来源目录较深，Windows 检出含完整归档的 Git 分支时需使用 `git -c core.longpaths=true ...`。
本机 Python 收集器已实际提取这些路径；直接读取 Git 对象使用 `git cat-file`，避免 `git show` 的 Windows 路径长度检查。

## 新实验的前提

本次未提交任何新科学实验。未来入口必须与独立实验卡一起审批、提交并发布到 GitHub，记录完整 40 位 commit。
实验卡写明可否定假设、固定参数、数据/seed、比较标准、失败处理和方法/诊断 NFE 预算。
A2a、恢复、A2b0、A2c 的原入口和卡已明确禁止通过本包装器再次提交。

新 Python 入口在 `projects/utils/`，工作目录为隔离快照的 `projects`。
入口必须遵守 `zaps-job-env-v1`：

- 从 `ZAPS_RESULTS_DIR` 取得输出目录，只往该目录写科学结果；从 `ZAPS_COMMIT` 取得来源版本。
- 用 `ZAPS_MAX_METHOD_NFE`、`ZAPS_MAX_DIAGNOSTIC_NFE` 限定计数，在每次模型调用前检查剩余预算，失败调用同样计数。
- `CUDA_VISIBLE_DEVICES` 已限定物理 GPU，入口内部使用 `cuda:0`。
- 保存完整 run、指标、成本和门控；退出前保存 `budget.json`，包含整数 `method_nfe`、`diagnostic_nfe`。
- GT 仅离线评价，不选择在线分支、输出或 epoch。

包装器会强制墙钟超时，并在结束时核对预算账本。通用包装器无法仅靠环境变量拦截任意 Python 代码的模型调用，
因此实际新入口的计数 hook 和在线预算检查必须先审阅；预算契约不是对任意旧脚本的认证。
大工件通过既有路径/符号链接复用，隔离快照不改变这些共享数据；实验入口应将其视为只读。

## 提交、查询、收集

先把以下 JSON 模板保存为独立 spec，替换占位值；模板本身不可执行：

```json
{
  "run_id": "NEW_APPROVED_RUN_ID",
  "commit": "REPLACE_WITH_FULL_40_HEX_COMMIT",
  "gpu": 0,
  "card": "experiment_cards/NEW_APPROVED_CARD.md",
  "entrypoint": "utils/NEW_APPROVED_ENTRY.py",
  "arguments": [],
  "method_nfe": 0,
  "diagnostic_nfe": 0,
  "timeout_seconds": 3600,
  "nfe_contract": "zaps-job-env-v1"
}
```

同步和检查不会执行实验；Sync 只向 `/data0/lzy/ZAPS-code.git` 拉取明确 commit。
首次代码缓存借用原仓库对象，不修改原仓库；缓存依赖这些对象，原仓库维护/清理前先确认没有活跃任务。

```powershell
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action Sync -Commit FULL_40_HEX_COMMIT
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action Check -Spec D:\ZAPS\tmp\new_experiment.json
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action Submit -Spec D:\ZAPS\tmp\new_experiment.json -Execute
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action Status -RunId NEW_APPROVED_RUN_ID
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action Logs -RunId NEW_APPROVED_RUN_ID
pwsh -File D:\ZAPS\scripts\experiment_remote.ps1 -Action Collect -RunId NEW_APPROVED_RUN_ID -OutputDirectory D:\ZAPS\tmp\new_result_directory
```

任务目录为 `/data0/lzy/ZAPS-jobs/<run_id>`，包含 spec、包装器源码、状态、日志、隔离代码快照和 results。
提交时检查 GPU 空闲，后台 supervisor 再检查一次；全局非阻塞任务锁保证本入口同时只运行一个任务。
此锁不能约束其他用户或独立进程，实际运行前仍需与学校 GPU 使用安排一致。
监督进程脱离 SSH 会话，断线不重启；同一 ID 由原子 mkdir 占用，成功、失败、中断均不自动重试。
连接中断后先查相同 ID。`absent` 才表示未占用；`orphaned`/`incomplete_submission` 需人工核对，不能换 ID 自动重跑。

Status/Logs 不加载模型，日志只读末尾最多 32 KiB/80 行。
Collect 接受已结束/失败/失联状态，输出轻量归档和二进制哈希清单；失联任务仍需核对残留进程，不能据归档存在宣称完成。
源张量仍保留服务器，清单不构成张量备份。
包装器负责后台执行，本次未设置定时分析、自动科研决策或聊天唤醒。

## 已验证范围

使用临时 Git 仓库、模拟 `nvidia-smi` 和纯 CPU 假任务验证了独立代码同步、后台脱离、完成状态、防重 ID、单任务锁、
超时失败、预算账本超限、忙卡拒绝、路径穿越/冻结 A2 拒绝和收集哈希。测试不导入模型，新增模型 NFE=0。
Windows 的 AuditExisting 已端到端收集并校验全部现有结果，Status 已实际返回不存在任务的 `absent`。
科学入口、真实 GPU 新任务及其在线 NFE hook 尚未验证，须随新的实验卡单独完成。
