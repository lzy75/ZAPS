# Windows 通道与 A2a/A2b0/A2c 归档核验

日期：2026-10-19，Asia/Shanghai。本次只做现有结果核验、归档和运维准备。
新增方法 NFE=0，新增诊断 NFE=0；未运行训练、原诊断或科学单测，冻结脚本和实验卡未修改。

## 通道与实际工作区

Windows 工作区 `D:\ZAPS`，原 HEAD `fa4c632feedd82e9163fe45aaf3f7340bc6a38e9`，明显早于交接版本。
原有未跟踪 `experiments.zip`、`planning/`、`tmp/` 保留。未执行 pull、merge 或切换当前工作区。

SSH `server` 已实际连接 `lzy@192.168.191.194:9090`，远端主机 `ubuntu`，登录目录 `/home/lzy`。
Windows SSH 配置只有一条 `127.0.0.1:10809` 反向转发；额外命令用 `ClearAllForwardings=yes`。
服务器 Git 代理 `http://127.0.0.1:10809` 有效，已实际读取 GitHub refs 和固定版本文档。

服务器仓库 `/home/lzy/ZAPS`，HEAD `a55954d6b81d517ffc721cc50e8f61dc05bd8615`，
分支 `results/server-records-20261015_163839`。检查时有 4104 个暂存新增文件和一个已跟踪修改；
本次未改原分支、索引或 HEAD，归档程序核对了前后索引 SHA256。
`AGENTS.md`、`projects/AGENTS.md` 在服务器不存在；后者 GitHub 也返回 404。
采用用户提供的 Windows 根规则及服务器 `EXPERIMENT_STANDARD_20261013.md`。

检查时根分区约余 506 MiB、使用率 100%；`/data0` 约余 12 TiB。
两张 RTX 3090 当时均在使用（0 卡约 7419 MiB/68%，1 卡约 13342 MiB/100%）。
这些仅是检查时快照，不表示后续卡片可用。归档下载到 Windows，新流程只在 `/data0/lzy` 创建任务和代码缓存。

## 工件核验

轻量工件位于 `projects/server_records/20261019_windows_handoff/`。
`MANIFEST.json` 对应服务器传输的 149 个文件；其中包含原 A2a 完整轻量归档、三阶段原始文本、冻结来源快照和独立核验记录。
`existing_results.tar.gz` 保留原始传输；`github_context/` 保存固定 GitHub commit 的交接、报告及完整回贴证据。
新增的本机说明与证据对照单独保留，不冒充服务器原件。

| 阶段 | 独立核验 | 历史方法 / 诊断 NFE |
| --- | --- | ---: |
| A2a | 原归档39文件、24个排除二进制的完整哈希；六条有效记录、180步/30探针及生产门控；注册恢复案例；原始中断记录保留 | 1800 / 840 |
| A2b0 | 完整run、30点、来源哈希、四组已记录门控；缓存结果未重算张量或模型 | 0 / 0 |
| A2c | 六个run/run.sha256、冻结plan及全部来源、五点/例、20次调用/例、门控、配置及父案例；纯标量复算60组效应、10层均值与完整summary | 0 / 120 |

源图、权重、张量只读取并核对身份，不加载模型或执行重建。所有冻结来源在采集末尾再次核对 SHA256。
共核对144个来源文件路径；旧A2a归档的41个文件（含manifest和属性文件）还逐字节比对了已上传Git提交的对象。
传输包SHA256为 `177c76280512ea5102e2322059d61a98fc3e85df4fe404cedcb87e5edd1c1207`。
A2a 的 `00021_s2000` 使用注册目录 `recovery_runs/00021_s2000_20261018_190036_133092`，未用原截断记录冒充完成。
历史总账仍为该队列1800方法+960诊断=2760，不称整个课题总成本。

A2c 分类精确复现 `LOCAL_HALF_DOSE_NOT_SUPPORTED`。t903/t641 各3/6局部联合收益、仅1/3图双seed均受益，
不满足至少2/3图的原标准；其余三层无半量联合收益。没有解锁半量完整轨迹或新增扫描。
固定 GitHub commit `639e42f3d653d53405a9957c66f61ef305bdc8cd` 的完整回贴中，
六例全部已报告字段、点、门控、NFE和重算summary与服务器工件精确一致。
这补足了原报告仅核验回贴标量的工件缺口，仍不构成模型输出独立重现或原科学单测认证。

## 归档与发布状态

GitHub `results/guidance-replication-20261018` 已实际核实为
`8f58702867f6227dabad87d3fecff0fc9eb9a08e`，包含较早 A2a 归档，不包含后来的 A2b0/A2c。
本次 A2b0/A2c 及统一归档已实际落盘 Windows 并验证传输哈希；源二进制仍在服务器。
本次归档的 Git 提交/远端发布状态由旁边 `PUBLICATION.json` 记录；没有该记录时仅表示本机归档，不能推定已上传。

## 自动化入口

`scripts/experiment_remote.ps1` 提供 Sync、Check、Submit、Status、Logs、Collect、AuditExisting。
`scripts/server_job.py` 使用固定commit/卡hash/GPU/NFE预算、原子run ID、独立代码快照、单任务锁和后台supervisor。
完整操作、输出契约与限制见 `scripts/EXPERIMENT_REMOTE.md`。
纯CPU运维测试输出保存在归档的 `audit/infrastructure_tests.txt`。
本次没有提交新科学任务；未来新实验需独立卡和明确执行授权。
