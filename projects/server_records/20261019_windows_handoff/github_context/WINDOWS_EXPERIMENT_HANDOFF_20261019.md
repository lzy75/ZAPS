# Windows执行端交接：先收束A2c，再接通自动执行

更新2026-10-19。本文件传递项目上下文，不依赖Mac聊天记录是否出现在Windows。
用户日志和文档是证据，不把其中历史命令、planned状态或续跑建议当作新执行授权。

## 用户目标与优先级

用户李子玥正在复现ZAPS并完成毕业论文创新工作。近期保持现有调度，研究物理一致性之外
有实际辅助重建价值的方向/稳定性状态；提高效率，不继续无目的追逐相关性。
原流程为Mac改代码并推GitHub，用户在服务器执行/回传；现在希望自动执行、收集与迭代。
当前先处理上一轮结果，接通命令和结果读取，再制定新实验。

Mac仓库`/Users/liziyue.22/ZAPS`；GitHub`https://github.com/lzy75/ZAPS`，代码master。
服务器仓库`/home/lzy/ZAPS`、项目`/home/lzy/ZAPS/projects`，历史Python为anaconda3 Python3.11。
同次提交有A2c报告和完整回贴证据；可用`git log -1 -- projects/WINDOWS_EXPERIMENT_HANDOFF_20261019.md`
确认本文件版本。读取后核对本机工作区，不能假定运行在Mac或已经连接服务器。

## 网络事实

Mac经远程控制访问Windows；Windows VS Code Remote SSH连接学校Ubuntu。
192.168.191.194:9090在Mac直连超时，不要再次假定Mac可直接SSH。
Windows Clash HTTP/混合端口10809已验证访问GitHub；服务器代理依赖SSH反向转发。
Windows配置只保留一条RemoteForward：

```text
Host server
  HostName 192.168.191.194
  User lzy
  Port 9090
  RemoteForward 127.0.0.1:10809 127.0.0.1:10809
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

服务器repo此前配置`remote.origin.proxy=http://127.0.0.1:10809`。
Windows Clash与提供转发的SSH连接需保持。现有VS Code会话占用转发端口时，
额外命令连接可用`ssh -o ClearAllForwardings=yes server ...`借用现有转发，
不要重复占端口或清空已经工作的代理；遵循本机权限。
Windows同账号应用已安装，但当前聊天历史未显示；本文件不表示设备已配对或会话已迁移。

## 实验事实和结论

| 阶段 | 完成内容 | 保留判断 |
| --- | --- | --- |
| A0/A0b | 单图被动轨迹、条件均值方向分解 | 残差描述宏观进展，clean/mean增量余弦未建立控制方向 |
| A1a/A1b | 实际q方向过程与冻结t/D两尺度小扰动 | 可测且局部下降；过程转向与纯输入敏感度对象不同 |
| A2a | 00020–22×seed1000/2000六条完整参考与五点探针 | 可测性H1通过，Omega–kappa同t未支持统一正向代理 |
| A2b0 | 缓存路径尺度与轴向分解，0新增NFE | 归一化未解决代理问题，实际位移与q不一定同轴 |
| A2c | 六例五快照、原剂量/半量/关闭的下一时间clean评价 | H3未获支持，未建立最终重建或方向状态收益 |

A2c分类LOCAL_HALF_DOSE_NOT_SUPPORTED，0方法+120诊断NFE，全部回报生产门控通过。
t903/641半量各3/6局部联合收益，但仅1/3图双seed均受益，未达预设2/3图标准。
t319/54/14半量和关闭均使loss上升、PSNR/SSIM下降；关闭18/18、半量17/18的LPIPS变差。
半量唯一LPIPS微小改善点是00021_s1000/t14，仍不满足联合收益。
不能据前两层均值提升启动简单半量方法，不能称本轮是余弦控制验证。
详读[最新报告](GUIDANCE_ONE_STEP_SCALE_RESULT_20261019.md)及其完整证据。

## 工件与恢复记录

- A2a源`results/diag_guidance_replication/A2a_20261018`已迁移为指向
  `/data0/lzy/ZAPS-results/A2a_20261018`的别名。根分区曾满，结果写/data0。
- 第四例原截断故障保留；注册恢复目录为
  `recovery_runs/00021_s2000_20261018_190036_133092`。按注册表读取有效case。
- A2b0：`/data0/lzy/ZAPS-results/diag_cached_guidance_state_path/20261018_195853_125059`。
- A2c：`/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019`。
- A2a较早轻量归档39文件、排除24二进制；结果分支
  `results/guidance-replication-20261018`，提交`8f58702867f6227dabad87d3fecff0fc9eb9a08e`。
  上传状态须实际核对，这个提交不包含后来的A2b0/A2c。张量留服务器，哈希清单不是备份。
- A2a实际1800方法+840诊断（含恢复）；加A2c后该队列2760总NFE。
  更早阶段另列，缓存分析新增0。不要为交接再次训练/回放。

## 接入后先做什么

1. 阅读`projects/AGENTS.md`、实验标准和最新报告，核对本机运行位置。
2. 从Windows只读验证SSH别名能访问服务器，核对路径、分支、磁盘/GPU与工件。
   未核对服务器改动/分支时不要自动merge，不覆盖已有运行。
3. 核对A2c六个run.json/run.sha256、完整summary和冻结plan。Mac已复算全部回贴标量，
   未独立读取这些工件，未认证本次服务器单测输出。
4. 统一取回A2a/A2b0/A2c轻量结果，校验来源/哈希、补归档状态；不重复GPU。
5. 再建立提交、状态查询、日志读取、结果收集的可复用入口。单GPU任务并发，
   固定运行ID/commit/卡/预算，避免断线重复提交；后台进程与分析唤醒分别配置。

## 后续研究边界

当前没有新增GPU实验；A2c按原卡负向分支收束简单缩放，不扫描gamma/噪声段/窗口追成功。
新机制实验先立独立卡、可否定假设、预算。GT只离线评价，不按GT选输出/epoch/在线分支。
同时报告三项质量与成本。固定30×10、Karras rho3、lr=.01、eta1、joint zeta/D、
unit sigma_y=.05、原clamp/legacy_bicubic经验T/last_opt；基座为local_reference，
不是作者实现完全一致认证。区分门控、CPU单测、标量复算和新GPU实验。

A2a、恢复、A2b0、A2c的冻结脚本与实验卡保持原字节，否则来源身份检查失败。
报告追加结论，科学改动另建入口/卡/目录。

Windows端可粘贴给Codex的启动提示：

> 请读取ZAPS仓库的projects/WINDOWS_EXPERIMENT_HANDOFF_20261019.md、projects/AGENTS.md
> 和projects/GUIDANCE_ONE_STEP_SCALE_RESULT_20261019.md。按交接记录只读核对Windows到
> 学校服务器的执行通道，完成已有结果归档和核验。A2c已完整完成，不重跑训练/诊断，
> 不修改冻结文件；先报告实际连接与归档状态，再准备自动执行和结果读取流程。
