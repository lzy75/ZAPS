# A54断线恢复：执行协议补丁，不是新算法实验

状态：实现完成，Mac恢复/锁测试4项通过；服务器真实恢复待用户执行。

父实验：A54_current_norm_gain，代码a82968d，服务器已报告6项测试通过及
PREPARE/FREEZE PASS，随后SSH `client_loop: send disconnect: Broken pipe`。
该输出不能判定GPU是否启动；原launcher收到HUP会退出并终止子进程。

目标：排除连接断开造成的重复实验和不透明预算重置；不是重新选择策略、病例或输出。
唯一改动是独立协调器 `utils/a54_session.py`。原A54计划、代码、卡、分配、
源文件SHA绑定和判据全部不改。没有新增科学臂。

执行：`python -u utils/a54_session.py`。客户端跟随日志；真正协调器以
独立session、文件stdout/stderr、DEVNULL stdin启动。客户端退出不发送子进程终止信号。
重连后同命令跟随现有任务，`--status`输出状态和最近25行。两个GPU/单GPU串行规则
仍由原launcher决定，反向代理失效只可能影响最后推送，不应终止GPU计算。

恢复只在原launcher及两个shard文件锁均空闲时执行：

- 已complete案例完全保留，原item门控继续检验；不因画质差重跑。
- stale running或KeyboardInterrupt案例最多每例恢复一次；从相同冻结初始状态重算该例的三个控制臂。
- 原目录原子移动到 `A54_20261026/interrupted_attempts/`，不删字节。
  写前日志 `interruption_recovery.json` 保存原路径、文件hash/bytes、原始run记录和调用计数。
- 明确数值/软件异常、损坏账本、身份不匹配、第二次中断均停止，不自动重试。
- 原summary仍表示有效完成臂的900前向；中断尝试单列于恢复账本，随原递归归档上传。
  合计为有效调用＋中断记录调用，不声称中断成本为零或历史总成本精确可知。

新增控制器没有UNet调用；每个重算例最多90次前向/90次输入反向。
原中断成本另加；原900预算不是所有尝试的总上限。恢复发生过的案例数和消耗最终单列打印。

测试：文件保留和幂等；不重试科学失败/不动完成案例；写前移动日志恢复；真实fcntl互斥。
当前只有基础执行测试通过，不宣称SSH现场已恢复或A54画质实验已完成。

日志：`/data0/lzy/ZAPS-results/diag_norm_gain_value/session.log`，
协调器状态：同目录 `session.json`。这些变化日志放在科学结果根目录外，避免归档期间变更。
