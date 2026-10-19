# A2d执行：一次提交，六例自动完成

完成更新：A2d已回传6/6完成、180诊断NFE，H4分类LOCAL_DIRECTION_REFRESH_NOT_SUPPORTED。
不要再执行下方历史GPU启动命令。当前只做[结果核验及零NFE统一归档](GUIDANCE_DIRECTION_REFRESH_RESULT_20261019.md)。

状态2026-10-19：独立入口已实现，本机数值/判定检查已执行，生产GPU实验未运行。
用户明确恢复Mac主控流程，不等待Windows Codex任务。共享核心及旧冻结工件保持不变。

目标：固定原网格/先验/噪声，比较原引导C、半跨度方向刷新F和第二段幅度匹配但保持原方向的M。
F须同时相对C和M有局部联合收益，且至少一个原定时间层有2/3图双seed满足标准，
才能解锁另卡的终局验证。方向效应、旧余弦控制价值和最终重建收益分开判断。
详见[实验卡](experiment_cards/A2d_direction_refresh_feasibility_20261019.md)。

## 已做的本机计算

新入口11项中9项通过，2项因本机无Torch跳过；A2c兼容检查8项中6项通过、2项跳过。
40组NumPy随机向量验证了共享先验/噪声、幅度匹配、方向位移恒等式；
另检查零刷新、零中点方向、不可分辨信号、三指标取舍、双seed图像簇判定、失败成本和哈希复用。
Python编译、CLI帮助、bash语法及Git空白检查通过。
这些不是GPU结果。服务器启动器要求新入口11项全部通过且无skip，否则不启动实验。

## 在服务器执行一次

```bash
bash <<'BASH'
set -euo pipefail
cd /home/lzy/ZAPS
git fetch origin refs/heads/master:refs/remotes/origin/master
git merge --ff-only origin/master
mkdir -p /data0/lzy/ZAPS-results/diag_direction_refresh
nohup bash projects/run_a2d_20261019.sh >> /data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019.log 2>&1 < /dev/null &
echo "A2D_PID=$!"
echo 'LOG=/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019.log'
BASH
```

沿用已工作的Windows Clash/SSH反向转发；不用再次配置代理或切换Codex会话。
nohup将检查和计算整体放入后台，SSH断开后进程可继续；主机终止/OOM仍会中止。
日志追加，启动器与计算入口各自加锁，重复提交不会再开始同一批计算。
先严格测试、审计A2a/A2c六例工件，再自动串行首例和其余五例；模型只加载一次。

## 查询，不重复提交

```bash
tail -n 8 /data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019.log
```

出现各例CASE和最后A2d Summary后，只读重印汇总：

```bash
cd /home/lzy/ZAPS/projects
python utils/diag_direction_refresh.py --summary-only
```

回贴A2d Summary和Records即可；若报错，回贴日志最后30行，不重复提交失败案例。
该汇总不启动模型。运行中成本为已落盘调用的下界；只有6/6完成才有完整H4结论。
原始POINT、门控、run.json及SHA保存在独立/data0目录，每例另保存实际中点q_h和F/M输入。
后续阶段统一归档轻量文件，张量保留服务器。

计划新增方法0、诊断180 NFE；不重训、不细化正式时间网格，不重新计算A2c。
A2a含恢复1800方法+840诊断，A2c诊断120，新实验若完整完成，该队列为1800方法+1140诊断=2940。
更早单图探索另列；不得把未来在线刷新需要的额外模型调用隐瞒为免费状态读数。
