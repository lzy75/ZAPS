# A56M1 显式分配与禁止进程中断恢复适配器

状态：implemented；资源执行适配，不是新增科学实验或模型结果。

用户要求先查归属、暂不终止任何任务。快照 GPU1 空闲10785MiB，GPU0空闲6026MiB。
M1原生峰值reserved8216641536字节，加1GiB余量，共8.652GiB；GPU1快照满足，
实际准入仍由原入口实时重新检查。GPU使用率0/2%不作为可终止证据。

唯一执行改动：用户指定一个GPU index/UUID，替换原入口的idle-queue信号交接为
“仍活动则拒绝”。不发送kill/killpg，不暂停/终止旧队列或其他任务。断线安全，重复命令跟随日志。
原A56/M1科学源码、plan、72次profile、固定30点、8例、checkpoint/FP32、1e-5门控均不改。
完整案例和四条已保存基线复用，若服务器状态仍为3/8则剩余2100次原前向；适配器新增NN=0。
原科学失败/partial不自动重试，不清除锁/日志/张量。旧队列活动或资源不足则停止，不能解读为质量失败。

独立执行记录：/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/A56M1_no_interrupt_20261027。
dispatch.json与本适配器/卡/测试快照追加到M1归档；不修改原科学身份。

运行：python -u utils/a56m_no_interrupt_resume.py --gpu GPU-e21d8c3c-cde6-c10f-878d-b20a4ded1742。
状态：同入口 --status。正向仅指原冻结ImageNet8例完成和质量摘要可评价，不自动证明迁移。
Mac仅检查3项资源契约；原真实Torch门控仍在服务器执行，不能skip冒充模型结果。
