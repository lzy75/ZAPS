# A81R1：官方Motion核的NumPy种子修复

状态planned；2026-11-04。软件初始化修复，沿用[A81完整协议](A81_daps_multiplier_transfer_20261104.md)。

原A81停在模型前CPU门控：其Motion测试两次构造权重不一致。复用的diag_learning_rate_ablation.set_seed只设置Torch；官方Kernel依赖numpy.random。
新wrapper先np.random.seed(kernel_seed)，再调用原make_operator（仍设置Torch并执行原官方构造）。不修改vendor、原A81源、倍率、容差或原采样流程。
真实门控仍要求两次Motion状态torch.equal逐位一致，不改成宽松容差；原采样/gamma1/RNG/调用数/严格权重门控保留。
允许启动条件：旧A81无plan、模型case或模型load记录、旧coordinator已停止且日志确为报告的Motion单失败。旧session/log只读保存。
新根 `/data0/lzy/ZAPS-results/diag_daps_multiplier_transfer/A81R1_20261104`，不复用失败状态、不清日志、不覆盖观测。
固定图、两个任务、种子和官方配置保持A81；原尚未生成正式观测。12次运行/12000NN/240000原生观测梯度，0 UNet VJP/优化反向；候选额外120000 Hq；探针另记。
判据和正负分支完全沿用A81，不按本修复改变成功条件。仅软件检查通过不作为重建收益。
新入口 `bash run_a81r1_offline_20261104.sh --gpus 1`，仍使用校验过的官方离线wheel。
新结果分支 `results/daps-multiplier-seed-r1-20261104`，原失败状态快照纳入轻量归档。
