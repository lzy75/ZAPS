# A56S2 显存等待与冻结计划续跑

状态：操作恢复待服务器执行；不是新科学实验或模型结果。

## 目标与证据

服务器已通过19项检查并冻结20个案例，随后在 GPU 显存准入失败，尚未启动分片。
本次仅将该已知零调用的资源不足转为等待；不修改任何原科学源码、冻结计划、样本、
种子、算子、30点网格、倍率规则、预算或质量判定，不释放/终止他人的 GPU 进程。

冻结图像：FFHQ 00377/00389/00617/00641/00791/00912；
ImageNet 00067/00151/00193/00195，均 SR/IP。以服务器现有签名计划及哈希为准。
原结果目录：`/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/selection_r2_20261027/A56_selection_r2_20261027`。

## 门控、成本与分支

- 仅接受 A56_SELECTION_R2 已冻结计划、原 session failed、日志含已知12 GiB准入错误，
  且不存在任何 case run.json / attempt_*。先核对原科学源身份，保存失败日志与状态。
- 保留12 GiB空闲门槛，遵守 CUDA_VISIBLE_DEVICES 和显式 ZAPS_A56_GPU0/1。
  无卡达标时30秒检查一次；一张达标时两个分片串行，两张时并行。
- 等待不调用 UNet。总预算仍6000方法 +3624诊断前向，最多1204输入VJP。
- 继续执行原已绑定 launcher，原 prepare 仅复用计划，绝不重选图。
- 准入瞬间的显存竞争仅在仍无任何 case attempt 时允许回到资源等待；
  任何模型/数值/科学/未知失败一律保留并停止，禁止自动重跑失败案例。
- 独立 detached session 与日志支持 SSH 断线后用相同命令重新附着。
  同时持有旧 selection session/client 锁，禁止重复启动。
- 恢复 manifest、资源选择与失败快照另存至计划目录 resource_resume_20261027/，
  绑定原 plan SHA、旧来源与本恢复源 SHA；原 session.json/session.log 不覆盖。
- GPU结果、模型计量与跨图结论仍待真实服务器完成；本机仅验证资源/恢复逻辑。

入口：`python -u utils/a56_resource_wait_session.py`；仅查看用 `--status`。

