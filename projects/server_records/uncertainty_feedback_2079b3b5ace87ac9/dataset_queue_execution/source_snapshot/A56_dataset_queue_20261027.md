# A56 冻结案例按数据集分队列（运行恢复，非新方法）

状态：实现/本机纯调度检查；真实模型与峰值仍待服务器执行。

## 目标与唯一改动

原A56S2两个分片均包含FFHQ与ImageNet，统一要求12GiB，致使较轻的FFHQ也等待。
改为 dataset × 原全局case序号奇偶，共四个不相交工作组：FFHQ各6例，ImageNet各4例。
每张可用GPU最多一个进程、最多两张卡并行；组内仍逐张batch1，单例六臂仍顺序执行。
不拆臂并行、不重选图、不改变模型精度、反向、算子、网格、RNG、原始计划和判定。

FFHQ资源准入改6GiB，ImageNet保留12GiB。6GiB是基于旧A60相同FFHQ图路径
约2.45GB allocated峰值、既有6GiB调度惯例的工程门槛；不是新图/训练峰值保证。
本次额外记录baseline峰值和旧sampling各臂峰值。显存不够仍可能失败；不通过OOM重跑追分。
ImageNet单例的计算图不会因案例分组而变小，仍需要一张足够显存的卡。

## 身份、预算与门控

沿用 `/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/selection_r2_20261027/A56_selection_r2_20261027/plan.json`，
签名/科学源/样本身份全查，不修改原计划及所有旧SHA绑定源。20例合计仍9624次前向、最多1204输入VJP。
分组仅改变设备/运行顺序，不改变任何单例算法。原模型加载/真实runtime/测量/CG/VJP/输出等价门控全部保留。

CUDA_VISIBLE_DEVICES 为硬边界；如提供 ZAPS_A56_GPU0/1，只在所指定的可见物理卡中调度。
同卡别名去重，最多一工作进程/卡。高门槛组优先派往大空闲卡，另一张卡可先跑FFHQ。
组开始前再次检查门槛；仅尚无任何本组attempt的准入竞争可回队列，科学/模型失败不自动重跑。

若上一资源等待器仍是waiting_for_gpu，且公共launcher锁可独占、无任何case attempt，
验证该用户/proc命令/独立进程组后只停止该等待进程组，保存其状态/日志，然后接管。
若原任务已进入launcher或NN，则不停止、不并发写目录，提示继续原任务。

新 coordinator 持有公共launcher/旧session/旧分片锁；新四组各自加锁，无重复case写入。
只复用通过原 finished/baseline_done 验证的完整案例；残缺/失败attempt停止。
所有20例完整才执行原汇总、图片和Git归档；恢复源/分组/资源选择/峰值及旧等待器快照一起归档。
不拿部分图像GT结果调整队列或参数。

入口 `python -u utils/a56_dataset_queue_session.py`；独立detached session，断线同入口重连。
日志在 selection_r2_20261027/dataset_queue_20261027/；执行证据在计划目录 dataset_queue_execution/。
若两卡都不足12GiB，FFHQ可先完成，ImageNet继续等待，不能承诺整体加速比例。

