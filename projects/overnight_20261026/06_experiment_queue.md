# 实验队列与交付
- A48：用户报告20/20，195新增诊断；元数据尚待本机取回，rho无动作干预。
- A49：prepared_and_pushed，commit1b6aaf3；0新UNet，服务器命令已交付。
  入口 projects/run_a49_20261026.sh；结果分支results/ffhq-response-reliability-audit-20261026。
- A47：已有高噪声边界脚本，但用户当前优先指标重构，未报告运行，不自动解锁。
- 后续新GPU实验：尚未定稿，不虚构A50成功；先读A49和最新论文关键方法，
  只选一个机制并写独立卡/预算/对照/判断分支，再编写可执行脚本。

每个新GPU包必须交付：一句话可否定假设、四任务范围、唯一变量、源run/快照、
开发/确认名单、传感器/重试/方法NFE、真实Torch无skip准入、单段前台命令、
日志/缓存恢复规则、最终配对结果和图片。用户手工服务器执行，不冒充本机自动算GPU。

第五轮候选A的明确公式及消融见CONTROL_LAWS_DRAFT.md。下一个独立包优先固定网格
全30步的当前得分几何，而非继续t319二次目标/阈值搜索。实现/卡片尚未冻结，
不得运行不存在的A50命令。真实VJP需要保存网络图并计backward/CG/walltime。
若采用IEEE传感且共享一次前向，需相同IEEE生产基线及原native参照，
不能同时声称生产native不变、IEEE免费传感且每步只一次前向。
原模型末次t0->输出无新增网络，原输出语义与gamma作用仍须独立对照。

A50已建立独立planned_preflight卡和current_score_geometry.py数值内核：
四项NumPy代数检查及语法通过；另三项Torch预检编写完成但Mac未执行。
尚无完整真实模型driver/冻结plan，不提供A50假命令。预检通过仅证明代数/数值接口，
不是当前FFHQ真实VJP、未用图、终局收益或整轮可运行认证。

补充：已实现single-forward传感/VJP/真实残差CG/GPU内积约化，以及整图排除的
TIME/CONSTANT/循环SHUFFLE分配。纯NumPy/设计共6项通过；服务器Torch共5项待执行。
完整设计前向4872、反向最多1212已逐调用核对；driver/launcher/归档仍在实现，
不能把内核就绪当完整实验可运行。数值地板/精度/差分门槛已在A50卡明确。

## A50实现就绪，服务器门控与结果待回传
独立入口 `projects/run_a50_20261026.sh` 和卡片已完成。该条更新取代上文“尚无driver”的旧恢复点。
20旧例×8臂×30调用+72实际模型计量门控=4872新增诊断前向，反向最多1212；不新增训练。
每一步用当前raw均值的真实VJP及噪声正则化观测回投确定gamma；固定全部时间点，另列IEEE生产对照。
这与旧相邻余弦、低秩协方差、t319脉冲均不同；已有JAPS投影不能称原创，gamma投影取消当步zeta幅度。
Mac仅6项NumPy和语法检查通过。服务器先执行5项无skip Torch预检，实际网络门控失败即保留并退出。
失败/中断phase不自动重跑；只有完整phase能缓存复用。归档或push失败只重试归档/推送。
尚无实际冻结plan、服务器Torch或画质结果；当前只交可执行下一轮，不声称其有效。

最低代码01522b8已成功推master；服务器一段式命令见RUN_A50_SERVER.md。所有Torch/UNet门控仍未执行。

## 必须采用A50r2：原checkpoint兼容性修正

后续完整历史阅读发现r1新代码错误：模型参数全requires_grad=False不兼容旧Attention的
自定义checkpoint，即使外层use_checkpoint=False。普通线性玩具未覆盖此接口，r1不应继续启动。
保留r1全部文件字节，另建current_score_runtime_r2.py、diag_current_score_geometry_r2.py、
run_a50r2_20261026.sh和独立r2卡/目录。新上下文只取输入导数，保留参数requires_grad供
checkpoint内部使用，不累计parameter.grad或更新权重；新增直接导入原CheckpointFunction回归。
6项Torch及实际FFHQ仍待服务器，不能以静态定位声称数值通过。若r1已启动，实耗保留另计。

A50r2最低代码a479de4；更新命令见RUN_A50R2_SERVER.md。请替代上一条A50启动命令。
