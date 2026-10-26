# A55r2：修复精确SR伴随被全局反向保护误拦

状态：实现完成，真实Torch/服务器门控待执行。
这是A55执行修复，不是新的科学假设或阈值变更。原A55全部代码、卡、计划、失败记录字节保留。
用户报告错误 `RuntimeError: A55 controls forbid any input/parameter backward`；
来源定位到 `modules/degradations.py::_LinearMapAdjoint.forward`。

## 原因与唯一变更

超分H^T原本使用线性Resizer对局部零probe的VJP实现精确伴随。
原A55把所有torch.autograd.grad都禁掉，将这个合法算子工作误判为去噪器反向。
改为只允许原 `_LinearMapAdjoint.forward` 精确源码调用点、leaf probe、create_graph=False的梯度。
其他grad调用和去噪器input_vjp仍禁止。以直接函数patch保留真实调用栈，不用Mock插入额外栈帧。

算子VJP归为operator_grad_attempts/completed单列，而不是声称完全没有自动微分。
去噪器/模型参数反向仍为0；算子/伴随/时间成本继续计数。生产的目标、方向、CG、精度、
RNG、30步位置、长度约定、样本、GT限制和原A55成功/失败判据均不变。
参照[A55卡](A55_current_direction_value_20261026.md)。

## 恢复与预算

父目录 `/data0/lzy/ZAPS-results/diag_current_direction_value/A55_20261026`。
仅在其launcher及两个shard锁均空闲时检查，保证不读取仍变化的run。
原complete控制案例通过原item身份/数值/输出SHA后复用，不因画质选择是否复用。
失败或stale-running记录全部保留，明确其他异常则停止，不拿这次修复绕过数值失败。
旧失败只允许已识别的全局grad保护报错；旧成本及原始记录快照加入新归档。

新目录 `/data0/lzy/ZAPS-results/diag_current_direction_value_r2/A55r2_20261026`。
剩余新前向上限480，按旧complete控制案例减去已完成调用；新UNet输入/参数反向0。
原A55的已记录前向尝试另加，不把失败消耗抹为0。源A52四条V、A53/A54十条U继续缓存。
原summary的臂/工程门槛不改，不新增调参或采样位置。

## 必须通过的新增实际算子检查

真实 `SuperResolutionOperator` 的H^T在新guard内可运行，与guard外逐元素一致，
并满足内积伴随检验；独立的非算子input gradient仍被拒绝。
这些Torch检查在服务器执行，缺Torch不跳过。
本机只做NumPy/导入/语法；不冒充真实算子或GPU通过。

入口 `python -u utils/a55r2_session.py`，脱离SSH计算、前台跟随日志。
先跑9项必要测试，再冻结新plan并继续，最终保存同样的10张对比图与多数据集可用清单。
结果分支 `results/ffhq-current-direction-value-r2-20261026`。
若依然失败保留记录，不自动重跑。没有新画质结论，失败是执行保护错误而非方向方法被否定。
