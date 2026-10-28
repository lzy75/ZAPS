# A67R1：模型默认梯度标志修复

依据已上传结果da62cd027f2956085baf64ae393aab743fcefa40，两个BASE完成各300前向、
10优化反向后，失败于`Model weights not frozen`检查；没有保存输出/参数快照，
没有诊断回放。原模型默认参数requires_grad=True；score一直在no_grad下执行，
ZAPS优化器只接收ζ/D。要求所有标志本来就是False不符合真实加载接口。

本修复仅在measure执行期间把NN参数requires_grad设False，验证无参数.grad与
参数版本变化，finally恢复原标志；不更改ζ/D训练、倍率、阈值、精度、网格、
样本、观测或评价。旧源码、旧plan与原失败记录保持字节不变。

修复入口只接受上述两个特定失败，保留完整失败目录、日志与状态到repair_r1。
600前向、20优化反向、2warmup计入累计成本；输出未落盘，需重算这两个BASE。
成功的完成记录不重跑，其他错误不自动恢复。R1新增错误也不自动重试。

新增测试覆盖默认requires_grad=True、上下文恢复、异常恢复、无权重更新，以及
TinyDiffusion完整300流程在冻结前后输出一致。恢复测试覆盖失败身份与成本拒绝。
Mac只运行无Torch部分；服务器必须通过新增Torch测试及原A67测试再运行模型。
不增加新的科学方案或实验预算；在原A67预算之外另加上述失败成本及新warmup。

运行`bash projects/run_a67r1_20261028.sh --gpus 0,1`，状态/重传仍支持--status/--archive-only。
