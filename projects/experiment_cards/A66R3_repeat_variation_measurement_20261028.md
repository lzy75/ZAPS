# A66R3：保留重复差异，完成质量与完整计时测量

## 实际证据与问题

结果提交0e0760ed55cab416fa735fdbce4f3eff2a74a69f：LEGACY_IEEE图00000
repeat1完成300F/10参数反向后，输出与repeat0相对差.004733731，超过1e-6。
不能称它逐像素可重复，也不能把差异直接当画质退化或仅为舍入误差。
已有独立30步首5图完成，无回退，平均PSNR29.36249，对相同5图IEEE基线
均值+1.20380dB；这是未完成全队列的开发信号，不是最终方法收益结论。

历史映射TransposeMode使用CUDA F.interpolate(bicubic)参与参数优化。
PyTorch2.6源码明确其CUDA反向使用atomicAdd而非确定性。
https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/aten/src/ATen/native/cuda/UpSampleBicubic2d.cu
仅设置cudnn.deterministic不保证该算子确定性；其是否足以解释全部差异尚需诊断。

## 新协议（明确区别于通过旧检查）

不改重建计算、算子、反向、精度、seed、噪声或任何参数；不放宽旧1e-6阈值。
每次完整执行均保存输出/指标/完整计时及repeat_gate(passed,tolerance,error)。
当repeat_gate=false时继续收集其他图像和重复，而不是丢弃输出并终止整队。
不挑最好输出；主描述仍用预先指定repeat0，同时逐图报告所有重复指标范围/标准差。
保留旧失败状态及300F费用，重新执行该丢失输出的计时项属于新测量协议，
不把旧run改成complete。最终总报告明确旧一致性未通过，不输出已确认方法收益。
300F新方法的参数前缀门控仍false，其版本继续暂停，不借此解除它的检查。

## 准入与成本

每个GPU worker在加载模型前，用固定CPU生成的probe检测CUDA bicubic前向/梯度
8次，并用torch.use_deterministic_algorithms(True)检查是否触发已知非确定性；
全部为算子调用，不调用UNet，不用GT；保存探针时间/梯度差异及原生确定性错误。
恢复原torch确定性模式后才进入未改变的模型执行。算子8次偶然一致不证明全流程确定。
真正的OOM/非有限值/计数错误/权重修改/噪声配对错误仍立即失败。
只允许隔离已报告的LEGACY_IEEE/00000/repeat1/300F/10参数反向的这一次重复差异。
其余未知失败不自动重试。原R1/R2失败及费用全部保留。

父A66目录不变，旧源码和plan SHA不变，R3有独立manifest绑定新增入口。
复用20条A66完成项及10条A65基线。剩余90次完整运行=20250F，最多750输入反向、
650参数反向；资源恢复热身单列。失败重复300F额外计入总成本，不作为完成项复用。
服务器必须运行真实Torch检查；Mac只验证恢复与记录逻辑。
