# A13：固定尺度下的数值精度对照，不更改敏感度门槛

planned，2026-10-20，独立量测排查。父实验A11/A12各六例完成、中心/原输入/信号可分辨，
但联合usable均0；t14仅q方向六例尺度一致，正交均不一致；t54两方向均不一致。
因此是当前有限差分量测未建立，不能写成方向不重要。原分类/门槛不变，不扩扰动尺度或换方向追结果。

## 假设及选例

H13：保持相同x、q、已保存正交方向、eta=.01/.02、噪声时间与PRE参数，
仅关闭TF32精度策略，就能改善双尺度clean向量的一致性。
固定00024_s1000，在t14、t54均中心clip=0；按已见裁剪状态选择，是事后量测排查，不是盲测或科学泛化样本。
两层并行，是两个同一病例的量测任务，不是两个独立图像。
父目录/data0/lzy/ZAPS-results/diag_clean_field_response/A11_t14_20261020与A12_t54_20261020，
A6源目录、模型、观测和PRE由这两个plan继承；代码基于8a602b6，全部实际源/新代码身份绑定SHA。

## 两臂、固定条件和预算

N=current native：当前进程默认后端策略，完整记录Torch/CUDA/cuDNN/GPU和精度标志；不得猜测A11/A12原策略。
I=IEEE：只设置对应的matmul/cuDNN卷积FP32精度策略为ieee；不改dtype、不用FP64、不改deterministic或benchmark。
Torch支持新精度API时只用新API，否则只用旧API，禁止混用。退出上下文精确恢复原策略。
API依据[PyTorch官方CUDA说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)。
没有kernel级追踪，不把允许TF32等同于已证明某kernel用了TF32。
每臂沿用A11/A12的2中心重复+2方向×2尺度×2符号=10NFE；两层最大40诊断/0方法NFE。
若进入IEEE上下文后策略与native完全相同，复用N结果，避免同策略重复；此层仅10次，并明确ALREADY_IEEE。
不warmup、不训练、不采样suffix、不载GT原图、不改变原正式网格或输出。
原SR4 unit sigma=.05、legacy_bicubic、clamp、D/ζ、模型权重、t/index不变，仍local_reference量测。
原始正交方向直接从A11/A12 fields.pt读取，禁止再抽一个方向。扰动输入构造与原函数相同。

## 门控及预声明读数

源A11/A12完整SHA/门控/汇总复算；源fields/PRE/trajectory/观测核对。只在准备时审计公共历史一次。
N中心对原中心clean/q相对误差≤1e-6；每臂两中心clean/q重复≤1e-7，参数、RNG不变。
I中心与N差异完整报告，这是干预效应，不要求I与旧缓存逐位一致。
N相对旧保存四个J向量的最大相对RMS差≤.05，才记native响应复现；不满足单列，不放宽或重跑。
每臂沿用原assess函数（向量相对差≤.20、输入表示/信号分辨、同向），不能改成仅比较gain大小宣布通过。
原输入/向量任意门控失败或身份不兼容则停止；失败尝试调用前计数保存，保留日志不覆盖。

- 策略未改变：NO_POLICY_CHANGE_ALREADY_IEEE，不能把相同计算声称精度修复。
- N响应未复现：NATIVE_RESPONSE_NOT_REPRODUCED，精度原因未被隔离，所有新读数保留。
- N不可用、I两方向可用、正交向量spread至少减少50%：IEEE_POLICY_IMPROVES_METROLOGY_THIS_CASE。
- I仍不可用：IEEE_FP32_NOT_SUFFICIENT；不转而增加eta直到通过。
- 其余：OTHER_METROLOGY_OUTCOME，保留具体门控读数。

通过只支持这两个固定状态上的量测改善，不支持方向控制、精确导数、曲率、重建质量或步数收益。
若不支持，不继续自动精度/尺度搜索；返回量测方法取舍。工作包仍45/100。
本轮最大完成后队列方法3600/诊断2044=5644；A11/A12后已有5604，早期单图另列。

## 执行

独立目录/data0/lzy/ZAPS-results/diag_clean_response_precision/A13_20261020；t14/t54各一worker、独立状态/张量/日志。
命令bash run_a13_20261020.sh，默认GPU0/1；仅一张可见GPU则顺序。启动前每选中GPU至少6GiB空闲，不干预其他作业。
CPU单测检查策略恢复（含异常）、判定和调用预算；真实Torch检查交服务器，不能skip。
全部任务结束后统一摘要、串行轻量归档和推送；失败/上传失败不自动重跑GPU。
