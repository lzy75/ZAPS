# 实验卡模板：先填写、后运行

状态：planned / running / completed / failed / inconclusive / rejected。
未写明目标、预期与分支处理，不启动GPU实验。每次保存填好的卡片、
resolved_config及结果；不在结果出来后修改原假设/成功门槛。

## 1. 问题与身份

- 实验ID / 日期 / 工作线R-I-A：
- 一句话目标：
- 可被否定的假设：
- 当前证据，为什么现在做：
- 参照run绝对路径 / completed状态 / git revision：
- 既有类似实验及结论：
- 本次新增信息（原样重跑必须说明预注册的seed/样本或接口修复）：
- 允许结论范围：local_reference / mechanism_only / state_gain / paper_spec等：
- 不允许结论（例如原文复现、GT早停收益、跨数据集泛化）：

## 2. 唯一改动与固定条件

- 唯一改动/因子：
- 对照臂与实验臂（因子实验写明交互设计，不伪称单因素）：
- dataset/task/image IDs及图像hash：
- calibration IDs / validation IDs（禁止有交集）：
- checkpoint/hash/model配置：
- H/H^T实现、边界/resize、算子源码hash：
- y张量及hash / seed / 物理值域 / noise sigma或协方差：
- model-state值域与物理域转换：
- 真实时间网格/噪声端点/连续转整数方法：
- sampler、variance、eta、clamp/threshold：
- ζ/D初始化、参数索引/插值、joint学习、Adam lr/betas/eps：
- loss sum/mean、gradient clip、参数投影：
- epochs / final_mode / 是否额外采样：
- float/uint8指标口径、SSIM与LPIPS版本：
- torch/CUDA/GPU可见状态与RNG协议：
- GT只作离线评价/校准；不进入在线调度、停止或输出挑选：是/否。

## 3. 预算与门控

- 每图方法NFE上限：
- 每轮S上下限 / 总epochs / 总NFE约束是否可行：
- warmup/额外UNet/拒绝尝试/最终净化是否全部计数：
- 离线调度搜索成本及摊销：
- 诊断回放NFE（不得当作方法NFE，但不得隐瞒总实验成本）：
- 输出等价、时间映射、参数索引、采样噪声、指纹门控及固定容差：
- 可变S时不同RNG尾状态的预期、CRN/Brownian耦合约定：
- 失败/超界/异常路径停止规则：
- 有限终止的端点约束、最大末跳、budget_exhausted语义：

## 4. 预期结果与事先处理分支

| 结果类别 | 可观测指标/预注册门槛 | 支持/否定什么 | 下一步动作 |
| --- | --- | --- | --- |
| 正向支持 | 填写数值标准，不仅“更好” | | |
| 负向/否定 | | | 停止哪项扫描/保留哪个基线 |
| 不确定/小于噪声 | | | 固定少量seed确认或保持简单基线 |
| 门控失败 | | 不解读为方法效果 | 先修接口/一致性，保留失败记录 |

联合报告：PSNR、SSIM、LPIPS、物理残差、actual NFE、耗时；
状态实验另报t/u/h/S、ζ/D映射、终止理由和异常/饱和率。
通过某一指标不能自动忽略另一个指标退化。

## 5. 运行与结果归档（运行前填命令，运行后只补结果）

- 明确的命令行（显式参数，不能只依赖会变化的默认值）：
- 日志/结果绝对目录（新目录，不覆盖父实验）：
- completed/failed状态及门控报告：
- 配对汇总 / 每图数值 / 原始张量和CSV：
- 结论：假设支持/否定/不确定，而不是只写“PSNR=...”：
- 与预期不符的解释及证据边界：
- 是否解锁下一实验ID；是否同步正式single/batch/其他模型：
- 停止该路线的条件与保留的证据：
