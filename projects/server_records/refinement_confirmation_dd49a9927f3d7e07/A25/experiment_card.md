# A25：相同网格族内的图像—状态时钟匹配诊断

2026-10-21，planned；五轮上限中的第3轮，不重置计数。
替代原队列尚未执行的方向消融/迁移计划；A23/A24原卡、源码和结果保持冻结。

## 问题、已有证据和可否定假设

A24六例均完成且回放门控通过。RC相对C PSNR−.015428 dB；RC相对幅度匹配R
仅+.007655、1/3图像正向；SLOT_REPLAY相对RC −.021793。因此不再把RC−R
作为方向有效证据，不推进原槽位参数恢复。R_MATCHED/FIXED_GRID/SLOT均未胜C。
唯一达到预定“机制线索”条件的是RC−共享中位固定网格：+.012815、2/3图像正向。
这也可能只是中位网格聚合的损失，不能说明必须匹配当前图像状态。

本轮只问：在保持RC已产生的完整网格分布时，网格与本图对应是否比错配更好？
若自有时钟不优于其他图像时钟，则该弱线索不支持个体状态匹配。
不调rho/窗口/指标符号，不修改引导倍率，不做新图扩样或全10轮重训。
允许结论仅为回顾性机制诊断；不能认证新状态策略、曲率、效率或泛化。

## 唯一因子与固定条件

六例仍00020–22，各seed1000/2000。按image_id排序，同seed的三图形成环。
每例新增两条回放：DONOR_PLUS使用下一图的源RC完整30点网格；DONOR_MINUS
使用上一图的源RC网格。两个置换均无自身、同seed、保持整套源网格的多重集。
不读质量选供体，不择优保留置换或终点；两种完整映射分别报告并固定等权平均。
源SELF=源A23/A24的RC，原C及共享FIXED_GRID直接复用，均不重新运行UNet。
源时钟的LOIO参考彼此重叠，可能包含目标图原无干预轨迹；这是已见开发数据内的
错配诊断，既不是独立测试也不是跨图泛化。参考不含GT质量标签。

每次使用目标图自己的PRE/init、y、ζ/D与原29个slot draw；只借供体整数t序列，
不借供体潜变量、参数、图像或噪声。目标ζ/D仍按实际t的原noise-anchor插值。
不改变field、clamp、DDPM eta1、terminal或参数训练。30点/999→0/严格递减，
均须落在原A23位置单元。Common random numbers按slot配对，不声称同一Brownian路径。
FFHQ SR4 unit sigma_y=.05、legacy_bicubic local_reference、30×10/lr.01/last_opt、
native精度和原uint8/SSIM/AlexLPIPS口径全部继承A23/A24指纹；不认证原文完全复现。

## 预算、身份与门控

父目录/data0/lzy/ZAPS-results/diag_state_feedback_diagnosis/A24_20261020。
必须六例completed、全部门控、run/plan/summary SHA及summary重算通过，且
RC−FIXED_GRID达到A24事先的机制线索条件，才解锁360次诊断。
6例×2供体×30=360 diagnostic NFE，method新增0，全部hook计数。原270前缀复用。
已冻结的外部供体网格可作为300-NFE固定配方，但这不是目标图的在线自适应方法；
获取供体时钟的历史费用已计队列，不能称离线预处理免费。先采本图RC再回放本图
时钟需330次，不能把SELF条件回放包装成新300-NFE方法。

原噪声回放使用既有0-UNet恢复/门控；本轮不再支付已通过A24的SELF数值回放。
准备时绑定全部A24完整run/outputs SHA和A23源身份；工作前后检查目标来源、供体run、
代码、参数、native精度与外部RNG。每臂30次、精确按冻结供体时钟执行；GT在所有输出
保存后才载入评价，不进入供体映射。失败成本和工件保留，不覆盖/自动重训。

## 预定读数和处理

完整报告SELF、两供体、供体等权平均、原C、A24 FIXED_GRID的loss/PSNR/SSIM/LPIPS，
每图先聚合两个相关seed；不把两个供体或两个seed计作独立图像。
供体均值是两种配方的期望指标，不混合像素、不选择最好供体，不是新部署输出。
另报每例与SELF不同的位置数及平均|delta t|，避免把相同网格解释为敏感性发现。

SELF−两供体均值PSNR≥.01 dB、至少2/3图正向，且SELF相对每个完整置换均值均>0，
仅标CLOCK_MATCHING_HINT_WITHOUT_NET_GAIN；阈值是工程线索，非统计显著性。
其余完整结果标CURRENT_STATE_CLOCK_MATCHING_NOT_SUPPORTED；不得以其中最差置换
构造正结论。SELF对C净收益单列，A24的负结果不能被对供体的相对优势抵消。
用户视觉容差SSIM下降≤.002、LPIPS升高≤.005与物理非增保护线继续完整报告。

无匹配线索：关闭当前R/Omega-reference/单元位置规则，不再扫描rho/窗口/权重追分。
有匹配线索但SELF仍不胜C：只保留有限匹配信息的诊断结论，不能扩大为成功方法或
自动开启迁移；另有机制依据才考虑剩余最多2轮。五轮是上限，可以提前给负向结论。
门控失败：只修技术原因，不解读重建效果。

## 执行与归档

入口utils/diag_state_clock_matching.py；bash run_a25_20261020.sh。
新根/data0/lzy/ZAPS-results/diag_state_clock_matching/A25_20261020；双卡各3例，
同卡时串行shard，等待后一次输出汇总与失败日志，轻量归档push、张量留服务器。
Mac做置换/成本/判定单测，Torch与实际核心/noise预检交服务器，无skip才启动。
本轮完整后队列7200 method+5408 diagnostic=12608，早于队列的历史费用另列。
