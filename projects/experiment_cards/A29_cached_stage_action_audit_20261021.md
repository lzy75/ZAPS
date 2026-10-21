# A29：A28分阶段信息与动作价值的零调用审计

2026-10-21，工作线A，状态planned；新增method/diagnostic NFE均0。
这是看过A28摘要后制定的探索性缓存分析，不修改A28门槛，不称独立确认。

父run为/data0/lzy/ZAPS-results/diag_state_action_response/A28_20261021。
A28完整12案例/72状态/2736调用；付费条件均值响应通过局部风险/物理响应预测门槛，
两个PSNR目标仍弱；step2仅1/12局部探针可用，导致LOCAL主分析没有执行。

## 问题

1. 可靠的噪声阶段内，历史方向、局部方向、局部幅度分别有无信息？
2. 付费RESPONSE的增量来自哪个通道？原始loss与误差的跨阶段尺度是否主导整体MAE？
3. 从条件均值残差选择倍率，是否比阶段固定倍率有下一端点PSNR收益？
   若只是更好预测物理量，不能据此直接进入全程控制。

## 固定数据、定义与分析

- 复用原六图00020–00025、两个seed、六个原状态；没有新图、新噪声或新网络调用。
- 原plan/summary/逐点run及case gates的SHA、完成状态、调用数、72点完整身份核对。
  Mac可核对轻量记录与服务器SHA清单，未重新载入原服务器二进制或执行CUDA。
- 每个阶段单独整图留出，两个seed一起留出，固定ridge1；变换只在训练图拟合。
- STAGE、PHYSICAL、HISTORY复用A28；新增DIR_ONLY、MAG_ONLY、LOCAL、
  RESPONSE_ERROR_ONLY、RESPONSE_PHYS_ONLY、RESPONSE。无参数扫描。
- PHYSICAL/HISTORY/RESPONSE在每层全部12状态比较；方向模型只能在共同可靠样本比较，
  至少8/12、六图都有记录且每图留出后训练充分，否则明确UNRESOLVED，不插值为零。
- 跨层只汇总各层相对其匹配基线的MAE比值，等权阶段，不混用原始不同尺度loss。
  LOCAL可分析的五层范围单列；不能据此把A28全六层门槛改成通过。
- 所有四个目标全部保留，PSNR是相同终点noise的float clean读数，仍非最终重建。

## 倍率选择审计

在原跨度和另外两跨度内分别比较，绝不跨不同终点noise按PSNR选步长。

- G1/G2：每状态固定倍率1/2。
- STAGE_PHYS：仅用留出图之外的训练图，按本阶段候选mean loss均值选固定倍率，不用GT。
- STATE_PHYS：当前状态按三个候选mean loss选倍率，不用GT。
- STAGE_QUALITY：仅用训练图的离线PSNR选择本阶段固定倍率；属于开发GT校准对照。
- ORACLE：使用当前状态已知PSNR择优，仅作动作库上界，不能作为可部署策略。

只统计局部选择的动作分布、与固定/阶段选择的差异、物理loss及PSNR配对变化。
oracle相对“同阶段全部图已知PSNR挑最优固定倍率”的差是本局部库的事后上界；
不称最终画质空间，不把oracle上界当状态控制实际收益。

## 分支

- 如果只有物理响应预测改善，局部PSNR选择没有收益：不直接扩展为全程画质控制，
  应先明确物理约束和画质风险之间的关系。
- 若方向信息在部分可靠层有用：只解锁该范围候选，未来控制需明确不可测区域回退，
  新图重新冻结验证；不能复活全程余弦有效的主张。
- 若局部动作oracle仍几乎没有个体余量：不增加选择器复杂度。
- 任何看起来正向的结论只称探索性候选，部署额外探针必须计方法成本。

命令入口python -u utils/diag_cached_stage_action_audit.py；默认父run为服务器A28，
输出为/data0/lzy/ZAPS-results/diag_state_action_response/A29_20261021。
原A28不覆盖、不改写；本轮不自动启动GPU。
