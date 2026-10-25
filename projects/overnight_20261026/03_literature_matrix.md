# 文献矩阵与分轮核查记录

当前已核对的版本、方法、假设、源码及未完成项以literature_sources.json和本文件后续章节为准。
首表保留最早检索状态，“未读主文”等历史标签不能覆盖后面的实际核查。
本轮另核对原ZAPS v1 §3.1—3.2/Algorithm1、§4.1—4.2及arXiv版本/正式ECCV2024备注：
逐图ζ/D和端到端物理损失已有先行工作，不能作为新增状态控制本身的新颖性。
只保留当前条件几何γ和真实数值误差h两个候选，详见[综合报告](00_integrated_report.md)。
检索时间：2026-10-26；只用论文/正式会议/作者代码等一手来源。
必须核实最新版本、正式接收状态、关键公式/假设、算子域和额外成本，不能只读摘要。

|文献|当前已打开的一手页|与问题的关系|后续必须查|
|--|--|--|--|
|AdaPS, Robust Posterior Diffusion-based Sampling via Adaptive Guidance Scale|https://arxiv.org/html/2511.18471v1|两个似然梯度近似的对齐决定缩放；式16为一维投影系数|有2026 Jacobian-Aware后续/更名线索，需核对最新正式版；DDIM系数不能直接当ZAPS倍率|
|Diffusion State-Guided Projected Gradient, ICLR2025|https://proceedings.iclr.cc/paper_files/paper/2025/hash/4d5b7f7a0603b75e39ffebbf95e3b7ee-Abstract-Conference.html|当前扩散状态的低秩子空间约束引导方向，研究强度/噪声稳健性|下载主文，明确子空间定义、额外NFE/分解开销；不能与当前协方差子空间混同|
|Variational Control for Guidance in Diffusion Models, ICML2025|https://proceedings.mlr.press/v267/pandey25a.html|轨迹匹配和terminal cost框架，适合解释局部J不等同终局目标|主文公式、控制近似、时域/梯度成本，是否五天内可实现最小子模块|
|Gotta Go Fast When Generating Data with Score-Based Models, 2021|https://arxiv.org/abs/2105.14080|嵌入式误差与自适应SDE步长基础|无条件SDE与带物理引导DDPM的适用差别、噪声耦合与拒绝成本|
|SURE Guided Posterior Sampling, arXiv2512.23232|https://arxiv.org/abs/2512.23232|仅搜索找到，未读主文；可能关联无GT风险与轨迹修正|不得先宣称可用；核实噪声估计假设、SURE/GSURE在非满秩任务的限制与开源成本|

继承基础文献也要审：ZAPS/TMPD/CA-DPS/Nyström/EDM/DPM-Solver/AYS/GITS，
标清哪些是标准已有工具，不可当独创。新近论文与我们工作高度重叠应写新颖性风险；
不以“最新”替代质量。优先收敛为不超过2个可落地、可否定的候选主线。

## 首轮版本核查：必须用JAPS正式后续，而非只引用AdaPS旧版

[arXiv当前记录](https://arxiv.org/abs/2511.18471)明确v1为2025-11-23，v2为2026-03-29，
v3为2026-06-21；当前题名JAPS，作者备注Accepted by TMLR2026。
[v3主文](https://arxiv.org/html/2511.18471v3)和[作者代码](https://github.com/liavhen/JAPS)
已打开；[OpenReview论坛](https://openreview.net/forum?id=m63GJnhIN2)当前浏览器验证拦截。
不能把v1的AdaPS式16和v3章节/定理混配引用，后续需按v3重新读推导与实际代码。
已确认它研究direct/proximal似然近似之间的Jacobian关系，并含条件DDIM系数修正；
“当前状态对齐→自适应似然缩放”不是空白，需要明确本项目的新增问题与证据。
当前仅确认版本和框架，不声称已完成v3全主文/实验复核或可直接移植ZAPS。

## 首轮补入：AdamDPS是一项不能忽略的近邻基线

[作者主文](https://www.cs.cornell.edu/gomes/pdf/2026_belardi_arxiv_adam.pdf)，题名
Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling，页首
ICLR2026、arXiv2603.16797v1。已读Algorithm1及式10—11、Fig9与后续loss分析：
其利用引导的一二阶历史矩稳定更新，Fig9相邻余弦是过程诊断，不能把它当因果阈值。
论文同时给出更低终局guidance loss未必更好重建的例子，与我们A48待查错位问题有关。
这支持研究动机，不是当前ZAPS上的有效性证明；原文比较还有开发集调参，不能称无参数。
下一轮需核对官方实现、时间坐标/幅度以及在已学习ζ/D上是否重复归一化。
不能直接将Adam向量预条件搬来并当作本项目的新状态体系。

## 第二轮：JAPS主文—代码—ZAPS兼容性已推进

v3 §3.1—3.4、作者condition_methods.py的clean_likelihood/JacobianAwareBase/JAPS
类与gaussian_diffusion.py的DDIM更新已逐段核对。详见JAPS_ZAPS_COMPATIBILITY.md，
记录原Git blob，未执行作者代码。明确额外真实VJP相对当前stop-gradient ZAPS不是免费，
unit/model域、CG/FFT边界、learned zeta/D和真实Jacobian均不能直接等同。
推导出原eta1 skip条件系数K=(1-a/b)/sqrt(a/b)，并做0网络代数检查；
它只约束跨度和作用单位的定义，不是更优网格、端点半群或算法新颖性证据。

## 第三轮：SURE风险边界与真正的自适应误差已核查

首表中“未读主文”的SGPS/自适应SDE状态由本节和literature_sources.json更新。
详见PROXY_RISK_AND_ERROR_CONTROL.md，逐条列明公式适用范围及本项目推论。
已读SGPS §3.1/Algorithm1/§4.1：3NFE每外步，另有100步条件Langevin；
近似Gaussian、局部强凸和有偏SURE梯度假设不能移作当前ZAPS的保证。
已读Eldar §IV/V秩亏部分、Projected GSURE投影式和Ramani2012 §III：
可观测风险与完整图像质量不同，非满秩任务不能靠测量SURE获得任意全图MSE。
已读GottaGoFast §3.1—3.2：同噪声嵌入对、混合容差RMS、接受/拒绝和两次score成本；
这与随机轨迹余弦不同，也不是可直接粘贴到skip-DDPM的现成定理。
不追加第三条SURE追分路线；用这些边界约束两个候选机制的可否定设计。

## 第四轮：三个近邻的方法、假设及工程代价

详见LITERATURE_MECHANISM_SCREEN.md与literature_sources.json：TMPD §3—4/E.1、
DiffStateGrad §3/C.1—2及官方helper、DTM §3/A.1/B.2/C.2已核对。
它们分别是条件矩似然、当前状态子空间方向约束、带轨迹代价的控制框架，
不能混称“我们的曲率稳定性理论”。DTM实际仍有贪心/Tweedie终局近似与内循环成本；
其v2式30→31的平方三角界缺少一般交叉项条件，本轮给出独立代数反例，
不直接引用为本项目的通用上界。该检查不否定作者已报告的算法实验。
近期四任务大增益不足以填补原创性：标准条件化、投影和误差控制都须明确归属。
# 新增官方实现核查：AdamDPS

已核实最新arXiv v2为2026-04-22、正式ICLR2026 PDF；作者8份相关源码/配置已读，
核心3文件及采样循环blob已存literature_sources。方法/单位/reset/额外反向成本/新颖性
边界见ADAMDPS_COMPATIBILITY.md。矩的平滑操作相对DPS不额外调用UNet，但原DPS真实
输入VJP相对当前ZAPS近似不是免费；作者调参及任务默认噪声也不能省略。
这项是理论与强先行工作的核查，不新增第三条机制或冒称本地复现。

## 自适应 SDE 作者源码与成本接口补核

已读 GottaGoFast 作者 JAX 自适应类/循环和完整简化 Torch 移植，详见
[SDE_SOURCE_ACCOUNTING.md](SDE_SOURCE_ACCOUNTING.md)。一次尝试内共噪声不能扩展成
拒绝后保留 Brownian 路径的保证；简化 Torch 返回的尝试次数与真实 score 调用次数
也有接口差异。JAX 主实现的计数不同，不能混为一谈或由此否定论文实验。
所有候选 B 的拒绝/终端/失败前向都要由独立 hook 计数；当前仍只有合成代数检查。
