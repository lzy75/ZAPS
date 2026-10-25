# 文献矩阵：首轮线索，非完成综述
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

## 01:08版本核查：必须用JAPS正式后续，而非只引用AdaPS旧版

[arXiv当前记录](https://arxiv.org/abs/2511.18471)明确v1为2025-11-23，v2为2026-03-29，
v3为2026-06-21；当前题名JAPS，作者备注Accepted by TMLR2026。
[v3主文](https://arxiv.org/html/2511.18471v3)和[作者代码](https://github.com/liavhen/JAPS)
已打开；[OpenReview论坛](https://openreview.net/forum?id=m63GJnhIN2)当前浏览器验证拦截。
不能把v1的AdaPS式16和v3章节/定理混配引用，后续需按v3重新读推导与实际代码。
已确认它研究direct/proximal似然近似之间的Jacobian关系，并含条件DDIM系数修正；
“当前状态对齐→自适应似然缩放”不是空白，需要明确本项目的新增问题与证据。
当前仅确认版本和框架，不声称已完成v3全主文/实验复核或可直接移植ZAPS。

## 01:08补入：AdamDPS是一项不能忽略的近邻基线

[作者主文](https://www.cs.cornell.edu/gomes/pdf/2026_belardi_arxiv_adam.pdf)，题名
Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling，页首
ICLR2026、arXiv2603.16797v1。已读Algorithm1及式10—11、Fig9与后续loss分析：
其利用引导的一二阶历史矩稳定更新，Fig9相邻余弦是过程诊断，不能把它当因果阈值。
论文同时给出更低终局guidance loss未必更好重建的例子，与我们A48待查错位问题有关。
这支持研究动机，不是当前ZAPS上的有效性证明；原文比较还有开发集调参，不能称无参数。
下一轮需核对官方实现、时间坐标/幅度以及在已学习ζ/D上是否重复归一化。
不能直接将Adam向量预条件搬来并当作本项目的新状态体系。
