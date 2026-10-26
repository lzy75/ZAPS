# A58：两类blur的等价目标结构预条件准入

状态：planned，入口已实现；Mac设计/导入/语法检查，服务器Torch/GPU尚未执行。先0NN数值计量；必须准入后才触发质量阶段。
固定原30点、观测、初始化与创新、sigma=.05、checkpoint和输出口径；不调步长、正则或旧CG迭代上限。

## 问题与可否定假设

A50r2两blur高噪声CG多数未解决。A51/A51b增加旧CG预算/改double未普遍解决。
独立的新理由是空间结构：零边界卷积H的正常矩阵可用循环卷积的频域逆作SPD预条件器。
本包只改变求解路径，目标仍为 B w=r，B=.25 H Hᵀ+eta I，eta=sigma² sqrt(alpha)/(1-alpha)。
P的频谱为 .25 |FFT(k)|²+eta，P^{-1}通过FFT计算；P只用于PCG，绝不把原零边界H改为周期H。
原float32核系数提升float64；这改变算术精度，不能宣称与原float32执行逐bit一致。

假设：同一128次迭代上限，PCG64在两blur×t999/319×WHITE/RANGE的8项严格真实残差≤1e-5。
对照LEGACY32和无预条件CG64，同样128 cap；记录实际operator/preconditioner调用和秒数，迭代cap相同不是成本相同。
若8项有任一不通过，计量给出否定结果并停止，不触发NN、不提高cap、不改变eta或阈值。
合成RHS准入不能证明实际去噪残差都解决，质量轨迹中仍须每步检查；失败fallback明确记账。

## 父源、队列和条件质量

父A50r2 `/data0/lzy/ZAPS-results/diag_current_score_geometry_r2/A50r2_20261026`，
两blur真实冻结算子和eta来源沿用A51身份校验；不重写任何旧文件。
计量独立seed580027，WHITE和同white经.5H的RANGE；3solver×8=24 solves，0NN。

只有计量全通过后，旧5FFHQ图×Gaussian/Motion共10例，固定同测量、同原PRE和29创新。
四臂 C_IEEE、OLD_U（原128 CG回退）、U_PCG（当前pullback方向）、V_PCG（同当前物理目标长度的物理方向）。
U_PCG与V_PCG都用同一个PCG64目标，不在V臂算去噪器输入VJP。全部输出冻结后才评价GT。
PCG失败仍使用原q回退，不能将失败步剔除；报告逐步resolved覆盖、最终PSNR/SSIM/LPIPS及算子/反向/墙钟成本。

## 成本、判据和停止

0NN准入24solves；质量阶段至多10×4×30=1200新诊断前向、≤600输入VJP，0优化更新。
源300NFE生产路径复用不当作新运行；四臂科学成本为各自原270＋最后30，均需另报探针/算子/反向。
U_PCG对C每任务平均PSNR≥+.05且SSIM≥-.002、LPIPS≤+.005、至少4/5图PSNR正向，仅称旧开发候选。
须同时报告对OLD_U和V_PCG的差值；不因C弱就把整体增益归因于当前状态。
若无质量增益或U不优于V，停止本机制扩展并保留负结论，不无限调参。
完整真残差≤1e-5是数值资格，不是后验正确、四任务迁移或PSNR改善保证。

输出新根 `/data0/lzy/ZAPS-results/diag_blur_pcg_admission/A58_20261027`，
独立结果分支 `results/blur-pcg-admission-20261027`。入口 `python -u utils/a58_session.py`。
断线同入口跟随，不自动重做失败；计量失败也自动归档，0NN负结论是完整产物。
