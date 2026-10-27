# A56完整迁移与A61数值准入：实际回传结论

日期2026-10-27。使用连接的GitHub工具读取固定提交的JSON，未合并results、未下载PT、未Mac重跑模型。
三个summary经manifest SHA256/字节数、Git blob SHA1校验；240项逐图配对均值、胜图数、30点覆盖及成本算术核对通过。
详细机器可读证据见audit.json。旧partial目录与FFHQ早读记录保留，不覆盖。

## 1. 进程与完成状态

精确ps已确认三个占卡进程都不属于lzy，未采取中断。
其中一项Tl仅表明快照时停止/被跟踪，不代表可占用或可终止。

A56M1提交cba134427fe38e9dbbbcfc31213913911341d997：ImageNet8/8，partial0，恢复新增2100前向。
完整A56分支results/frozen-direction-transfer-selection-r2-20261027，提交7e67d6a3c8872ee903a461d6aa461d610a031693。
20/20案例，6张新FFHQ+4张新ImageNet×SR/IP，共10个依赖图像簇，一个seed，sigma.05。
U/V/Q/TIME全部30点解决，没有拿未解决步骤剔除后计算均值。

## 2. 最重要的画质表

|数据/任务|C_IEEE本地ZAPS|物理V|原方向Q|当前U|保留当前u的TIME|
|---|---:|---:|---:|---:|---:|
|FFHQ SR，6图|19.791|26.645|26.541|27.726|27.758|
|FFHQ IP，6图|21.976|22.340|22.342|30.369|30.404|
|ImageNet SR，4图|17.932|22.282|22.467|24.080|24.188|
|ImageNet IP，4图|19.350|20.799|20.799|26.371|26.196|

均值PSNR/dB。C为锁定本地参考，不是已认证作者benchmark。不能把U−C的全部差值当作新增实时指标收益。

ImageNet U−V：SR+1.79775、IP+5.57199；U−Q：SR+1.61281、IP+5.57199dB。
这些对照的平均SSIM/LPIPS均改善。SR有3/4图PSNR正向，IP4/4；FFHQ两任务均6/6。
冻结FFHQ旧规则迁至ImageNet另一模型/数据仍有方向机制的正向候选，不是仅在FFHQ重复旧图。
但ImageNet只有4图、一个seed，模型和数据同时改变，不等于广泛统计泛化或所有单图均优。

例外必须保留：ImageNet00195 SR，U−V为−.09941dB，U−Q为−.26986dB；
ImageNet00193 IP虽U比C PSNR高3.91605dB，但该单图LPIPS退化+.05130。
组均值通过不意味着每图保护线都通过。

## 3. 实时幅度仍不稳定：TIME不能误称无状态

U: T=K ||v||u/||u||；TIME: T=K g_bar_task(t) u(x,t)。TIME的g表来自旧FFHQ，
但**每一步仍计算当前u**。V/Q也有当前观测信息，不是整个方法无状态。

|U−TIME|PSNR|SSIM|LPIPS|PSNR正向图|
|---|---:|---:|---:|---:|
|FFHQ SR|−.03242|−.003263|+.006194|2/6|
|FFHQ IP|−.03450|−.000199|−.002663|4/6|
|ImageNet SR|−.10820|−.020926|+.000173|2/4|
|ImageNet IP|+.17502|+.005149|+.014650|2/4|

原平均保护线SSIM>=−.002、LPIPS<=+.005。ImageNet IP的PSNR增益伴随LPIPS超线，
且PSNR均值主要受00193的+1.13459dB影响；不能叫稳定通过。SR的SSIM也明显变差。
因此完整摘要的4/4指方向规则对指定C/V/Q的工程候选，绝不是4/4超过TIME或四个FFHQ任务。

综合A53/A54/A55/A60：当前方向有价值，旧cos衰减与当前标量g没有稳定增量，
简单滞后方向会降质。最强可解释主线是物理目标v经过当前去噪响应得到u，
不是已经证明随机轨迹余弦/曲率标量或自适应幅度优越。

## 4. 费用和效率不能混算

A56原队列6000方法+3624诊断=9624前向，1204输入VJP；另加M1 profile72前向/32输入VJP，
总9696前向/1236输入VJP，无partial。不是每张图都需要这些总数；继承科学协议各臂300前向。
U每条末轮额外当前输入VJP，TIME同样当前输入VJP，V/Q没有该部分；同前向不等于同时间。
已记录各臂末轮累计时长和算子子集，不能从共享服务器固定顺序计时推出生产加速。
A57独立30调用等价只在旧SR/IP cohort证实；此次迁移保留继承300调用收费，不事后扣除270。

## 5. A61：数值瓶颈已获可用资格，不是blur画质结论

提交cfc36bd7a142f5b9c527b859d80e9ecfb2ec3346，8/8通过；0新UNet/0输入VJP。
两核H/H^T相对原double空间实现误差约5e−16—1.5e−15，零边界未改。

|高噪声t999|WHITE迭代|RANGE迭代|WHITE真实残差|RANGE真实残差|
|---|---:|---:|---:|---:|
|Gaussian|758|607|9.6954e−6|9.8562e−6|
|Motion|801|651|9.9629e−6|9.7987e−6|

t319四项59—63迭代，残差也<=1e−5。高噪声预算由条件数上界预先算1069，不是见画质后选预算。
这也说明旧128迭代不足；新方案更换FFT动作路径并提高明确求解预算，不能说旧同成本PCG成功。
共12324次FFT变换，另有原空间复核、RHS生成和CPU时间；0UNet不等于0计算。
当前只证明合成WHITE/RANGE可解，真实去噪轨迹的残差覆盖和最终质量尚缺。

## 6. 下一步A62

沿用A58原五图和四臂结构，保留原q、旧128-CG回退U、新FFT-U、新FFT-V。
实际每步重新计算真实FFT目标残差，0/24/29步以原空间double独立复核；任何新求解失败即停并保留账本。
完整10例后才读GT，1200新诊断前向/<=600输入VJP，无学习/倍率扫描。
目标是分辨数值覆盖和方向质量，不偷换为新图、TIME增量、曲率或四任务全部成立。
通过后再建立blur的强TIME/错配对照和新图确认；这轮尚无相应证据。

曲率/联合控制继续按20261027_integrated_next_steps.md的两包设计推进：
同t有符号局部曲率→含gamma1的幅度动作；状态时效→触发刷新。两轴单独有增量后才联合作2×2对照。
当前固定30点，不修改A56已冻结网格，也不把Jacobian响应重新命名为轨迹曲率。

## 7. 原图位置（已有真实输出）

服务器完整A56根：/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/selection_r2_20261027/A56_selection_r2_20261027。
每例`<case>/transfer_comparison.png`，含原始GT和各方法；8张ImageNet对照已在结果分支。
GitHub路径前缀：projects/server_records/uncertainty_feedback_2079b3b5ace87ac9。

- ImageNet SR边界例：[00195对照](https://github.com/lzy75/ZAPS/blob/7e67d6a3c8872ee903a461d6aa461d610a031693/projects/server_records/uncertainty_feedback_2079b3b5ace87ac9/imagenet_super_resolution_00195_s1000/transfer_comparison.png)
- ImageNet IP正向例：[00067对照](https://github.com/lzy75/ZAPS/blob/7e67d6a3c8872ee903a461d6aa461d610a031693/projects/server_records/uncertainty_feedback_2079b3b5ace87ac9/imagenet_inpainting_00067_s1000/transfer_comparison.png)

本页从数值与归档路径核验，不声称本轮已逐张视觉审阅PNG。
