# A44：冻结连续规则的多图迁移与状态配对消融

状态planned；2026-10-25；A线；机制迁移测试，非独立最终确认。

## 问题、父实验与范围

A43主候选NYSTROM在8个状态均未通过增量门槛，停止其补全/倍率/阈值救援。
次要臂PROJECTED在00337的inpainting/t319相对PHYSICAL +.46450dB、ISO
+.06846dB，SSIM和LPIPS也通过既定保护线。这是探索性信号，不修改A43主
判定。新问题：冻结PROJECTED之后，其逐图匹配的信息是否能超过强固定倍率
和别图的状态倍率？不再拟合新权重、不增大上限、不放宽数值/画质门槛。

父目录：A41 `/data0/lzy/ZAPS-results/diag_ffhq_action_geometry/A41_20261025`；
A43 `/data0/lzy/ZAPS-results/diag_continuous_uncertainty/A43_20261025`。
均须completed、签名、代码/模型/测量/输出身份及数值门控通过，原文件不覆盖。

图像固定为A41中排除00337后的全部五图：00446、00559、00608、00873、00951；
四任务SR/Gaussian/inpainting/motion，seed1000，共20例。它们未参与A42/A43，
但在更早实验中使用过，**不是全新盲测集**。复用已完成训练，新增训练NFE=0。
不因为某任务负结果而删图/换图。ImageNet、正则、步长/网格均不在本轮。

## 唯一科学改动与对照

固定k24/t319→t218施加一次引导脉冲，后面全部原native采样。位置由A43
四任务末段表现确定，在本轮前锁定；本轮只检验强度的逐图匹配，**不能证明
状态触发位置、完整轨迹控制或步数分配**。全部原30步、first270、last_opt、
ζ/D/tau、sigma=.05 unit-domain、mask/kernel、精确算子、RNG与指标口径继承。

|臂|含义|
|--|--|
|C|原配对参考|
|FIXED8|所有任务/图均8倍|
|TASK_FIXED|各任务固定使用A43/00337的PROJECTED倍率；SR127.086、Gaussian7.589、inpaint28.526、motion8.327|
|PHYSICAL|A43连续物理损失规则，冻结拟合/噪声预算/实际检查|
|ISO|同样控制流程，V内各向同性高斯目标|
|STATE|同样控制流程，V B V^T高斯目标，即A43 PROJECTED|
|SHUFFLE|把同任务下一张图（按ID排序、循环）的STATE倍率施于当前图；保留倍率集合，破坏逐图匹配|

固定倍率从A43已签名选择记录读取，和上述数值一致才运行。所有20例选择先
冻结，再生成SHUFFLE分配，再评估终局。没有GT输入选择器，不依据终局指标
重排倍率。SHUFFLE是离线信息消融，不能作为可部署在线方案。

传感器和规则完全复用A42/A43：5动作、raw中心重复、rank<=2、两尺度IEEE
导数，原PSD/偏斜/尺度门控；连续二次拟合误差<=.05、噪声预算、.001量化、
最多3次实际前向检查。拒绝也计费。状态不可用则保持1。没有NYSTROM分支。
每例原C重放30次须<=1e-6相对误差，测量损失复现A41<=1e-6，raw重复<=1e-7；
参数/RNG/精度不变，全部UNet调用逐次记账。

## 预算与判据

新增diagnostic最多1660：20×[30原C重放+14传感+9验证+30后续终局]；
重复倍率和原五动作终局可复用。每例choose<=53，finish<=30，失败已记录
成本占同一阶段上限，不能通过重新运行重置预算。突然中断计费仍可能下界。
科学采样每臂300，STATE完整传感部署保守<=317，PHYSICAL可更便宜；
不宣称等300预算、NFE节省或付费传感优越性。历史训练不计新增但不是免费。

按图像簇、按任务分别报告C/FIXED8/TASK_FIXED/P/ISO/STATE/SHUFFLE的全部
PSNR/SSIM/LPIPS、全域和informative物理loss、倍率分布、NFE及失败。
每任务初步匹配支持须：STATE-C均PSNR>=.05且至少4/5图为正；STATE相对
TASK_FIXED、PHYSICAL、ISO、SHUFFLE各均PSNR>=.02且至少3/5图为正；各比较
均满足SSIM>=-.002、LPIPS<=+.005。STATE相对FIXED8也单列。
所有任务都报告；只少数任务通过就叫任务局部线索，不叫四任务体系成立。
5旧图只作工程筛查，不报显著性或把task×image=20当20独立图。

若支持，才准备全新图/第二seed独立确认及每一步触发机制；若固定或错配同样
有效，则否定本轮逐图匹配价值，保留动作收益；若PROJECTED仅特定图有效，
停止该局部高斯映射的调参救援。所有负结果保留。

## 运行与归档

`bash /home/lzy/ZAPS/projects/run_a44_20261025.sh`：先补A43轻量归档（零UNet），
再双卡choose→全体冻结→finish，单卡则串行。每分钟进度，集中最终输出。
新目录 `/data0/lzy/ZAPS-results/diag_frozen_state_matching/A44_20261025`。
轻量结果和全部20张对照图一次推送 `results/ffhq-state-matching-20261025`，
PT留服务器。Mac仅NumPy与语法检查，真实Torch/CUDA服务器执行，不允许skip。
