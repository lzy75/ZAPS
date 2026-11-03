# A79R1：DiffPIR 数值形式与倍率模块的二因素对照

状态planned；2026-11-03；独立目录，不覆盖A79。

## 问题、父证据、可否定假设
A79结果分支提交4ee15052126e6aef54ee9f90da7de25caf86ca7e记录0模型调用。
真实Gaussian观测、t999时rho约7—8e-7，官方float32 Woodbury式正规方程相对残差分别约.0393/.0299，未通过原1e-3门控。
本次区分“数值表达式变化”与“倍率模块收益”；不将官方算术更换后结果冒充原版。
父A79 `/data0/lzy/ZAPS-results/diag_diffpir_multiplier_transfer/A79_20261103` 必须仍为0模型成本；A71仅提供冻结核及图像路径。

## 因子、公式与固定条件
四臂：DIFFPIR、SHRINK、STABLE_DIFFPIR、STABLE_SHRINK。
第一因子为官方float32解/数学等价的稳定float32解；第二因子为倍率1/不变的A71倍率模块。
仅sf=1圆周去模糊：令F为FFT、B为核频谱，R=conj(B)F(y)+rho F(m)。
官方写法 z=F^-1[(R-conj(B) B R/(|B|²+rho))/rho]；等价稳定写法 z=F^-1[R/(|B|²+rho)]。
两者解同一个(H^T H+rho I)z=H^T y+rho m；后式避免小rho下相近数相减。
参考来源为固定提交的官方DiffPIR `utils/utils_sisr.py`；这里仅是sf1代数化简，不作为方法创新。
原A79源、vendor、实验卡和失败记录不改。稳定解仅在新runtime临时绑定，退出恢复。
其他配置完全沿用A79：两数据集各3图、两blur、物理sigma=.05、100点原生quadratic网格、原文lambda/zeta、相同初始化/随机创新及原输出。
已经生成的旧y按hash复制；其余按同协议生成。不读取学习参数，不用GT选择超参数。

## 门控、预算与成本
先0NN计量原float32、原float64、稳定float32；早中晚t=999/500/5。
原float32是否满足原1e-3阈值继续报告，失败仍为失败。该臂仅作作者算术参考。
稳定解准入：原float64在float64算子下相对残差<=1e-8；稳定float32在float64算子下<=1e-5；稳定float32对原float64相对解误差<=1e-3。
真实Torch测试必须通过，不因Mac无Torch而skip：稳定式对原float64、gamma1输出/RNG一致、原/新绑定恢复、无反向、调用数。
6图×2任务×4臂×100=4800前向，0输入VJP/优化反向；每次加载真实探针另计，FFT操作单列。
每run原生99次prox、100前向；输出/噪声/裁剪不改。完整计时口径沿用A79。
仅使用指定GPU，资源准入和断线安全队列沿用A79；模型失败不自动重试。

## 判据与结果分支
分别在原算术配对及稳定算术配对报告模块差值，另报告求解形式差值。
每数据集/任务平均PSNR>=+.1dB、至少2/3图胜、SSIM不降且LPIPS不升，才列为该配置正向候选；权衡单列。
若收益仅来自数值修正，不归因于模块；若模块退化，不扫描新参数救分。门控失败则停在模型前保留原因。
这是小型开发对照，不是原论文完整benchmark，不证明普适或实时状态必要性。

## 入口与结果
`bash /home/lzy/ZAPS/projects/run_a79r1_20261103.sh --gpus 1`
根 `/data0/lzy/ZAPS-results/diag_diffpir_multiplier_transfer/A79R1_20261103`。
轻量结果分支 `results/diffpir-numeric-transfer-r1-20261103`，含所有原阈值失败与新门控记录、逐图质量/clock/倍率轨迹。
