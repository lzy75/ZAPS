# A75：当前倍率规则迁移DPS，原1000步基线

2026-10-31，A线，planned。用户明确授权优先DPS，复用已下载模型和既有复现代码。

## 目标与父记录

检验倍率模块能否脱离ZAPS学习参数，在DPS原物理修正上改善终局质量。
父A71 /data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029，72/72 complete；同三图00226/00443/00963、SR4及Motion、sigma=.05。
A72已表明噪声收缩对SR与blur作用不同，因此预先保留有/无收缩两臂，不能看结果再关闭某任务收缩。
既有DPS复现脚本diag_dps_same_observation.py：原DPS1000/DDPM/learned_range/PS scale=.3，保留其口径及隔离导入方式。
可否定假设：相对同设置原DPS，至少一项倍率版本在同NFE下出现联合质量增益；否则不宣称通用有效。

## 固定协议与唯一变量

原DPS、DPS+PROJECT（只做响应投影）、DPS+SHRINK（当前完整噪声收缩投影）。三组每个DPS步骤参与倍率，无10轮优化。
使用未经修改的vendored DPS p_sample_loop/p_sample与PS grad_and_value。官方configs/diffusion_config.yaml、task配置scale=.3。
固定1000原时间步、learned_range、clamp、原q_sample额外随机数消耗均保留；不把低步数较弱DPS当唯一基线。
复用A71模型、图像及y/算子：SR当前DPS Resizer，Motion当前保存的原生成核与零边界；有意共享H，非逐项原文算子或1000图均值复现。
q=-.3*grad ||y-H(m)||，新更新x_s=原已加噪sample+gamma*q，原噪声不缩放。
倍率响应仍为H(m+q/sqrt(alpha_t))-H(m)，不称真实下一去噪器响应。PROJECT令收缩因子为1，其余与SHRINK相同。
原scale=.3进入q，但投影版本会抵消其正标量直接幅度；这是替换长度校准，不是保留原scale作用的小修正。
所有臂相同输入观测、初始seed及原循环创新；GT只输出后评价；不调倍率上限、不选最好结果。

## 成本、时钟与准入

6case x3臂 x1000=18000方法前向、18000原DPS输入VJP，0优化反向。倍率模块不增加网络前向/VJP，但增加H和标量统计，计入wall-clock。
每个worker真实模型1次原DPS前向+输入梯度准入/warmup另计；TinyTorch门控另属接口测试，不冒充模型实验。
独立完整clock包含y加载、算子初始化、初始噪声、整个DPS循环及CPU输出同步；模型加载/warmup、指标、写图另计。
不得冻结参数requires_grad破坏旧checkpoint反向；用input_vjp_parameters兼容原checkpoint，autograd.grad仅输入，不优化NN，检查权重版本/grad未改变。
服务器强制Torch测试原条件与NULL(gamma1)全循环等价、相同初始RNG、无额外NN/VJP、源码与输入SHA，非有限停止保留，不回退1或自动重试失败case。
单卡串行/两卡显式分配；不结束其他进程。三组顺序按case轮换，同case同GPU。
Mac无Torch/CUDA，只做必要纯协议/CLI/语法检查；真实模型由用户服务器运行。

## 判读与分支

逐task联合PSNR/SSIM/LPIPS、逐图胜数与最差、完整秒数、NN/VJP/算子计数。两新臂全部报告，不输出事后择优混合策略。
开发信号：mean PSNR>=+.05、>=2/3图正、SSIM>=-.002、LPIPS<=+.005，非统计保证。
正向仅支持DPS小面板模块迁移；实时性独立贡献仍需固定表/错配后续，不能由模块胜原DPS推断。
若失败，不扩方法或调scale追分；保留原DPS。DiffPIR仅获方向认可，本轮不运行。

## 执行与归档

bash run_a75_20261031.sh --gpus 1（可用两卡--gpus 0,1）。
结果/data0/lzy/ZAPS-results/diag_dps_multiplier_transfer/A75_20261031；分支results/dps-multiplier-transfer-20261031。
summary/comparison.csv/md、真实PNG和1000步倍率轨迹自动归档上传；模型和输出PT留服务器。
