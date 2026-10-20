# A14：IEEE策略下六图双seed的固定时间clean响应

planned，2026-10-20；用户已授权连续、可并行实验。A13两状态仅cuDNN.allow_tf32由true变false，
native向量精确复现，原双尺度门槛恢复；仍只有一个事后选例，不能推广到全部案例。
本实验检验IEEE量测能否在原六图双seed、固定t14/t54复现可分辨的方向响应差异。
代码基于aad9153；实际源码、卡、测试、父plan/记录/模型/观测身份全部绑定到新plan。

## 来源、变量和固定条件

开发00020–22×seed1000/2000，活动缓存路径由A6.sources.A5的plan.source_cases定位，含恢复案例；
转移00023–25×双seed，缓存由A11/A12.case_sources定位。父A7已有两组完整最终F-C标签。
A13 /data0/lzy/ZAPS-results/diag_clean_response_precision/A13_20261020两任务必须complete、SHA/门控/判定符合已见正结果。
A11/A12源和A13完整identity一次审计，不递归准备/重跑历史。
唯一量测改动是精度策略固定为A13的IEEE设置；不改变生产轨迹、模型权重、原观测或PRE ζ/D。
SR4、unit sigma=.05、legacy_bicubic、clamped Tweedie、rho3 30×10、DDPM fixed variance eta1均继承。
原始q仍是原生产缓存的实际修正；不换成新IEEE中心q，不称它精确物理梯度。
同t/input、eta=.01/.02、2中心重复、两方向四对微扰和assess(.20门槛)全部沿用A11/A12。

转移组直接使用A11/A12保存的q/正交方向，不重抽；开发组o按旧Gram-Schmidt规则生成，
固定seed=811020+100*开发组排序case_index+t，仅一个高斯方向，不随机重试或按结果选方向。
00024_s1000两个状态直接复用A13已保存IEEE字段、策略和门控，新NFE0，明确标reused_A13。
其余22个状态各10诊断调用，总220新增诊断/0方法NFE。各层110调用，可分别在GPU0/1运行。
没有native重复臂，因为A13已隔离策略问题；没有新GT/图像加载或最终重建。

## 门控、结果和处理

源配置/网格、PRE/trajectory/y SHA、原q方向、t14/k28/index1及t54/k27/index2核对。
中心IEEE两次clean/q重复≤1e-7；相对旧native中心的差异只报告，不要求跨精度逐位一致。
旧native字段是身份/输入参照，不与IEEE差分中心混用；所有微扰中心在同一IEEE策略下计算。
新worker记录Torch/CUDA/cuDNN/GPU/策略；精度上下文、CPU/CUDA RNG、tau/zeta/D退出后恢复。
每次forward前持久记录调用，失败/中断不覆盖；两尺度输入表示/信号分辨/向量一致性按原assess判定。
每层分别报告开发/转移各3图的双seed一致计数；不把12例当12张独立图。
每组至少2/3图双seed都q更敏感，才记该组GUIDANCE_ALIGNED_RESPONSE_CANDIDATE；反向同理。
通过只说明量测及当前队列方向特异性，不说明方向指标能预测收益或指导更多步数。
不可分辨/尺度不稳全部保留，不换尺度、阈值或正交方向；A15相应层不拟合，不删除失败点。

## 运行和成本

新目录/data0/lzy/ZAPS-results/diag_ieee_response_value/A14_20261020；每层独立worker、每例独立记录/字段。
公共准备只审计一次，两个GPU任务均结束后才串行运行A15和归档Git，避免并发修改索引。
命令bash run_a14_a15_20261020.sh；仅一张可见卡则顺序；每选中卡至少6GiB空闲，不干预别人的作业。
完整完成队列方法3600/诊断2264=5864；早期单图另列。220=22×10，A13复用成本40已在原5644内。
这是付费诊断，每点10次调用，不能冒称比一次中点细化更便宜或免费在线状态。
服务器先执行无skip CPU Torch门控；Mac无Torch，只报告NumPy/协议测试。正式核心/历史脚本和卡不修改。
