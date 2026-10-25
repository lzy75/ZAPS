# 自适应 SDE：作者源码中的噪声路径与实际调用计数

核查时间：2026-10-26；仅静态阅读，未运行 JAX、Torch 或 UNet。

来源为 [Gotta Go Fast When Generating Data with Score-Based Models](https://arxiv.org/abs/2105.14080)
（arXiv v1，2021-05-28；本轮未核实正式会议接收信息）及
[作者仓库](https://github.com/AlexiaJM/score_sde_fast_sampling)。README 明确 Torch 文件是简化移植，
不应将它的返回计数直接等同于论文主实验所用 JAX 实现。

## 对我们有用的部分

同一次尝试内，Euler 与改进结果共用该次噪声，用混合绝对/相对容差归一化差异；
误差合格才推进状态/时间，拒绝会重新尝试，步长按误差反馈调整。
这里衡量的是嵌入数值解的局部差异，含两次 score 评估，与相邻随机轨迹余弦不同。
这支持 B 模块使用付费误差估计的动机，不证明我们离散 skip-DDPM 控制器已成立。

## 拒绝后的噪声不能被默认为共同 Brownian 路径

已读 JAX `sampling.py` 的 Adaptive 类和外层循环，以及完整简化 Torch 文件。
JAX 在下一轮循环重新 split RNG；Torch 每次 update 都调用新的 `randn_like`。
一次尝试内共噪声，并不等于拒绝/再细分时保留了同一条 Brownian 路径。
本轮未见这两处循环保存拒绝区间的 Brownian bridge，因此不能借此声称当前 B
离散共同创新路径具有无偏自适应 SDE 保证。需要独立的线性 Gaussian 检查和状态依赖拒绝审计。

## 源码计数接口的差异

|范围|源码观察|本项目的计数要求|
|---|---|---|
|简化 Torch 自适应预测器|每次 update 调用两次 reverse-SDE RHS；标准 RHS 每次调用一次 score|每次尝试计两次前向，含拒绝|
|简化 Torch 外循环|`N` 每次尝试加 1，返回 `N+1`；可选末端去噪另调用一次 score|通常 `denoise=True` 时实际为 `2*attempts+1`，不能照抄返回值|
|JAX 主实现|预测器尝试的计数加 2；可选末端去噪在返回计数前另调用 score|核对末端调用是否包含；不能用 Torch 移植的问题评价 JAX 主实验|

上表是静态接口核查，不是运行复现，也不据此断言论文报告的速度或科学结论错误。
未全面审阅其他 corrector 的调用计数。我们须用模型前向 hook 和显式反向尝试计数，
同时保存失败、拒绝、末端去噪与计时，禁止把返回变量名 NFE 当作真实网络调用证明。

## 可定位源码

- `sampling.py`：blob `f8fd8371da97892e133b1fc0d0a624481d3cc147`；相关区间 146–300、536–675。
- `sde_sampling_torch.py`：blob `f608bdf5495c86d6712a2ee9219c9daf26517a2f`；全文已读。
- `README.md`：blob `da1f05b6d98f1179a58998d0a91a57b65d28c473`。
- 作者移植所依据的 [score_sde_pytorch/sde_lib.py](https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py)
  reverse-SDE RHS 相关行 84–109 已读；本轮未将其整文件视为已审计。

当前结论只约束 B 的实验契约，不新增第三条机制，也未新增任何网络调用。
