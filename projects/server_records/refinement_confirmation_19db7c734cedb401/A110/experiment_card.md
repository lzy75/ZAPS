# A110 超分的在线零空间增量对照

状态 planned，A 工作线。使用 A103 已完成的 FFHQ 00000–00009、超分 ×4、模型坐标噪声 0.025 与 0.10 的全部 20 个案例。固定原观测、种子、checkpoint、30 点网格、历史引导映射及学习设置。A99 的终局分解与 A103 的缓存核查支持定位，尚需验证逐步去除零空间增量后，终局退化是否减弱。

## 唯一改动

保留 ZAPS、闭式标量 \(\delta_S\)、闭式对角 \(\delta_D\) 三组。第四组在各自状态按原公式得到二者，再用真实退化算子行空间的正交投影 \(\Pi_H\)，取 \(\delta_R=\delta_S+\Pi_H(\delta_D-\delta_S)\)。标量项保持完整。\(H\delta_R=H\delta_D\) 在同状态成立，轨迹各自演化。投影由原 bicubic 降采样两轴的真实 taps 和 SVD 构建。数值门控检查原 H、投影后观测响应及冻结代理损失。空间投影后的修正可超出原小波对角参数族，本组用于定位机制。

## 预算和判定

每例共享 270F/9 优化反向的原 ZAPS 前缀，四臂各 30F/1 优化反向，最终评估第十轮已生成的图。每臂方法预算 300F/10 优化反向/0 输入 VJP 或 JVP。实验新增 7800F/260 优化反向。记录实际网格、尾 RNG、H 和小波调用及前缀加末轮耗时。投影组耗时含每步同状态审计。GT 只评估终局。

报告 PSNR、SSIM、LPIPS、原始总 MSE、观测行空间与零空间 MSE、干净和带噪观测误差。若投影对完整对角的平均原始 MSE 降低且至少 7/10 胜，支持零空间增量在该噪声设置下有实际贡献。若投影后仍落后于标量且干净观测误差仍增，下一步定位观测域目标。质量正向沿用平均 PSNR ≥ +0.10 dB、至少 7/10 PSNR 胜、SSIM 不降、LPIPS 不升。保留退化和不确定结果，数值门控失败则保留现场。

## 运行与归档

在服务器 projects 目录运行 `/home/lzy/anaconda3/bin/python -u utils/diag_300nfe_feedback_confirmation.py sr`。父目录 `/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A103_20261111`。独立结果 `/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A110_20261111`，保存计划、逐图质量、输出和每步分解。
