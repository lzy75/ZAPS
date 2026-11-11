# A111 去模糊独立观测噪声确认

状态 planned，A 工作线。使用已完成 A102 的 FFHQ 00000–00009、Gaussian 和运动去模糊、模型坐标噪声 0.05，共 20 例。检验闭式对角相对闭式标量的终局收益对新观测噪声是否保持。

唯一改动为每个观测的生成种子，取旧 measurement seed 加 20261111。原图、重建种子、噪声标准差、Gaussian 算子、原冻结运动核、FFHQ checkpoint、30 点网格、学习设置及三种方法沿用 A102。新观测形状必须一致且张量不同，原始和新观测均保留。各臂使用相同新观测。真值用于生成退化观测及离线评估。

每例共享 270F/9 优化反向的原 ZAPS 前缀，原 ZAPS、闭式标量、闭式对角各 30F/1 优化反向。每臂方法预算 300F/10 优化反向/0 输入 VJP 或 JVP。实验新增 7200F/240 优化反向。记录时刻、尾 RNG、调用和前缀加末轮耗时。采用第十轮已生成图像。

对角相对标量须平均 PSNR ≥ +0.10 dB、至少 7/10 PSNR 胜、SSIM 不降且 LPIPS 不升，逐任务判定。同时报告两者对原 ZAPS、干净与带噪观测误差和原观测的对照摘要。负向与指标取舍均保留。结果只说明同图、同噪声强度下对第二组噪声实现的稳定性。

在服务器 projects 目录运行 `/home/lzy/anaconda3/bin/python -u utils/diag_300nfe_feedback_confirmation.py blur`。父目录 `/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A102_20261110`。独立结果 `/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A111_20261111`，保存新观测、计划、输出、逐图指标和原观测对照摘要。
