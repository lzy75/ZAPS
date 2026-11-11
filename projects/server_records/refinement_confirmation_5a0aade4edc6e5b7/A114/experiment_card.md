# A114 ImageNet 去模糊的 300 NFE 迁移验证

状态 planned，A 工作线。A111 在同一批 FFHQ 图像的第二组噪声下复现闭式对角相对闭式标量的 Gaussian 和运动去模糊收益。A113 超分在两档噪声下回到标量水平，保留为适用范围结果。本轮检验 A111 的原闭式标量和原闭式对角能否迁移到 ImageNet 和已有 ImageNet 扩散模型。

固定 ImageNet 256×256 的 00000–00009，两个任务，模型坐标噪声 .05，重建种子 1000 加图号，观测种子 Gaussian 401000、Motion 601000 各加图号。Gaussian 61、sigma 3、零边界。运动核沿用 A102 已冻结 DPS 运动核。ImageNet 模型 FP32，哈希与已存本地参考一致。Gaussian 的 30×10、15/10/5 网格、lr=.001、初始 ζ=.2、D=.2、db4/level3 和 DDPM 配置来自 ImageNet 本地参考。运动去模糊复用该学习配置和 A102 运动算子，属于本地任务扩展。保存每图和观测哈希。

每例三臂使用同一新观测和 270F/9B 学习前缀。比较原 ZAPS、闭式标量、闭式对角，各自运行末轮 30F/1 优化反向，输出第十轮已生成图。每臂方法等效 300F/10 优化反向、0 输入 VJP/JVP。联合新增 7200F/240 优化反向，单独报告实际计数。核对时刻、尾 RNG，保存原有质量指标、图像和前缀加末轮的同步耗时。GT 用于生成观测和终局评价。

## 判定

逐任务报告三组的 PSNR、SSIM、LPIPS 和耗时。正向标准与 A111 相同，候选相对参照平均 PSNR ≥ +.10 dB、至少 7/10 PSNR 胜、SSIM 不降、LPIPS 不升。对角相对标量是迁移主比较；同时报告两者相对 ZAPS。任务和样本均保留，图像和模型一起变化，结论限本地配对迁移。Gaussian 的本地参考分数和论文报告值使用不同样本与噪声约定。

服务器 projects 目录用至少 9 GiB 空闲的已分配 GPU 运行 `/home/lzy/anaconda3/bin/python -u utils/diag_imagenet_closed_blur.py`。独立结果 `/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A114_20261111`。完成后 `--archive` 推送 `results/imagenet-closed-blur-20261111`，失败记录保留。
