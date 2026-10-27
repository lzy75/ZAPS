# A62：两blur实际轨迹覆盖和质量资格

状态implemented；没有Mac Torch/CUDA实验结果。A61合成8/8通过是父资格，不当作重建收益。

## 问题、假设与协议

A61独立提交cfc36bd7a142f5b9c527b859d80e9ecfb2ec3346保留原零边界/原核/1e-5，
以新预算在CPU通过Gaussian/Motion两t×两RHS。高噪声实际需要607—801迭代，
不能把旧A58的128迭代失败改写为成功，不能宣称相同算子成本或0计算。
新假设：同一数值目标在真实去噪轨迹上可全部解决；当前响应方向是否优于同物理长度规则的v，
并相对旧q基线保持正向PSNR与SSIM/LPIPS保护线。

父A58 plan与A61 plan/summary签名和算子身份逐项核对。五张旧FFHQ图×两blur=10例，
同原PRE、测量、mask/kernel、seed1000、sigma.05、29个创新、原30点、原checkpoint/FP32。
不调时间点、噪声、模型、zeta/D或倍率。只新增CPU FFT求解的数值路径；GPU做原模型前向/输入VJP。
四臂：C_IEEE原q；OLD_U原128-CG与原q回退；U_FFT新可解决v对应当前u；V_FFT新v本身。
同长度规则不是分叉轨迹后每步向量范数数值相等。OLD_U的未解决比例单列，不能藏为方法失败样本剔除。

## 门控与成本

所有实际FFT步骤重新计算原零边界目标真实残差<=1e-5；0/24/29步还调用原空间double H/H^T
独立检查实际解，必须通过。其余步骤使用A61认证等价的FFT动作真实检查，不冒称逐步都做了空间复核。
核系数原float32 promote64，B=.25 HH^T+eta I、eta与A61一致；不加正则不改tol。
条件预算和每solve300秒/2048上限不改。新FFT任何失败立即停止并保留调用数，不用q替换后继续报成功。
U_FFT每步一次输入VJP；V_FFT不调用输入VJP。原checkpoint输入反向/模型参数/RNG守卫保留。
四臂各30前向，共1200新诊断前向、<=600输入VJP、0优化更新。源训练成本复用不再计算，
也不据此称此队列已经证明独立30调用部署或NFE节省。另记FFT/空间复核动作、传输及墙钟。
源C_IEEE端点重放误差<=1e-6。全部10例输出冻结后才评价GT，原8bitPSNR/SSIM与AlexLPIPS口径。
同一GPU串行，入口显式GPU UUID且开始>=6GiB free；逐案例仍检查。无信号终止任何其他进程。
旧completed复用；partial/failure不自动重试，失败账本同样归档。

## 可否定判据与后续

每任务U_FFT对C均值PSNR>=+.05dB、SSIM>=-.002、LPIPS<=+.005、至少4/5图PSNR正向，
仅为旧开发质量候选。同步报告对OLD_U/V_FFT及全30点覆盖，不只挑对C好看的结果。
U不优于V或有质量退化，保留任务边界，不强迫四任务一致，不继续多权重扫描。
通过后才建立旧图TIME匹配幅度/错配方向等更强控制，再用未用图确认；本轮没有TIME或新图证据。
当前u与方向刷新、曲率指标的独立贡献不由本轮直接证明。无跨数据集/第二seed/噪声/联合控制结论。

新根/data0/lzy/ZAPS-results/diag_blur_fft_quality/A62_20261027。
结果分支results/blur-fft-quality-20261027。入口python -u utils/a62_session.py --gpu GPU-e21d8c3c-cde6-c10f-878d-b20a4ded1742。
断线安全，30秒进度，结束集中summary/归档/push。真实Torch toy方向门控不skip，服务器执行后才启动模型。
