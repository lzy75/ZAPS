# A10：干净预测的局部变化是否包含细化价值信息

2026-10-20，planned；A研究线，独立零新增网络分析。

## 问题、来源与范围

A7实际q余弦/Ω未提供稳健增量；不调整其窗口、正则或阈值。
A8固定t14端点消融支持先验更新P保留完整F收益；A9同额外调用的末端重估提高PSNR但LPIPS恶化，
中点与末端没有Pareto优势。因此更换研究对象为clean预测场，尚不部署控制器。
可否定假设H10：在固定t14，用三个已访问clean预测的非均匀差分变化幅度，
比物理状态及仅clean速度更好地预测已有F相对C的最终收益。
仅探索性转移筛查：00020–22开发、00023–25转移，每图seed1000/2000，所有终局已经见过。
不是前瞻确认、真实Hessian/ODE曲率、严格局部误差、采样稳定性定理或在线收益。

父数据绝对目录：A2a活动案例由A5 plan.source_cases定位（含恢复注册后的第四例）；
A4 /data0/lzy/ZAPS-results/diag_terminal_continuation/A4_20261020；
A6 /data0/lzy/ZAPS-results/diag_fresh_refinement_confirmation/A6_20261020；
A7 /data0/lzy/ZAPS-results/diag_direction_value/A7_20261020；
A8 /data0/lzy/ZAPS-results/diag_endpoint_components/A8_20261020；
A9 /data0/lzy/ZAPS-results/diag_terminal_polish_control/A9_20261020。
必须completed、SHA/门控通过；A9完整汇总需复算一致且为NO_ROBUST_MIDPOINT_ADVANTAGE_ESTABLISHED。
代码基于5955ec4；全部实际父文件/代码身份冻结到新plan。

## 唯一候选及固定对照

z为缓存clamped Tweedie估计，u为原有归一化log sigma坐标；仅k=28(t14)、27(t54)、26(t124)：

    h1 = u_k-u_(k-1), h0 = u_(k-1)-u_(k-2)
    v1 = (z_k-z_(k-1))/h1, v0 = (z_(k-1)-z_(k-2))/h0
    speed = RMS(v1)
    roughness = RMS(v1-v0)/((h1+h0)/2)

三点二阶差分的幅度同时响应速度大小和方向变化；不是相邻图像余弦。
它沿含随机噪声的实际轨迹计算，可能受随机扰动、clamp及阶段影响；不能称固定时间场敏感度。
只测t14，排除混合噪声阶段产生的相关性。完整三张缓存只访问指定过去/当前切片，无GT/未来张量参与特征。
冻结三个ridge模型，lambda=1、开发组标准化、常数列scale=1、截距不惩罚：
B=(log R,log qRMS)；V=B+log1p(speed)；A=B+log1p(roughness)。
V是同特征数的速度对照，避免把单纯移动幅度当作粗糙度增量。没有模型/窗口/平滑/归一化搜索。
标签直接复用A7中两桥平均的F-C四项最终收益，PSNR/SSIM为正差，LPIPS/loss为负差。
不改成P标签，不重新计算分支、不选桥、不将两seed/两桥当独立图像。
任务/模型/算子/观测/PRE参数/30×10 rho3网格/原始随机过程/指标口径全部继承，仍是local_reference。
GT仅已有离线标签，不加载原图，不选输出、步数或终止。

## 预算、门控和判定

新方法0、诊断0 UNet NFE；不加载模型、不访问CUDA、不重训。仅CPU Torch安全读取原缓存，NumPy统计。
A2a–A9队列仍为方法3600/诊断1884=5484，早期单图另列。
A7来源审计复用；重算A7输入和原固定分析，核对A9记录/汇总/全部identity_files；缓存SHA严格核对。
缓存需CPU float32 [30,3,256,256]；只取三个切片后转float64，时间顺序/有限性/正跨度检查。
单测验证非均匀线性/二次路径、定速转向、未来数据不影响特征、转移标签不参与拟合、固定划分和门槛。
服务器Torch测试不得skip。身份不兼容/缺缓存/不完整直接停止，禁止自动补跑父实验。

H10支持门槛：A三质量图像平均MAE均不高于B、V、开发阶段均值；至少两项质量MAE
相对B和V都降低>=10%，且分别有>=2/3转移图改善。loss单列，不替代质量门槛。
通过只标探索性候选，下一卡在未使用图像前冻结规则做前瞻检验；不直接声称状态收益。
未通过停止这个三点clean差分候选，不改窗口/正则/特征追胜；转为明确的研究路线取舍汇总。
常数/不可分辨、缺输入或重复误差地板下的小收益保留，不强行支持。

## 执行和归档

命令：bash run_a10_20261020.sh；新目录/data0/lzy/ZAPS-results/diag_clean_roughness/A10_20261020。
锁、独立plan/run/SHA/points.csv/完整统计及日志；完成记录精确复用，失败记录保留不覆盖。
CPU完成后一次打印紧凑摘要，轻量结果追加results/refinement-confirmation-20261020。
上传失败不重跑分析或GPU；不改任何已冻结脚本/卡/核心。工作包仍45/100，不因运行轮数自动加分。
