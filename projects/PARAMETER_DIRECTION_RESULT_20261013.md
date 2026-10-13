# ImageNet Gaussian 终端ζ导数：已完成结果与下一项

日期2026-10-13。依据用户回传的完整run.json及pairs.csv；不是新的重建。
源码revision：4d4b15d0abd046779f36efe13545a05a1f2107ad。
experiment=archived_terminal_zeta_direction_audit，status=complete，新增600
诊断NFE，optimizer_steps=0。原始JSON/张量在服务器；本文件是分析存档，
不冒充原始运行文件备份。

## 已核实门控和结果

AD/no_grad输出relative_error=0、RNG等元数据一致；两臂存档指纹通过，
原loss32=.0029690340161323547完全一致。恢复原PSNR19.59868338、
SSIM.31679025、LPIPS.65045989。没有产生新的300NFE方法结果。
父快照SHA256：7b8786f952f76fcb89a8d179c63ca46135eebc6f45531b9210c8f6bda08c3d97。

| t/index | AD | FD中位数 | FD跨度/max幅值 | 分类 |
| --- | ---: | ---: | ---: | --- |
| 999/29 | .0022834933 | .0017141349 | .296766 | INCONCLUSIVE_SCALE_DEPENDENT |
| 333/14 | -.0001189304 | -.0000726150 | .195952 | SCALE_MISMATCH |
| 143/6 | -.0002263082 | -.0001726347 | .086168 | SCALE_MISMATCH |

三处×三尺度全部同号，没有支持局部参数导数方向反转。
333/143 AD约为FD的1.64/1.31倍。333刚好接近.20稳定门槛，不能强称
高精度认证。999 FD全为正，但不同h波动29.7%；不能挑一个h判PASS。
本结果仅审计3个ζ，不认证D/全梯度或作者实现。stop-gradient本来就是
近似；同号幅值误差不能直接推成几dB根因，也不据此调整学习率。

## 首步扰动轨迹（复用现有记录，不新增NFE）

原t999输入RMS1.000038、残差204.9774；下一t916输入RMS17.413166，
x0裁剪99.9013%、残差301.8244。h=.001时首步ζ扰动±.001，下一输入
RMS变化±.0834503（约.48%），不是小扰动后新增巨大爆炸。
后续clip统计改变，正反向loss斜率不对称；可能存在非线性/裁剪边界或
FP32前向敏感性，但当前仅存clip总比例，没有同像素mask交集，不能定位
哪类因素主导。AD/no_grad零差也不是独立重复数值噪声的完整测量。

已完成t999同状态VJP：exact_norm7418.14799334、raw_norm37063.09851289，
raw/exact=4.99627380664，cosine=.690301。残差统一为clipped squared
objective，非DPS的L2范数目标。

## 已获用户确认的下一项

[R-first-guidance实验卡](experiment_cards/R_first_guidance.md)：仅t999按
`a=exact_norm/raw_norm=.20014915889341542`缩放原引导方向，剩余29步不变。
两条冻结快照配对回放，60诊断NFE，不训练/改mask/改噪声/扫描倍率。
验证首步幅度是否对全程画质具有因果影响，而不是继续盲扫梯度。
这项是mechanism_only；哪怕正结果也不等于原文修复或状态创新。
独立入口/实验卡已实现，真实GPU待运行；共享核心、FFHQ/batch不变。
