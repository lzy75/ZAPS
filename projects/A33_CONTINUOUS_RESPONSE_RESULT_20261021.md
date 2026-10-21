# A33 连续响应反馈：已完成结果与A34依据

2026-10-21。来源：结果分支commit 19285a6，归档
projects/server_records/refinement_confirmation_7277c6bc8eae4e3e/A33。
Mac核对manifest全部164个轻量文件SHA、72个arm的complete状态与门控；
76个二进制不在Git归档中，仍在服务器，Mac没有运行Torch/UNet。
本机副本 /Users/liziyue.22/ZAPS-deliverables/analysis/A33_20261021/A33。
服务器 /data0/lzy/ZAPS-results/diag_continuous_response_feedback/A33_20261021。

六张重复开发图00020–00025，各seed1000/2000。72/72任务完成，无部分attempt。
新增诊断2160 NFE，每个科学方案为原270+末30=300 NFE，传感器额外UNet=0。

| 方案 | PSNR相对C | SSIM相对C | LPIPS相对C | PSNR相对匹配TIME | 最终跨case残差RMS/噪声 |
|---|---:|---:|---:|---:|---:|
| I_cont2 | +1.235600 | +.037551 | +.001644 | −.001241 | 1.000624 |
| I_cont8 | +.684387 | −.069778 | +.206868 | −.001287 | .464017 |
| I_free | −1.838949 | −.222086 | +.414637 | −.001253 | .065727 |

I_cont2相对旧A32离散I：PSNR+.015448dB、SSIM+.001024、LPIPS−.005363。
它是当前较好的反馈画质候选，但匹配STAGE_cont2相对C为+1.236841dB，仍稍高。
上限2的画质容差相对C通过；相对F的LPIPS增加.006109，超过+.005。
三个实时反馈组均未超过匹配时间组，原预注册状态增益条件均未通过。
不能声称所有状态反馈无效，也不能把+1.236dB全归因于实时状态选择。

连续化之后，I_cont2有263/360个倍率脱离旧离散集合；I_cont8/I_free均357/360。
因此“只有离散化掩盖状态价值”已不足以解释本轮结果。
I_free实际最大gamma=38.53953；图00020_s1000的I_cont8/free出现明显彩色颗粒，
原图及固定ROI来自真实输出，没有增强。物理残差降低与终局质量在此设置下明显分离。
终局残差RMS由sqrt(2*mean(loss))计算，不是逐case RMS的算术均值。

## 指标与理论边界

r=y-H(clean)，a=H(clean+q/sqrt(alpha_bar))-H(clean)，c为r/a余弦，b=||a||/||r||。
gamma_LS=max(0,c/b)是同t响应代理的非负最小二乘解析解，而不是终局画质最优解。
q已包含原ZAPS的ζ/D。gamma上限2是开发设置下的可用工程限制，不是全局稳定性定理。
当前c/b是物理响应读数，不等同于旧相邻q余弦或轨迹曲率/稳定性。
超分辨率算子有未观测方向，且y带噪；低测量残差不足以唯一确定高质量重建。
终局用GT的PSNR/SSIM/LPIPS及统一ROI图评价；GT不得进入在线选择。

## 下一步A34（计划，非新结果）

使用已有R=RMS(r)/sigma_y，检验噪声目标tau=.9/1/1.1的连续反馈与匹配时间表。
这是discrepancy-inspired假设：达到与测量噪声相当的代理残差后不继续加强测量拟合。
依据为Hansen《Regularization Tools》§2.9对discrepancy principle的阐述：
https://www2.imm.dtu.dk/~pcha/Regutools/RTv4manual.pdf 。
传统原理支持按噪声程度控制数据拟合，不能直接证明本次同t代理的跨步效果或最优画质。
目标±10%是模型失配敏感性试验，不是统计置信区间。
详见 experiment_cards/A34_noise_target_response_feedback_20261021.md。
