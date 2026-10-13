# R-first-direction：同幅度的首步精确方向验证

日期：2026-10-13；工作线R；mechanism_only。
用户确认“同幅度、仅换首步方向”，授权继续实现。真实GPU结果待服务器。
不修改共享核心、正式single/batch、FFHQ、参数默认值或已有父文件。

## 1. 目标与父记录

可否定假设：在已经校准的首步幅度下，首步近似方向仍限制全程画质。
父实验`first_guidance_20261013_213249_896043`用户回报：PSNR19.5987→
20.7478，SSIM.3168→.3951，LPIPS.6505→.5823，下一RMS17.4132→3.6147，
clip99.901%→99.306%。支持幅度敏感，但饱和仍在、方向尚未干预验证。
本机只收到摘要，完整父JSON/缓存张量由入口在服务器核验，不伪称已下载。

父绝对目录：
`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/first_guidance_20261013_213249_896043`。
其源码revision f8eb9a2；依赖完成的VJP审计`guidance_clipping_20261009_142630`、
终端ζ参数审计及原第10轮PRE-update快照。源训练是30×10=300NFE。
复用父输出`first_norm_matched.pt`，不重跑原始/校准基线或重训。
不是重复硬mask（重建失败约9.3dB）、SR全程归一化、DPS30或晚期噪声。

## 2. 唯一改动及固定条件

统一第一步目标：

`L999=.5*sum((y-H(clamp(Tweedie(x_T,999))))²)`。
`g_exact=-∂L999/∂x_T`，包含真实UNet输入Jacobian及该目标的clamp导数。
权重requires_grad保持custom checkpoint所需的True，但不优化/积累权重梯度。
每个输入图只反传一次，避免custom checkpoint ctx被释放后重复backward。

父参照首步：`δ_ref=ζ999 B999(a H^T r)`，a来自父VJP范数比，约.2001491589。
新首步：`δ_test=ζ999 g_exact*||B999(a H^T r)||/||g_exact||`。
这是同目标同修正幅度，只改变方向；不是DPS L2 norm/scale=.3，不改变ζ。
剩余29步使用原wavelet近似公式，仍joint学习后的ζ/D快照，不冻正式D分支。

实现仅在首个Hv计算临时返回：
`Hv_test=(sqrt(alpha999)*target-v)/(1-alpha999)`，使实际未修改核心
`(v+(1-alpha999)*Hv_test)/sqrt(alpha999)`等于target；检查真实fp32代数误差，
不是把精确方向再经过B或改变posterior。之后transpose/synthesis直接返回原结果。
禁用scope时bitwise-null，CPU小张量测试；不新增一整条GPU null。

固定ImageNet00001、seed1001；原H61/sigma3/zero padding；原model-domain
观测sigma=.05；官方fp32权重；paper15/10/5完整999→0网格；fixed-small
DDPM eta1；原x0 clamp；db4 level3；第10轮更新前ζ/D（源lr=.001），原x_T。
无训练/Adam/投影/噪声缩放/调度修改/方向扫描/GT调幅或选择输出。
指标仍原uint8 PSNR/SSIM、alex LPIPS，GT在轨迹结束后用于两条预定输出评价。

## 3. 预算与门控

新增1次UNet输入VJP+30步轨迹=31诊断NFE；weight backward开销计入秒数。
cache复用计0新NFE，保留其父运行的30诊断NFE与300方法NFE来源，不声称
缓存方法免费或新方法只需31NFE。所有实际顶层UNet调用hook计数，失败调用亦计。

- 父status=complete，60NFE、无重训，原baseline/fingerprint/loss/pairing全部通过。
- 父配置、参数/VJP父JSON hash、snapshot、checkpoint、H、y、图像hash一致。
- 缓存张量形状/精度/有限性、terminal loss及重新评价的三个指标与父JSON一致。
  父脚本没有记录输出.pt的保存时hash；本次记录读取hash并核对loss/指标，
  是缓存兼容性门控，不冒称与过去保存时逐元素认证。异常时停止，不能
  为保持31NFE预算偷偷换一个缓存，或自动补跑原轨迹。
- t999保存input必须与x_T逐元素相同；只一个probe。live epsilon与原capture
  atol/rtol2e-5；live exact/approx norms与旧VJP rtol2e-5/atol1e-6。
- 复算校准后第一步posterior（零新score）下一RMS须匹配父轨迹，tol同上。
  该reference用原capture保存的no-grad epsilon，先与本次可微前向核对；
  不把fp32 grad-mode舍入差当作父reference的改变。精确VJP仍由本次
  实际可微前向计算，记录可微及reference目标值，不偷换成冻结epsilon梯度。
- 首步和后29步模型时序保持30次；精确方向实际core代数allclose
  atol/rtol2e-5；修正范数与参照rtol2e-5/atol1e-6。
- 首步输入/残差/clip指纹，沿用atol1e-6/rtol2e-5、clip_atol2e-5。
- 将原generator重放29次相同形状DDPM噪声，零新UNet，重建预期RNG尾；
  原第一步posterior含一次、再诊断重放28次。模型eval每次前后RNG检查必须
  不消耗随机数。新实际轨迹RNG尾必须与预期相同，仍29个采样draw。
  这些额外29个诊断draw计入记录/耗时，不是重新生成测量噪声或新轨迹。

门控失败/非有限值/序列异常立即写failed并停止，不解读画质。
source CPU及一张采样GPU RNG恢复；可见GPU数可不同，其余generator不修改。
独立生成的新方向、cached tensor的hash记录；父tensor不覆盖。

## 4. 预注册判读

全部delta相对**已校准近似方向20.7478**，不是相对19.5987原始基线。

| 类别 | 事先量化条件 | 下一步 |
| --- | --- | --- |
| 支持首步方向敏感 | dPSNR>=.2dB、dSSIM>=.005、dLPIPS<=-.005同时成立 | 讨论Jacobian近似与匹配训练；不直接修改正式核心 |
| 无最终画质支持 | dPSNR<=.05、dSSIM<=.001、dLPIPS>=-.001同时成立 | 降低该首步方向候选优先级；不扫更多方向/倍率 |
| 混合/小幅 | 其他结果 | 不挑最好指标宣称成功；先讨论，不自动扩大实验 |
| gate failed | 任一身份/代数/随机过程检查失败 | 先修接口，不把失败当算法结果 |

单图冻结路径的成功不是原文基线认证、跨图泛化、状态感知收益，且不能
认证剩余29步近似正确。失败也不能证明所有方向都无关。

## 5. 运行与记录

```bash
cd /home/lzy/ZAPS/projects
python -u utils/diag_zaps_first_direction.py \
  --first-run-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/first_guidance_20261013_213249_896043 \
  --device cuda
```

输出新目录`first_direction_时间戳`：experiment_card.md、run.json、summary.csv、
trace.csv、first_exact_equal_norm.pt。逐步追踪只新轨迹；父记录完整路径/身份hash
链接保留。状态complete仅指运行完成；是否支持由判读/三个指标决定。
真实GPU待服务器验证；本机无PyTorch，协议测试与张量测试分开报告。
