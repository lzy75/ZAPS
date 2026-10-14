# 已回传：状态/测量域配对结果

日期2026-10-14；用户回传summary和Decision，非本机GPU运行。
完整两组目录路径尚未提供，不杜撰hash/逐步曲线结论。

| Setting | null PSNR/SSIM/LPIPS | state PSNR/SSIM/LPIPS | dPSNR/dSSIM/dLPIPS |
| --- | --- | --- | --- |
| ImageNet Gaussian unit | 21.7256/.5387/.3958 | 21.5459/.4925/.4357 | −.1796725/−.0462794/+.0399318 |
| FFHQ SR model | 29.6929/.8788/.1245 | 29.7759/.8778/.1296 | +.0829590/−.0009999/+.0051220 |
| FFHQ SR unit | 26.7013/.8202/.1608 | 26.6606/.8195/.1604 | −.0406699/−.0006547/−.0004082 |

ImageNet null对缓存三指标零漂移；状态改变187步点。FFHQ改变211/215步点。
当前状态候选未获得综合收益：ImageNet三指标变差；FFHQ model仅PSNR取舍；
FFHQ unit接近无收益。否定当前映射候选，不证明两个指标无信息。
FFHQ域效应（unit-null−model-null）：−2.9916473dB/−.0586729/+.0363209。
PSNR状态交互−.1236289dB。两个物理SNR及引导单位不同，旧FFHQ经验T未换，
不能称只改噪声的因果结果，也不能按高分选择所谓正确论文协议。

用户下一项改为[FFHQ四任务公开设置复现](experiment_cards/R_ffhq_four_tasks_unit.md)：
恢复paper15/10/5、Adam默认lr=.001、公开初值、unit σ=.05，不加状态。
不是重复Karras/lr=.01/legacy T的26.7013，原状态实验不覆盖。
本轮优先核对四任务local reference，状态trace离线分析尚未完成。
