# R：FFHQ四任务，公开原设置＋unit物理测量协议

2026-10-14；用户请求执行四任务原配置/调度复现对照，GPU待运行。
独立入口，不改共享核心、默认设置、经验FFHQ开发分支或旧记录。

## 目标与来源

核对新的unit物理测量约定下，FFHQ四任务是否接近原文量级。
不是状态收益或只改值域的消融：恢复公开原学习率/网格/初值/算子合同，
不能复用lr=.01、Karras rho3、legacy_T的26.7013冒充原配置SR结果。

来源：主文§4.1–4.2、Table1/2、Algorithm1：
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114.pdf
补充§1.1–1.2（Adam默认，任务初值）：
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11114-supp.pdf

| Task | 原文PSNR | SSIM | LPIPS | ζ初值 |
| --- | ---: | ---: | ---: | ---: |
| Gaussian61/sigma3 | 26.06 | .757 | .121 | .2 |
| Random Inpainting70% | 27.79 | .813 | .078 | .1 |
| Motion61/intensity.5 | 24.16 | .709 | .141 | .2 |
| Bicubic SR×4 | 26.63 | .768 | .104 | .1 |

这些是1000图均值。单图只作协议/量级检查；1–20图子集均值也不认证原文
1000图复现或统计显著性。不把更低/更高的单图判成整个数据集失败/成功。

## 固定配置与诚实边界

FFHQ模型原权重，256RGB；UNet状态[-1,1]。物理测量
A_u(x)=H((x+1)/2)、Jacobian伴随=.5H^T；unit域加σ=.05 Gaussian，
不裁剪观测。D全步=.2；joint Adam lr=.001/betas(.9,.999)/eps1e-8；
paper15/10/5取整网格30步×10轮、300NFE每task/image；fixed-small DDPM eta1；
db4 level3；保持已有clamp/score no_grad/surrogate OFF/MSE mean/clipnorm1/ζ投影。
无状态、Karras、经验上采样引导、LR扫描、晚期降噪、最终额外采样或GT挑输出。
输出last_opt第10轮更新前，保存PRE与post参数且不混用。

随机修补：一个像素mask共享RGB，Bernoulli遮挡概率.7，每图独立显式seed；
冻住mask，不能在优化/H调用时重抽。噪声加到完整H输出，包括遮挡位置，
其未观测项loss常数不参与梯度；记录此约定，作者内部mask/noise细节未公开。
mask/noise/采样使用显式分离的seed流：采样seed_base+index，
noise再加1000000，mask再加2000000；不把mask和noise从同一重置序列生成。

运动：使用DPS/motionblur/motionblur.py中原论文引用的Kernel生成器家族，
size61/intensity.5，显式numpy kernel seed0；不会回退为共享代码中的简化直线核。
使用生成bitmap的可写numpy拷贝归一化，避免PIL只读array不兼容；不改变路径生成。
Motion真实伴随使用翻转核的conv，zero padding。SR使用已有DPS bicubic Resizer
与其实际autograd VJP伴随，不用legacy T。生成器和mask/核张量hash全记录。

论文未公开：准确图像名单/预处理细节、网格端点取整、DWT层数/边界、
padding、核/噪声seed、clamp/损失约束、输出和指标量化等完整约定。
因此只称public-setting local reference，不称作者完全一致。不能为追分改这些约定。
GT用于合成y和预定最终指标；optimize x0_gt=None，不使用GT调参、停步或选输出。
所有任务同图同采样seed，共享x_T/按步序标准正态；任务间H/y不同，不做跨任务因果归因。

## 预算、门控与采样范围

--image显式指定单图：四任务1200方法NFE。
或--data-dir/--max-images显式指定1–20图：4×300×N预算，10图12000NFE；
没有默认自动扩样。脚本先打印选中manifest和预算，保存所有路径/hash/seed。
没有匹配已完成原配置unit四任务缓存，所以本轮不是重跑Karras/lr=.01。
观测生成、kernel/mask与affine/VJP/内积伴随门控0 UNet NFE；实际FP32相对容差2e-5。
所有四任务首图算子门控必须通过后才加载/调用UNet；后续每图mask/观测同样检查。
模型actual hash/fp32/1000步、原网格、原全步初始化、300访问时间/索引、
有限loss/grad/output和同标准正态步序尾RNG均检查，额外score计数上限严格约束。
失败写failed和实际已用NFE，不把未完成任务纳入均值、不原样自动重试。
本机没有Torch则数值测试skip，服务器先运行测试后真实GPU实验。

## 预期与处理

逐图记录PSNR/SSIM/LPIPS和未量化float PSNR；汇总每任务mean/std，
与论文Table1/2列并排及差值，明确不同样本范围。原生unit物理loss单列。
不预设换域会提升四任务，不把接近26.63的一张图当复现成功。
单图status=SINGLE_IMAGE_CONTEXT_ONLY；小子集=SMALL_SUBSET_NOT_BENCHMARK。
任务明显低于原文量级则回看该任务残差/算子和约定，不重新扫描lr/初值；
接近也只讨论固定配置的小规模确认与状态主线，不自动宣称超越论文。
本轮是复现参考核查，不自动改正式FFHQ开发基座或加入状态。

## 执行/归档

`python -u utils/diag_ffhq_four_tasks_unit.py --image <path> --device cuda --seed-base 1000`
或`--data-dir <dir> --max-images 10 --start-index 0`（明确子集预算）。
新目录results/diag_ffhq_four_tasks_unit/时间戳：run.json、summary.csv、per_image.csv、
config/card、mask/kernel/观测/输出/PRE和post参数/hash、loss、runtime和实际NFE。
保护旧权重、旧观测与结果；未完成不能当作结果。

本机验证：CLI/语法/差异检查通过；本入口4项无依赖合同测试通过，3项
运动核/张量/VJP数值测试因依赖缺失skip。服务器先运行本入口7项测试，
要求无failed、无skip，再启动CUDA。尚无本轮GPU重建结果，不宣称门控已通过。
