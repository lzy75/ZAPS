# R：unit测量协议下 paper / uniform 单因素对照

日期2026-10-14；状态implemented / GPU pending。用户已明确同意继续本项。
本机新增5项契约测试通过，2项张量测试因无PyTorch跳过；真实GPU尚未运行。

## 1. 目标与去重

问题：修正Gaussian物理测量域之后，原来irregular落后uniform的现象是否仍在？
假设：公开测量协议的差异可能影响两种调度的质量排序；本次允许否定该假设。
父run（用户已回传completed摘要，原始记录由入口验真）：
`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/gaussian_unit_domain_20261014_133157_715747/run.json`。
父paper结果21.7256/.5387/.3958；旧model域paper19.5987、uniform21.5069
只作历史背景，不充当本次unit域uniform的缓存结果。
新增信息：只补unit域uniform一臂，不重复已完成paper，不展开网格/学习率搜索。

## 2. 唯一变量与固定条件

唯一变量是实际时间网格：paper15/10/5改成30点全局均匀整数网格。
`tau_i=round(999*i/29), i=0,...,29`，正整数精确最近取整，严格升序含0/999。
仍从999→0采样；30组ζ/D按新网格升序位置定义，从全部.2重新联合训练，
不是将paper已训练参数移植到uniform。原constructor网格只用于创建30组参数，
运行前仅替换tau；另存resolved_config与实际tau，不能把constructor配置当实际网格。

完全复用父 `measurement_unit.pt`，不重建观测、不新增观测噪声。
UNet/model状态[-1,1]，`A(x)=H((x+1)/2)`、Jacobian伴随`.5H^T`，
unit物理σ=.05，Gaussian61/sigma3/zero-padding，所有源码和权重hash不变。
同00001.png/seed1001；全ζ=.2、D=.2；joint Adam lr=.001/betas(.9,.999)/eps1e-8；
原db4 level3、fixed-small DDPM eta1、Tweedie clamp、score no_grad、surrogate OFF、
MSE mean、clipnorm1、ζ>=1e-6。每轮30×10轮，last_opt第10轮更新前输出，无额外采样。
GT只用于复核预定最终输出指标；不得进入训练、选网格、选epoch/输出或停止。
不改核心、默认配置、FFHQ/batch、状态代码或原值域入口。

## 3. 门控、随机过程与预算

- 父completed及全部门控通过、300NFE/10轮、原初值配置、unit物理协议；
  校验父JSON/原reference链、H/UNet/图片/支持文件/观测/输出/PRE快照hash。
- 当前Torch/CUDA/数值后端flags/GPU型号与父记录一致，否则停在0UNet；
  可见GPU数量变化可以通过明确的采样generator映射处理，不能静默取错GPU。
- 缓存paper输出经旧指标和同域loss复核，rtol2e-5/atol1e-6（loss atol1e-8）。
  仿射算子实际VJP与增量伴随0UNet门控沿用预注册2e-5。
- 同x_T，同epoch/ordinal的标准正态draws（common random numbers）；
  不同跨度的posterior方差和实际注噪随网格变化，不声称同一物理Brownian路径。
- 0UNet RNG预检；实际训练记录300次时间访问、无score耗RNG、epoch10/结束RNG；
  10轮、全参数联合、有限loss/梯度、模型无权重梯度。无真实UNet预热/额外回放。
- 新增仅300方法NFE、0诊断UNet；缓存paper本身历史300NFE仍列出。
  门控失败停止并保留已耗NFE，不自动重跑；不覆盖父目录。

## 4. 运行前判据与后续

统一报告`irregular minus uniform`，避免正负号混乱。
工程阈值不是统计显著性，也不是论文复现成功门槛。

- 若dPSNR>=.1、dSSIM>=-.002、dLPIPS<=.005：支持当前图上irregular有PSNR优势，
  且感知/结构无明显退化；不外推论文均值或全数据排序。
- 若dPSNR<=-.1：uniform仍有PSNR优势，单独标注其SSIM/LPIPS是否也改善；
  说明值域对齐没有恢复当前图上的PSNR排序，不代表所有流程错误。
- 若abs(dPSNR)<.1：视为当前图上PSNR接近；若其他指标取舍，明确报告。
- 其余（例如irregular PSNR更高但感知下降）标mixed，不称综合优胜。
- 全部结果：讨论后才决定另一固定图或状态实验；不自动扩展样本、换lr/初值、
  添加状态、降低晚期噪声。任何单图结果都不能认证作者实现/原文均值复现。

## 5. 操作与归档

`python -u utils/diag_imagenet_gaussian_unit_uniform.py --unit-run-dir <父目录> --device cuda`

父目录下新建 `uniform_comparison_时间戳`，保存card/run.json/summary.csv、
uniform_last_opt.pt、更新前快照、初始配置和实际网格、所有门控与身份。
本机没有PyTorch时张量测试明确skip，服务器先跑测试再跑预训练权重实验。
