# Gaussian unit测量协议：已完成结果摘要

来源：用户在本聊天回传服务器summary与Decision；不是本机GPU运行。
父入口commit45adf37；完整JSON/张量仍在服务器，下一入口按hash验真。

Run：
`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/gaussian_unit_domain_20261014_133157_715747`

| 预定设置 | PSNR | SSIM | LPIPS | 本轮新增UNet NFE |
| --- | ---: | ---: | ---: | ---: |
| 缓存原model测量域，paper | 19.5987 | .3168 | .6505 | 0 |
| unit物理测量域候选，paper | 21.7256 | .5387 | .3958 | 300 |

精确变化：PSNR+2.1268989012dB，SSIM+.2219597399，LPIPS-.2547061443。
预注册分类`SUPPORTS_GAUSSIAN_MEASUREMENT_PROTOCOL_SENSITIVITY`。

原全步ζ/D初值=.2、joint Adam lr=.001、paper15/10/5、30×10、
fixed-small DDPM eta1、db4 level3、last_opt均不变，无状态/晚期降噪。
算子改A(x)=H((x+1)/2)，Jacobian伴随=.5H^T，保留zero-padding H(1)。
旧标准噪声场复用，但物理unit σ从.025变.05；引导与loss尺度也随协议变化。
因此这是一项成套测量协议对照，不单独隔离了噪声、梯度或Jacobian近似。

三指标共同改善支持测量协议是重要敏感因素；不能认证作者内部约定，
不能与论文22.45单图作配对benchmark均值差值，不能称状态收益或唯一根因。
原19.5987仍保留历史local_reference；21.7256单独标unit协议候选，不覆盖manifest。

下一项已获用户同意：[同unit协议paper/uniform对照](experiment_cards/R_gaussian_unit_uniform.md)。
只补uniform一臂300新NFE，缓存paper无需重跑；不混用旧model域uniform21.5069。
目的为检查值域对齐后调度排序是否改变，不展开新网格/调参搜索。

## 同unit协议调度对照已完成（用户回传）

paper21.7256/.5387/.3958，uniform21.2757/.5571/.4440。
irregular−uniform：PSNR+.4499126355、SSIM−.0183744431、LPIPS−.0482091308。
分类`MIXED_PSNR_AND_PERCEPTUAL_TRADEOFF`：PSNR/LPIPS支持paper，SSIM取舍。
旧model域dPSNR−1.9082，在新unit协议下反转为+.4499；支持测量协议影响
当前图上的调度排序，不是论文FFHQ SR调度表的同任务均值复现。
完整uniform子run路径尚未由用户提供，此段只归档回传摘要，不杜撰路径/hash。

用户已授权下一组[固定30状态/测量域配对](experiment_cards/A_state_domain_paired.md)：
ImageNet unit Gaussian null/state，以及FFHQ两个测量域各null/state。
状态收益只在同域fresh null上计算；不将本节固定网格差记作状态收益。
