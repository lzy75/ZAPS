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
