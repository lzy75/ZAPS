# R-first-init：收束前的单次初始化消融

日期2026-10-14；R线local_tuning，不是paper-spec原设置。
用户授权有证据的最后验证并同意继续。独立入口；GPU待运行。
不修改核心、FFHQ/batch/defaults，不开启初值/lr/倍率扫描。

## 目标与父证据

原Gaussian paper15/10/5单图19.5987/.3168/.6505；首步幅度缩至同状态
精确梯度范数，冻结路径20.7478/.3951/.5823。再同幅度换精确方向失败
16.5142/.2173/.8160。两项机制诊断不能认证作者设置或数据集均值。
剩余有直接依据的候选：原公式首步ζ初始化过大，10轮lr=.001难以适配。
本次验证较小首步初值在完整联合优化中能否保留收益，不继续精确方向。
不是证明初始化是唯一根因或证明原文不可能复现。

显式父：`.../optimized_trace_gaussian_deblur_20261008_181857/first_guidance_20261013_213249_896043`。
源根为`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549`。
同source/hash的最新completed `first_direction_*`为方向父；可用显式CLI锁定。
父JSON、原快照、权重、H、y、图像身份必须匹配。失败父不得用于解锁。

## 唯一变量和预算

只改t999/index29的初始ζ：`.2*(exact_norm/raw_norm)=.0400298317787`。
其余29个ζ仍.2、全部D初值.2；原wavelet近似/反向/noise/方差/投影/clip
公式不变；ζ/D全程联合学习，没有永久首步乘a，没有精确梯度替换。
初始化有明确旧审计依据，非新GT搜索；仍属自己调参，不填入原文配置栏。

同00001/seed1001、Gaussian61/sigma3/zero padding、原[-1,1]测量sigma=.05、
官方fp32模型、db4 level3、paper30步、DDPM fixed-small eta1、10轮Adam
lr=.001默认参数、MSE mean、norm clip1、ζ>=1e-6、last_opt PRE-final-update。
源noise、x_T及每轮29个draw保持相同。GT训练时不传，仅最终评价。
本次300实际UNet NFE；无额外最终采样或原基线重训/回放。

0 score预检：seed重建x_T，模拟9×29噪声draw，必须与旧epoch10快照的
x_T/CPU和采样GPU RNG逐元素相同。再模拟第10轮29个draw获得预期尾状态。
诊断draw为1个x_T+290个transition，单列耗时/数量，0新UNet，不当作方法。
真正优化前重置seed；原optimize不改。每次score前后RNG守卫、model调用hook
计数；epoch10输入快照和最终RNG再次配对，300时序必须正确。
观察clip仅记录原函数返回的总范数及首ζ梯度，不更改裁剪或优化器参数。
被动Tweedie指纹包装沿用已门控的scope(enabled=False)，保留原计算图。

身份/随机前缀失败立即停止（0 score）；训练中异常保存failed；指标不解读。
CPU协议测试与服务器tensor/GPU验证分开，不声称本机已数值验证。
原父output.pt没记录保存时hash，缓存指标只做兼容性核对，不冒称历史逐元素认证。

## 预注册结果及收束规则

同时相对原19.5987及冻结幅度20.7478报告PSNR/SSIM/LPIPS。
综合收益：对原dPSNR>=.2、dSSIM>=.005、dLPIPS<=-.005。
保留冻结收益：对冻结dPSNR>=-.1、dSSIM>=-.005、dLPIPS<=+.005。

| 结果 | 处理 |
| --- | --- |
| 综合且保留冻结收益 | 支持初始化敏感；可提案冻结明示local_tuned版本，但非原文认证 |
| 综合但未保留冻结收益 | 部分初始化收益；不扫更多初值或加轮数 |
| 无综合收益/指标混合 | 结束本轮单图排查，保留原local_reference，推进状态工作 |
| gate failed | 仅修具体接口/身份，不为追分重跑或宣称方法失败 |

无论正/负都先收束本轮，不自动追加1000步、大规模benchmark、lr或初值扫描。
目前只能说“当前实现未认证复现原文水平”，不能证明“无法达到”；小样本
与作者均值非同样本。后续状态null/refined必须采用同一个明示基座；复现边界
和研究收益分账。若后来有作者代码/新公开细节再重新开放R线，而非永远放弃。

## 运行与记录

```bash
cd /home/lzy/ZAPS/projects
python -u utils/diag_zaps_first_init.py \
  --first-run-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/first_guidance_20261013_213249_896043 \
  --device cuda
```

输出`first_init_时间戳`：run.json、实验卡、epochs.csv、summary.csv、
first_init_last_opt.pt、last_unroll_state.pt。记录PRE/POST参数语义、梯度、
loss、全部身份/runtime/RNG门控和actual NFE；每轮有原optimize进度。
不会覆盖任何父记录。初始化调度文献/状态任务并行但不混入本次设置。
