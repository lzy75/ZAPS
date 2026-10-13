# R-parameter-gradient：Gaussian 终端损失的 ζ 优化方向诊断

日期：2026-10-13，Asia/Shanghai。
状态：独立入口已实现；本地仅协议/判读单测，真实 PyTorch/CUDA 数值验证未运行。
用户已确认本方案；不修改 ZAPS 核心、FFHQ、batch 或默认参数。
结论范围：mechanism_only，不是新原文基线或状态收益。

## 1. 目标与新增信息

可否定假设：当前 unroll 的 ζ 反传方向不能可靠反映实际终端测量损失的
局部变化，可能限制参数适配。当前前向引导使用 wavelet B，反向对 eps
stop-gradient；误差可能是近似性质，不自动等于代码错误或违反作者实现。

过去的 x_t 局部 VJP、硬 mask 重建、SR Eq.21 surrogate 已做；不重复。
本次核查的是完整30步 R(zeta,D;xi) 对三个 ζ 的导数，保持 D/噪声不变。
norm=1 裁剪在当前40轮参考未触发，不补跑开关或倍率扫描。

完成父记录：

```text
/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549
  optimized_trace_gaussian_deblur_20261008_181857/
    irregular_15_10_5_last_unroll_state.pt
    guidance_clipping_20261009_142630/audit.json
```

guidance status=complete、原回放与 probe gates 通过才可使用。
core/H/checkpoint/y/image 必须逐一匹配该 audit hash，不用文件名猜测身份。
用户上传记录提交 d2c8122；上传源 HEAD=3ba2a59，不冒充原运行版本。

## 2. 固定条件与唯一扰动

- ImageNet Gaussian；00001.png、seed1001；61×61、sigma3、zero padding。
- 原 y 为 model-domain [-1,1] sigma=.05；不重新生成观测。
- fp32 ImageNet 官方权重；原 alphas、DDPM fixed-small、eta1、x0 clamp。
- 原30步 paper 15/10/5 网格；ζinit=.2、Dinit=.2、db4 level3。
- 恢复第10轮采样前参数，即9次 Adam 更新后，**不是第10次更新后**。
  源训练 lr=.001、Adam默认、MSE mean、norm clip=1、ζ>=1e-6；本次
  不运行 Adam、不投影/裁剪梯度、不重训，只读取未裁剪参数梯度。
- D 保持已学习快照值用于隔离 ζ：不改变正式联合学习规则。
- ζ 探针精确时间 t=999/333/143，按升序 tau 的真实位置索引，不按反向 k。
- h=1e-4/3e-4/1e-3，预先固定；只扰动一个 ζ，正负两臂共享原 x_T、
  CPU及采样卡 RNG；记录 fp32 实际 plus/minus 参数与有效分母。
- 不补 mask、不缩放晚期噪声、不改变方差/H/lr/网格，不挑最佳PSNR。

L=mean((H(raw terminal R)-y)^2)，不在终端物理 loss 前增加 clamp。
AD 用原 fp32 MSE；FD 用同一 fp32 H/残差、double平方及求均值，以降低
损失归约舍入。该数值区别显式记录，不改变模型精度/物理目标。

GT仅对未扰动基线报告原口径 PSNR/SSIM/LPIPS；不评价/选择扰动后最佳图。
各扰动记录 loss/residual/时间/裁剪指纹；不生成替换重建或创新增益表。

## 3. 门控与预算

- 两次未扰动：一条保留参数梯度，一条no_grad；输出relative<=1e-7，
  原时间序列/30模型调用/RNG尾状态一致。AD/no_grad差异不拿来放宽自己。
- 两臂都与完整存档 scalar fingerprints 比较，沿用已验证容差；
  fp32 loss 与存档末轮满足 abs<=1e-8+2e-5*abs(source loss)。
- GPU数量可不同；只恢复原采样卡 RNG，复用完成审计的映射规则。
- 每个扰动结束 RNG 必须与未扰动一致；模型权重梯度不得积累。
- 所有异常停止并写 failed JSON；前后完整恢复 ζ/D，不覆盖源快照。
- 2×30 + 3×3×2×30 = **600新诊断NFE**。全部真实模型调用由hook计数，
  出错的已尝试调用也计入；不是新300NFE方法结果。额外H/归约/反传耗时
  计入秒数，基线 LPIPS 计算不计 score NFE。
- 先只一个不规则网格。Uniform/D方向/新图不自动解锁，也不扫步数。

## 4. 预注册判读

loss floor=max(两次基线double loss差,1e-10)。正负 loss 差须超过10倍
floor才是可分辨信号；至少两个尺度可分辨，所有可分辨尺度同号且
FD斜率相对spread<=.20才称稳定。不能选一个有利h忽略其他尺度。

| 输出类别 | 定义与后续 |
| --- | --- |
| DIRECTION_CONFLICT | 稳定FD与可分辨AD异号：优先追查反向近似，不直接改原文基线 |
| AUTOGRAD_NEAR_ZERO_WITH_FD_SIGNAL | 稳定FD可分辨、AD在noise floor内：检查遗漏敏感度，不当作裁剪问题 |
| LOCAL_AGREEMENT | 稳定、同号、相对误差<=.20：只降低这三处ζ的怀疑，不认证D/全参数 |
| SCALE_MISMATCH | 同号但相对误差>.20：量级问题候选；Adam可抵消部分尺度，不能据此断定PSNR根因 |
| INCONCLUSIVE_* | 信号太小或尺度不稳定：保留基线，不宣称错误/通过，不自动扩大搜索 |
| gate failed | 先修回放/身份/数值接口，不解读梯度或画质 |

## 5. 运行与归档

入口：`utils/diag_zaps_parameter_gradient.py`。
显式命令：

```bash
cd /home/lzy/ZAPS/projects
python -u utils/diag_zaps_parameter_gradient.py \
  --guidance-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/guidance_clipping_20261009_142630 \
  --device cuda \
  --probe-times 999 333 143 \
  --deltas 0.0001 0.0003 0.001
```

新目录：源 optimized_trace 下 parameter_gradient_时间戳，exist_ok=False。
每次调用即保存 run.json，含本卡、完整配置/身份/runtime/实际NFE/指纹；
另存 pairs.csv、summary.csv。complete仅指运行完整，不是梯度一致。

真实GPU结果：待用户运行。假设支持/否定/不确定：未判定。
是否修改核心/FFHQ/batch：不解锁；读结果后先讨论。
