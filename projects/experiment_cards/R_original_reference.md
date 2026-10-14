# R-original-reference：原设置本地基线的零新增NFE确认

日期2026-10-14。用户要求先确认原设置ImageNet复现基线，不采用改初值版。
独立只读诊断；核心、初值、默认值、FFHQ、batch均不改动。

## 目标与去重

Gaussian00001/seed1001、ζ初值全.2、D全.2、lr=.001、paper15/10/5、
30×10、fixed-small eta1、db4 level3、原MSE/约束/clamp/反向、last_opt。
已有19.5987/.3168/.6505，原轨迹回放和DPS同观测等检查已完成。
不重训、不重跑DPS、不重做mask/值域/梯度/倍率扫描，不采纳21.3650。
首步改动实验中的raw_baseline缓存仍是原始设置的独立回放，ONLY这份缓存
可用；first_norm_matched及first_init输出均不读取为本次基线。

旧sampler_parity_20261008_220150：两条原网格全30点回放通过；仅t999的
x0比较失败，relative4.088976e-5/max8.726865e-5，原atol/rtol=2e-5。
其余mean_shared_x0/mean/std/next均通过。本次不是原样重跑这个180NFE审计，
而是对已存14个x/epsilon补做“共同x0隔离posterior”和“同alpha64下的代数
等价性”检查，以分开结构问题、系数精度及FP32消减。原失败原样保留，
不把门槛改宽写PASS，不承诺这几e-5误差解释了最终几dB差距。

## 假设、固定输入与预算

可否定假设：原缓存的配置/观测/参数快照及评价仍一致；共同x0下的后验
均值/标准差与实际DPS函数兼容，FP64两种Tweedie写法等价。
父first_guidance_20261013_213249_896043完整completed，来源guidance VJP
审计和原优化trace必须门控通过，原参数方向审计parent/hash链必须兼容。
只读原本地权重hash（不加载UNet）、原H/y/image/core/hash、原PRE快照。
当前缓存probe/output的hash另存；旧保存时未记录hash不能补称历史字节认证。

新增方法NFE=0、诊断UNet NFE=0、optimizer=0、重建回放=0。
从保存的14个x/epsilon直接算标量/张量，unit-noise张量隔离std，无新随机draw。
CPU默认可运行，--tensor-device cuda仅选择张量运算设备，不加载模型。
真实UNet/输入梯度完全不调用；LPIPS只评价固定缓存，不计为diffusion NFE。
GT仅用于缓存指标复核/观测噪声域描述，不做输出/时间步选择。

## 检查与结论限制

1. 公开主参数与manifest锁定；snapshot网格/epoch10参数与旧JSON匹配。
2. current权重/原图/H/y/core与完成父hash相同；DPS源文件与旧审计相同。
3. 14个缓存点：真实core posterior vs DPS q posterior，共用x0，mean/std/
   unit-noise-next仍用旧2e-5 atol/rtol；不是实际噪声路径的新回放。
4. FP64同一alpha下直接式与系数展开式用1e-10 atol/rtol；FP32生产式仍以
   原2e-5报告，允许保留“未通过”而不是改实际算法。FP64通过不验证FP32
   原门控，更不验证score-Jacobian/优化梯度。
5. 只复核raw_baseline的PSNR/SSIM/LPIPS及terminal测量MSE；不重新生成图。
6. 报告实际sigma_y所用值域与[0,1]等效噪声，保留公开协议对齐问题。

若身份/配置/缓存/共同posterior/FP64代数失败：status=failed，停止确认，
只定位具体错误，不通过调初值/降噪/容差追分。不会覆盖历史缓存。
若检查通过：ORIGINAL_LOCAL_CACHE_CONFIRMED_WITH_OPEN_ALIGNMENT_ITEMS，
确认19.5987这份本地参考的缓存与机械组件证据，不代表作者实现认证、
原始FP32 sampler总PASS、整个ImageNet数据集均值或所有环节完全正确。
保留近似反向尺度差异、物理值域、作者未公开边界/裁剪/指标约定的清单。
输出后先复核这些未结项，再判断能否固定该本地参考开展A线；不自动解锁GPU。

## 操作与记录

独立入口utils/diag_imagenet_original_reference.py，输入已完成first-guidance目录。
```bash
cd /home/lzy/ZAPS/projects
python -u utils/diag_imagenet_original_reference.py \
  --first-run-dir /home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/first_guidance_20261013_213249_896043 \
  --tensor-device cpu
```

新目录original_reference_audit_时间戳内保存audit.json和本卡。
打印原配置/旧失败点/缓存机制结果/最终指标/噪声域/开放限制/零NFE。
原19.5987基线manifest不覆盖；先前初值/状态负结果保留为旁支历史。
