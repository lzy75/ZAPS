# A82—A84：当前噪声收缩模块直接迁移三套官方宿主

日期2026-11-04；用户明确授权。状态planned，独立A状态/跨求解器开发验证。

## 目标与可否定假设
在MPGD无投影版、DMPS及作者团队公开ΠGDM实现中，只替换原物理修正q为gamma*q，检验当前模块是否带来质量收益，成本是否仍主要来自算子。
不把三套宿主的绝对均值直接相减，不增加新规则/调参/曲率/步长选择。
参照原文PDF、分类及来源见 `reports/20261104_transfer_literature/README.md` 和 `papers.json`。

## 父run/面板/基线身份
仅从已完成A71（72/72）冻结计划取旧图路径：FFHQ00226/00443/00963；不复用旧观测、旧参数或模型结果。均model[-1,1] sigma=.05，seed=1000+ID，观测seed=2000+ID，构造seed=3000+ID。每套H/y独立生成，宿主内两个臂共享。
现有官方ffhq_10m.pt，SHA256=81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa，严格load_state_dict，禁止随机权重fallback/换精度/下载新权重。图像原为256px，不重新resize。
官方源码只下载小型子集，保留原字节与许可。来源固定Git commit：
- A82 KellyYutongHe/mpgd_pytorch，9f94b386ea4184960278d28a8bc2756898db70e2。
- A83 mengxiangming/dmps，91afe2540ac8dd8a7fa133f654560f736c80fc53。
- A84 NVlabs/RED-diff，66482f23e242bb31166c3662002d0a6f9f065030。
每个文件Git blob及SHA256在UPSTREAM验证；仅选择原AST定义，排除无关latent/classifier导入。sampling loop原样执行。

## 三套固定配置与已知边界
A82原x0ddim100，eta=.5，learned_range，clean clamp[-1,1]。两个任务均用原文附录D.1明确的100步scale5；发布SR100脚本4与CLI默认10有差异，显式披露，不通过质量挑选。原antialiasResizer SR4与61核sigma3反射Gaussian；不引入VQGAN。原文写平方残差而发布conditioning使用残差范数；保留原代码范数，不悄悄改成另一方向。
A83原DDPM1000，learned_range，scale1.75。原bicubic反射SVD SR4；Gaussian配置61/intensity3由原main实际构造31tap separable kernel，原截断保留。原代码NumPy缓冲被CUDA索引及NumPy单元素系数与CUDA混算需要仅索引/标量传输适配：float64数值不变，两臂相同，专门门控；不改变数学步骤。原native梯度实际在unconditional posterior sample处计算，保留代码行为，不悄悄修成论文另一版本。
A84公开PGDM原默认50点（980到0均匀）、eta0、awdTrue、grad_term_weight1、cond_awdFalse、mcgFalse。原raw Tweedie预测不裁剪，原伪逆初始化及原xs输出。原sr4是block-average SVD；原deblur_gauss是5tap/sigma25，保留原小奇异值截断。给定观测sigma=.05；原pgdm.py不使用噪声协方差，因此只称公开伪逆实现的含噪本地参考，不能冒称原文公式(7)含噪实现。正结果需后续在完整含噪基线确认。
三套均是本地小面板同实现对照，不是作者benchmark均值复现；停止后不能按分数更换配置/核。

## 唯一变量/公式/插入位置
完全复用A71：r=y-H(m)，omega=max(0,1-M*sigma_y²/||r||²)，gamma=omega*max(0,<r,d>)/||d||²。退化零响应按原策略gamma0，不能追加范围/阈值/超参数。
A82直接在原clean gradient correction q=-scale*grad||y-Hm||/sqrt(alpha_prev)处缩放，d=Hq；原DDIM混合与噪声保留。
A83在原conditioning最终`x_t += q`表达式处缩放；A84在原`xs = ... + grad_term*coeff`表达式处缩放。两者用d=H(q/sqrt(alpha_t))，仅为当前A75同时间identity-response代理，不是真实下一步网络响应。r仍用原模型m，不替换宿主的原方向。
原始corrective表达式仅允许匹配一次。BASE返回原表达式对象/算术不变；ONES经过完整候选感知但强制gamma1，必须与不含hook的原AST逐位相同。
既有正全局幅度仍可能被投影抵消；不声称保留原scale就保留了其实际长度，也不继承后验/流形保证。

## 预算/完整计时/资源
每组3图×2任务×2臂=12runs，共36runs。A82:1200NN、1200clean/operator梯度；A83:12000NN、0反向；A84:600NN、600UNet输入VJP。总13800NN、600输入VJP、1200算子梯度。输入VJP按调用计数，checkpoint内部重算仅含在clock，不冒充免费。
候选每步2次新增H（H(m)、H(q_clean)），三组共13800新增H；无额外网络和VJP。每次载入1真实网络探针，A84再1输入VJP计探针成本。Toy门控不使用真实UNet权重。
A84使用已经验证的原checkpoint兼容input_vjp_parameters：需要参数requires_gradTrue，autograd只取输入梯度，无参数.grad/更新；保留原模型函数，不关闭checkpoint。
complete clock从given y/原生算子重建和SVD、随机初始化到所有采样与CPU输出同步，包括进度保存；模型load/probe、共享观测生成、指标与写盘单列。单次配对时间不是稳定生产均值。
两任务shard，分配GPU>=6GiB，实测run峰值+1GiB再准入；单GPU串行，不结束其他进程。MKL_THREADING_LAYER=GNU在导入前设置。断线安全协调器/状态/归档单独入口；失败记录保留不自动重试，resource_limited只恢复未开始的run。

## 必过门控/判据/正负分支
本机：源码hash、唯一插入AST、原参数/预算、旧面板/workerCLI检查。
服务器：必有Torch/CUDA，不允许skip；Toy完整原生/BASE/ONES逐位同输出、同初始化、CPU/CUDA最终RNG与步点；SHRINK保持RNG/调用数，导数类型/成本正确；实际SR/Gaussian算子状态可复现且梯度有限；DMPS标量/index适配数值一致；真实权重严格载入及输出/输入VJP探针通过。
逐host/task：平均PSNR>=+.1dB，>=2/3图正向，SSIM不降、LPIPS不升，且配对指纹全部通过，才列正向开发候选。否则分别保留指标权衡/负结果/门控错误，不扫描系数救分。
正向后再决定固定倍率/错配消融及新图确认；本轮不加入SAIP/SURE/DTM/新投影。无GPU结果不宣称有效，不保证所有宿主迁移成功。

## 产物与服务器入口
`run_native_ray_transfer_20261104.sh --experiment A82|A83|A84 --gpus 0,1`。
独立根`/data0/lzy/ZAPS-results/diag_native_ray_transfer/A8x_20261104`。
冻结plan、shared observation/operator状态、逐run quality/full-clock/counts、倍率trace、outputPNG/PT、summary/CSV/MD。轻量归档自动push独立results分支，不合master，不下载PT。
