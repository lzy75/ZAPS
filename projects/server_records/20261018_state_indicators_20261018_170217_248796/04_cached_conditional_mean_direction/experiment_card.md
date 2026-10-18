# A0b：存档轨迹上的条件均值方向与转向敏感度

日期2026-10-16，Asia/Shanghai。状态implemented，GPU/张量检查待服务器。
父实验：[A0结果](../STATE_TRAJECTORY_RESULT_20261015.md)，代码1af8535；
用户回传observer/archive误差均0，最终26.70129265/.82015914/.16083704。
自动寻找服务器`results/diag_state_trajectory/*/run.json`中的最近completed兼容源，
无PSNR排名；`--trace-dir`可指定精确Records。原始绝对目录尚未回传，不杜撰。
必须复用FFHQ00000/seed1000历史unit-null，禁止自动改用其他配置或重新训练补源。

## 目标、唯一改动与固定条件

原clean相邻增量余弦有28个有效点但预测下一步质量的关联弱；早期裁剪近饱和。
可否定假设：去掉当步新注入噪声后，条件均值更新方向比含噪更新更可解释，
并可能提供比clean增量余弦更有用的后续质量信号。不能保证余弦趋近1或质量提高。
本轮只改变离线指标所用的方向定义；不改调度、步数、ζ/D、eta、H或原重建结果。
保持父Karras rho3、30×10历史优化、lr=.01、db4 level3、fixed-small DDPM eta1、
unit sigma=.05、legacy_bicubic经验T（.5T物理转换）、原clamp与PRE末轮参数。
score权重不加载，不调用UNet，不反传。图像/GT不加载，质量数据复用原逐点离线评价。

## 定义：三个概念分开

原定义z_k=clamp后的Tweedie估计、d_k=z_k-z_(k-1)、c_clean=cos(d_k,d_(k-1))。
含噪c_latent比较x_next-x_input的相邻增量；两者均把RGB全部像素展平。
图像本身的cos(z_k,z_(k-1))受共享图像内容/DC成分主导，不作为曲率指标。

新方向在同一存档输入上定义：

    mean_k = 原ddpm_posterior_step(x_k,z_k,t_k,s_k,eta=0)
    g_k = mean_k + 原guidance_correction_k - x_k
    c_mean_k = cosine(g_k,g_(k-1))

只用eta0读取原helper的均值，不执行新的eta0采样轨迹。
输入x_k仍受过去噪声/裁剪/有限跨度影响，g不是已认证ODE场或全确定性轨迹。
它的优势假设是去掉了当步新噪声，并避免对相邻clean估计做共享端点差分。
范数<=1e-8按缺失处理，不以eps强行定义零方向。

同一方向的两个等价角度读数（不是新增候选参数）：

    tangent_turn = ||g_k/||g_k|| - g_(k-1)/||g_(k-1)|||| = sqrt(2*(1-c_mean))
    turn_per_u = tangent_turn/(u(t_k)-u(t_(k-1)))

小角度theta下1-cos(theta)≈theta²/2，tangent_turn≈|theta|；后者较直观地保留
小角度的线性变化；新mean方向实际用双精度单位向量相减求范数，避免1-cos的
消减误差。它仍可能放大噪声。除Δu校正时间点间距，也会放大密集点误差，
不宣称自动更稳定。对clean/latent同样输出这两种读数以免混淆表达变换和方向改变。
正标量速度归一化不改变余弦，不声称仅除步长就修复角度。
原velocity-turn=||v_k-v_prev||/(||v_k||+||v_prev||)还受速度大小变化影响，
不是纯转向，也不是几何曲率。
严格光滑曲线几何曲率为||a_perp||/||v||²；离散几何曲率还需弧长与可靠切向，
当前随机DDPM样本不能直接套用ODE曲率/截断误差结论。本轮不输出“真实曲率”。

## 预算、身份与门控

新方法NFE=0，新诊断UNet NFE=0，无checkpoint读取、优化、图像质量网络调用。
计算开销仅30次算子/小波/均值和原噪声重建（29次存档生成器随机draw）；记录耗时。
父优化300NFE、上一轮回放90NFE作为历史成本，不称原方法总成本为0。
比较source run hash、原源码hash、观测/PRE/输出hash，匹配全部resolved_config。
首次绑定trajectory.pt的sha256；旧trace未存其hash，所以还需以下数值核对。
alpha_bar复用原hash认证源码`_precompute_diffusion_params(1000)`：float64 beta/cumprod
后float32；1000个点不靠猜测另造。与原每点u误差<=1e-10。
缓存首状态必须与PRE init_noise完全相同，末输出对父输出relative error<=1e-7。
原网格、逆向参数索引、tensor shape/float32/finite必须匹配。
在每个缓存输入上，用真实posterior helper、原ζ/D/H和PRE RNG重构next_x，
逐点relative error<=1e-6；残差与原记录误差<=1e-6+2e-5*norm，clean余弦<=2e-4。
重构末RNG须等于原生成器29次同shape draw后的状态。所有数据写新目录，不覆盖父。
某一门控失败即保存failed并停止，不放宽容差，不补跑UNet，不解释指标。

## 分析和分支

完整30点同时报告三种余弦、各自turn_per_u、随机噪声/均值步范数比、
prior/guidance夹角（仅解释更新组成，非额外控制候选）及原残差/质量读数。
所有cosine与后续ΔPSNR/ΔLPIPS/Δr的Pearson/Spearman/去线性u趋势关联。
预定另看raw clip fraction<=.1源点，保持原相邻配对，不跨筛选缺口拼接时间点。
该阈值来自已看过的图，是探索性分层，不能当独立检验或自动变成在线开关。

支持候选：新mean余弦有效>=24/29，在低裁剪段有可读动态范围，且更大的mean
余弦对应后续ΔPSNR更正/ΔLPIPS更负的证据相对clean较一致；不以某一个相关系数
过线自动宣布成功，不作自相关30点的独立样本p值。正结果最多支持这一图的指标候选。
若方向定义变化没有可解释改善，不继续扫滤波窗口/频段直到本图相关变强；保留
残差主指标与角度辅助诊断，固定少量图/seed确认。若有候选，再固定网格设计ζ
有界响应配对，先证明综合质量收益。单图不能解锁全量部署或宣称方法收敛。
EMA、局部线性拟合、空间低通均可作后续单因素候选，当前不同时引入。

## 命令与结果

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_diag_cached_direction -v
python -u utils/diag_cached_direction.py --inspect
set -o pipefail
python -u utils/diag_cached_direction.py --device cuda | tee ../cached_direction_20261016.log
```

服务器测试应无failed、无skipped；本机无torch，明确标记张量测试未执行。
新结果目录`results/diag_cached_direction/<北京时间戳>/`，保存run.json、directions.csv、
directions.png/svg（matplotlib可用时）、实验卡。回贴30点表、Gates、Summary、Records。
原始结果继续阶段性统一归档，不重复跑优化或上轮90NFE采样。

参考边界：GITS研究ODE采样轨迹，不是本DDPM指标的证明：
https://proceedings.mlr.press/v235/chen24bm.html
SDE自适应求解通常需要专门误差估计，不等同于相邻余弦：
https://arxiv.org/abs/2105.14080
